import os
import io
import json
import re
import math
import PyPDF2
import PIL.Image
from flask import Flask, render_template, request, jsonify
from google import genai
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

# Allow up to 25MB uploads
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")

MODEL_NAME = "gemini-3.8-flash"

# Limits for the "smart agent"
MAX_CHARS_PER_CHUNK = 18000
MAX_PDF_PAGES = 30
MAX_IMAGE_DIM = 1600


def clean_json(raw):
    """Aggressively clean Gemini output to extract a JSON array."""
    if not raw:
        return "[]"
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return raw.strip()


def split_text(text, max_chars):
    """Split text into chunks at paragraph boundaries when possible."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    paragraphs = re.split(r'\n\s*\n', text)

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current += sent + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent + " "
                if current:
                    chunks.append(current.strip())
                    current = ""
            else:
                current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 100]


def build_prompt(text, num_questions, difficulty):
    difficulty_guide = {
        "easy": "Focus on definitions and simple recall.",
        "medium": "Test understanding and application.",
        "hard": "Test deep understanding, edge cases, and tricky distinctions."
    }
    return f"""You are an expert exam writer creating a quiz.

Generate exactly {num_questions} multiple-choice questions from the material below.

DIFFICULTY: {difficulty.upper()} - {difficulty_guide.get(difficulty, difficulty_guide["medium"])}

STRICT RULES:
1. Each question must have exactly 4 options.
2. Exactly ONE option is correct.
3. Include a short explanation for why the correct answer is right.
4. Do not repeat the same concept across questions.
5. Questions must be self-contained.
6. Return ONLY a valid JSON array. No markdown fences, no preamble, no trailing text.

Format:
[
  {{
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 2,
    "explanation": "Brief reason why this is correct."
  }}
]

The "correct" field is the 0-based index of the right answer (0-3).

MATERIAL:
{text}
"""


def extract_output_text(interaction):
    """Try multiple ways to read text out of the interactions response."""
    for attr in ("output_text", "text"):
        val = getattr(interaction, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    output = getattr(interaction, "output", None)
    if output is not None:
        if isinstance(output, str):
            return output
        text_attr = getattr(output, "text", None)
        if isinstance(text_attr, str):
            return text_attr
    raise ValueError("Could not read response text from Gemini.")


def call_gemini(prompt, image=None):
    """Single Gemini call. Returns a validated list of questions or raises."""
    input_parts = [{"type": "text", "text": prompt}]
    if image is not None:
        input_parts.append({"type": "image", "image": image})

    # No response_format — we rely on the prompt + clean_json.
    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=input_parts
    )

    raw = extract_output_text(interaction)
    cleaned = clean_json(raw)
    quiz = json.loads(cleaned)

    if not isinstance(quiz, list):
        raise ValueError("Response is not a list.")

    valid = []
    for q in quiz:
        if not isinstance(q, dict):
            continue
        if not all(k in q for k in ("question", "options", "correct")):
            continue
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            continue
        if not isinstance(q["correct"], int) or not (0 <= q["correct"] <= 3):
            continue
        q["options"] = [str(o) for o in q["options"]]
        q["question"] = str(q["question"])
        q["explanation"] = str(q.get("explanation", ""))
        valid.append(q)

    if not valid:
        raise ValueError("No valid questions returned.")

    return valid


def generate_quiz_smart(text, num_questions, difficulty, image=None):
    """
    The 'big agent' that handles large inputs by chunking.
    Returns (quiz_list, meta_dict).
    """
    meta = {"chunks": 0, "original_length": len(text), "truncated": False}

    if image is not None:
        prompt = build_prompt(
            text if text else "Analyze the attached image and generate quiz questions from it.",
            num_questions,
            difficulty
        )
        quiz = call_gemini(prompt, image=image)
        meta["chunks"] = 1
        return quiz[:num_questions], meta

    if len(text) <= MAX_CHARS_PER_CHUNK:
        prompt = build_prompt(text, num_questions, difficulty)
        quiz = call_gemini(prompt)
        meta["chunks"] = 1
        return quiz[:num_questions], meta

    chunks = split_text(text, MAX_CHARS_PER_CHUNK)
    meta["chunks"] = len(chunks)
    meta["truncated"] = True

    questions_per_chunk = max(1, math.ceil(num_questions / len(chunks)))

    all_questions = []
    for chunk in chunks:
        if len(all_questions) >= num_questions:
            break
        try:
            prompt = build_prompt(chunk, questions_per_chunk, difficulty)
            questions = call_gemini(prompt)
            all_questions.extend(questions)
        except Exception as e:
            print(f"Chunk failed: {e}")
            continue

    if not all_questions:
        raise ValueError("All chunks failed. Try a smaller file or simpler text.")

    return all_questions[:num_questions], meta


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = request.form.get("text", "").strip()
        num_questions = int(request.form.get("num_questions", 5))
        difficulty = request.form.get("difficulty", "medium").lower()
        file = request.files.get("file")

        num_questions = max(1, min(num_questions, 50))

        image_obj = None
        file_info = ""

        if file and file.filename:
            filename = file.filename.lower()
            try:
                if filename.endswith(".pdf"):
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    pages_to_read = min(total_pages, MAX_PDF_PAGES)
                    pdf_text = ""
                    for i in range(pages_to_read):
                        pdf_text += reader.pages[i].extract_text() or ""
                        pdf_text += "\n\n"
                    text = (text + "\n" + pdf_text).strip()
                    file_info = f"PDF ({pages_to_read}/{total_pages} pages)"
                    if total_pages > MAX_PDF_PAGES:
                        file_info += " - truncated"
                elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    img = PIL.Image.open(file)
                    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
                    if img.mode in ("RGBA", "P", "LA"):
                        img = img.convert("RGB")
                    image_obj = img
                    file_info = f"Image ({filename.rsplit('.', 1)[-1].upper()})"
                else:
                    return jsonify({"error": "Unsupported file type. Use PDF, PNG, JPG, or WEBP."}), 400
            except Exception as e:
                return jsonify({"error": f"Could not read file: {str(e)}"}), 400

        if len(text) < 50 and image_obj is None:
            return jsonify({"error": "Please provide at least a paragraph of text or an image."}), 400

        if not client:
            return jsonify({"error": "Server is missing its API key."}), 500

        quiz, meta = generate_quiz_smart(text, num_questions, difficulty, image=image_obj)

        return jsonify({
            "quiz": quiz,
            "meta": {
                "chunks": meta["chunks"],
                "original_length": meta["original_length"],
                "file_info": file_info,
                "returned": len(quiz)
            }
        })

    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Max upload is 25MB."}), 413
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned malformed JSON. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({"error": "File too large. Max upload is 25MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
