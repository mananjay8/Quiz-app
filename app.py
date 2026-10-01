import os
import io
import json
import re
import math
import PyPDF2
import PIL.Image
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")

# Try these in order until one works
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Cache the working model after first success
_working_model = None

MAX_CHARS_PER_CHUNK = 18000
MAX_PDF_PAGES = 30
MAX_IMAGE_DIM = 1600


# ---------- GLOBAL JSON ERROR HANDLERS ----------
@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({"error": "File too large. Max upload is 25MB."}), 413


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.errorhandler(Exception)
def handle_any(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"Server error: {type(e).__name__}: {str(e)}"}), 500


# ---------- HELPERS ----------
def clean_json(raw):
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
5. Return ONLY a valid JSON array. No markdown, no code fences, no preamble.

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


def get_working_model():
    """Return the first model that works. Caches the result."""
    global _working_model
    if _working_model:
        return _working_model
    if not client:
        raise RuntimeError("Gemini client not initialized.")
    try:
        available = [m.name for m in client.models.list()]
    except Exception as e:
        print(f"Could not list models: {e}")
        available = []
    # Try each candidate
    for cand in MODEL_CANDIDATES:
        full_name = f"models/{cand}"
        if not available or full_name in available or cand in available:
            _working_model = cand
            print(f"Using model: {cand}")
            return cand
    # If nothing matched, use the first candidate anyway
    _working_model = MODEL_CANDIDATES[0]
    return _working_model


def call_gemini(prompt, image=None):
    """Single Gemini call using the CORRECT SDK method."""
    model_name = get_working_model()

    # Build contents: string + optional image
    contents = [prompt]
    if image is not None:
        contents.append(image)

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )

    # Extract text
    raw = getattr(response, "text", None)
    if not raw:
        # Fallback: try candidates[0].content.parts[0].text
        try:
            raw = response.candidates[0].content.parts[0].text
        except Exception:
            raw = ""

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
        raise ValueError("No valid questions returned. Raw: " + cleaned[:200])

    return valid


def generate_quiz_smart(text, num_questions, difficulty, image=None):
    meta = {"chunks": 0, "original_length": len(text)}

    if image is not None:
        prompt = build_prompt(
            text if text else "Analyze the attached image and generate quiz questions from it.",
            num_questions, difficulty
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
    questions_per_chunk = max(1, math.ceil(num_questions / len(chunks)))
    all_questions = []
    for chunk in chunks:
        if len(all_questions) >= num_questions:
            break
        try:
            prompt = build_prompt(chunk, questions_per_chunk, difficulty)
            all_questions.extend(call_gemini(prompt))
        except Exception as e:
            print(f"Chunk failed: {e}")
            continue
    if not all_questions:
        raise ValueError("All chunks failed. Try a smaller file.")
    return all_questions[:num_questions], meta


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/debug")
def debug():
    """Visit /debug to see which models your API key can access."""
    info = {
        "api_key_set": bool(GEMINI_API_KEY),
        "working_model": _working_model,
        "candidates": MODEL_CANDIDATES,
    }
    if client:
        try:
            models = [m.name for m in client.models.list()]
            info["available_models"] = models
        except Exception as e:
            info["available_models_error"] = f"{type(e).__name__}: {str(e)}"
    return jsonify(info)


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
                        pdf_text += (reader.pages[i].extract_text() or "") + "\n\n"
                    text = (text + "\n" + pdf_text).strip()
                    file_info = f"PDF ({pages_to_read}/{total_pages} pages)"
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
                "returned": len(quiz),
                "model_used": _working_model
            }
        })

    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Max upload is 25MB."}), 413
    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI returned malformed JSON: {str(e)}"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Generation failed: {type(e).__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
