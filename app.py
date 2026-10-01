import os
import io
import json
import re
import math
import time
import PyPDF2
import PIL.Image
from flask import Flask, render_template, request, jsonify
from google import genai
from groq import Groq
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

# ---------- API CLIENTS ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set (Groq fallback disabled).")

# ---------- MODEL CONFIG ----------
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

MAX_CHARS_PER_CHUNK = 18000
MAX_PDF_PAGES = 30
MAX_IMAGE_DIM = 1600
RETRIES_PER_MODEL = 2
BACKOFF_BASE = 1.2

# ---------- GLOBAL JSON ERROR HANDLERS ----------
@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({"error": "File too large. Max upload is 25MB."}), 413

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Route not found."}), 404

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

def is_retryable_error(exc):
    msg = str(exc).lower()
    retryable = ["503", "unavailable", "overloaded", "high demand",
                 "429", "rate limit", "quota exceeded",
                 "500", "502", "504", "internal", "timeout", "deadline"]
    return any(k in msg for k in retryable)

def validate_quiz(quiz):
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

# ---------- GEMINI PROVIDER ----------
def call_gemini_once(prompt, model_name, image=None):
    contents = [prompt]
    if image is not None:
        contents.append(image)
    response = gemini_client.models.generate_content(model=model_name, contents=contents)
    raw = getattr(response, "text", None)
    if not raw:
        try:
            raw = response.candidates[0].content.parts[0].text
        except Exception:
            raw = ""
    return validate_quiz(json.loads(clean_json(raw)))

def call_gemini(prompt, image=None):
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[Gemini] Trying {model_name} attempt {attempt+1}")
                return call_gemini_once(prompt, model_name, image=image)
            except Exception as e:
                last_error = e
                if not is_retryable_error(e):
                    print(f"[Gemini] Non-retryable on {model_name}: {str(e)[:150]}")
                    break
                if attempt < RETRIES_PER_MODEL:
                    wait = BACKOFF_BASE ** attempt
                    print(f"[Gemini] Retry in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    break
    raise RuntimeError(f"All Gemini models failed. Last: {str(last_error)[:200]}")

# ---------- GROQ PROVIDER (BACKUP) ----------
def call_groq_once(prompt, model_name):
    chat = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert exam writer. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=0.7,
    )
    raw = chat.choices[0].message.content
    return validate_quiz(json.loads(clean_json(raw)))

def call_groq(prompt):
    last_error = None
    for model_name in GROQ_MODELS:
        for attempt in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[Groq] Trying {model_name} attempt {attempt+1}")
                return call_groq_once(prompt, model_name)
            except Exception as e:
                last_error = e
                if not is_retryable_error(e):
                    print(f"[Groq] Non-retryable on {model_name}: {str(e)[:150]}")
                    break
                if attempt < RETRIES_PER_MODEL:
                    wait = BACKOFF_BASE ** attempt
                    print(f"[Groq] Retry in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    break
    raise RuntimeError(f"All Groq models failed. Last: {str(last_error)[:200]}")

# ---------- UNIFIED CALL WITH FALLBACK ----------
def call_ai(prompt, image=None):
    gemini_error = None
    if gemini_client:
        try:
            return call_gemini(prompt, image=image)
        except Exception as e:
            gemini_error = e
            print(f"[Fallback] Gemini exhausted: {str(e)[:200]}")

    if groq_client and image is None:
        try:
            print("[Fallback] Switching to Groq...")
            return call_groq(prompt)
        except Exception as e:
            print(f"[Fallback] Groq also failed: {str(e)[:200]}")
            raise RuntimeError(
                f"Both providers failed. Gemini: {str(gemini_error)[:120]} | Groq: {str(e)[:120]}"
            )

    if image is not None and gemini_error:
        raise RuntimeError(
            f"Image analysis failed and Groq does not support images. "
            f"Gemini error: {str(gemini_error)[:200]}"
        )

    raise RuntimeError("No AI provider available. Check your API keys.")

def generate_quiz_smart(text, num_questions, difficulty, image=None):
    meta = {"chunks": 0, "original_length": len(text), "provider": "unknown"}

    if image is not None:
        prompt = build_prompt(
            text if text else "Analyze the attached image and generate quiz questions from it.",
            num_questions, difficulty
        )
        quiz = call_ai(prompt, image=image)
        meta["chunks"] = 1
        meta["provider"] = "gemini (vision)"
        return quiz[:num_questions], meta

    if len(text) <= MAX_CHARS_PER_CHUNK:
        prompt = build_prompt(text, num_questions, difficulty)
        quiz = call_ai(prompt)
        meta["chunks"] = 1
        meta["provider"] = "auto"
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
            all_questions.extend(call_ai(prompt))
        except Exception as e:
            print(f"Chunk failed: {e}")
            continue
    if not all_questions:
        raise ValueError("All chunks failed. Try a smaller file.")
    meta["provider"] = "auto (chunked)"
    return all_questions[:num_questions], meta

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/debug")
def debug():
    info = {
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY),
        "gemini_models": GEMINI_MODELS,
        "groq_models": GROQ_MODELS,
    }
    if gemini_client:
        try:
            info["available_gemini_models"] = [m.name for m in gemini_client.models.list()]
        except Exception as e:
            info["gemini_error"] = str(e)
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

        if not gemini_client and not groq_client:
            return jsonify({"error": "No AI provider configured. Add GEMINI_API_KEY or GROQ_API_KEY."}), 500

        quiz, meta = generate_quiz_smart(text, num_questions, difficulty, image=image_obj)

        return jsonify({
            "quiz": quiz,
            "meta": {
                "chunks": meta["chunks"],
                "original_length": meta["original_length"],
                "file_info": file_info,
                "returned": len(quiz),
                "provider": meta.get("provider", "auto")
            }
        })

    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Max upload is 25MB."}), 413
    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI returned malformed JSON: {str(e)}"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
