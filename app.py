import os
import sys
import json
import re
import math
import time
import traceback
import gc

from flask import Flask, render_template, request, jsonify

# ---------- BOOT LOGGING ----------
print("=" * 60, flush=True)
print("[BOOT] QuizGenius starting", flush=True)
print(f"[BOOT] Python {sys.version.split()[0]}", flush=True)
print(f"[BOOT] GEMINI_API_KEY: {'SET' if os.environ.get('GEMINI_API_KEY') else 'MISSING'}", flush=True)
print(f"[BOOT] GROQ_API_KEY: {'SET' if os.environ.get('GROQ_API_KEY') else 'MISSING'}", flush=True)
print("=" * 60, flush=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

MAX_CHARS_PER_CHUNK = 8000
MAX_PDF_PAGES = 10
MAX_QUESTIONS = 30
RETRIES_PER_MODEL = 1

# ---------- LAZY CLIENTS ----------
_gemini = None
_groq = None

def gemini_client():
    global _gemini
    if _gemini is None and GEMINI_API_KEY:
        try:
            from google import genai
            _gemini = genai.Client(api_key=GEMINI_API_KEY)
            print("[INIT] Gemini ready", flush=True)
        except Exception as e:
            print(f"[INIT] Gemini failed: {e}", flush=True)
            traceback.print_exc()
    return _gemini

def groq_client():
    global _groq
    if _groq is None and GROQ_API_KEY:
        try:
            from groq import Groq
            _groq = Groq(api_key=GROQ_API_KEY)
            print("[INIT] Groq ready", flush=True)
        except Exception as e:
            print(f"[INIT] Groq failed: {e}", flush=True)
            traceback.print_exc()
    return _groq

# ---------- ERROR HANDLERS ----------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 15MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_all(e):
    print("[ERROR] Unhandled exception:", flush=True)
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

# ---------- HELPERS ----------
def clean_json(raw):
    if not raw:
        return "[]"
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    i, j = raw.find("["), raw.rfind("]")
    if i != -1 and j != -1:
        raw = raw[i:j + 1]
    return raw.strip()

def validate_quiz(q):
    if not isinstance(q, list):
        raise ValueError("Not a list")
    out = []
    for item in q:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("question", "options", "correct")):
            continue
        if not isinstance(item["options"], list) or len(item["options"]) != 4:
            continue
        if not isinstance(item["correct"], int) or not (0 <= item["correct"] <= 3):
            continue
        item["options"] = [str(x) for x in item["options"]]
        item["question"] = str(item["question"])
        item["explanation"] = str(item.get("explanation", ""))
        out.append(item)
    if not out:
        raise ValueError("No valid questions")
    return out

def build_prompt(text, n, diff):
    dg = {"easy": "definitions and simple recall",
          "medium": "understanding and application",
          "hard": "deep analysis and edge cases"}
    return f"""Generate exactly {n} multiple-choice questions from the text below.

Difficulty: {diff.upper()} ({dg.get(diff, 'understanding')})

Rules:
- Each question: exactly 4 options, 1 correct
- Include a short explanation
- Return ONLY a valid JSON array, no markdown, no preamble

Format:
[{{"question":"...", "options":["A","B","C","D"], "correct":0, "explanation":"..."}}]

Text:
{text}
"""

def split_text(text, limit):
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for para in re.split(r'\n\s*\n', text):
        if len(cur) + len(para) + 2 <= limit:
            cur += para + "\n\n"
        else:
            if cur:
                chunks.append(cur.strip())
            if len(para) > limit:
                for s in re.split(r'(?<=[.!?])\s+', para):
                    if len(cur) + len(s) + 1 <= limit:
                        cur += s + " "
                    else:
                        if cur:
                            chunks.append(cur.strip())
                        cur = s + " "
                if cur:
                    chunks.append(cur.strip())
                    cur = ""
            else:
                cur = para + "\n\n"
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 100]

def retryable(e):
    m = str(e).lower()
    return any(k in m for k in ["503", "unavailable", "overloaded", "429",
                                 "rate limit", "500", "502", "504", "timeout", "deadline"])

# ---------- GEMINI ----------
def gemini_call(prompt, model, image_bytes=None, image_mime=None):
    c = gemini_client()
    if not c:
        raise RuntimeError("Gemini not configured")
    contents = [prompt]
    if image_bytes and image_mime:
        from google.genai import types
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
    r = c.models.generate_content(model=model, contents=contents)
    raw = getattr(r, "text", None) or ""
    if not raw:
        try:
            raw = r.candidates[0].content.parts[0].text
        except Exception:
            pass
    return validate_quiz(json.loads(clean_json(raw)))

def gemini_all(prompt, image_bytes=None, image_mime=None):
    last = None
    for m in GEMINI_MODELS:
        for a in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[Gemini] {m} attempt {a+1}", flush=True)
                return gemini_call(prompt, m, image_bytes, image_mime)
            except Exception as e:
                last = e
                print(f"[Gemini] {m} failed: {str(e)[:120]}", flush=True)
                if not retryable(e):
                    break
                if a < RETRIES_PER_MODEL:
                    time.sleep(1.2 ** a)
    raise RuntimeError(f"Gemini exhausted: {str(last)[:150]}")

# ---------- GROQ ----------
def groq_call(prompt, model):
    c = groq_client()
    if not c:
        raise RuntimeError("Groq not configured")
    r = c.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an exam writer. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0.7,
    )
    raw = r.choices[0].message.content
    return validate_quiz(json.loads(clean_json(raw)))

def groq_all(prompt):
    last = None
    for m in GROQ_MODELS:
        for a in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[Groq] {m} attempt {a+1}", flush=True)
                return groq_call(prompt, m)
            except Exception as e:
                last = e
                print(f"[Groq] {m} failed: {str(e)[:120]}", flush=True)
                if not retryable(e):
                    break
                if a < RETRIES_PER_MODEL:
                    time.sleep(1.2 ** a)
    raise RuntimeError(f"Groq exhausted: {str(last)[:150]}")

def call_ai(prompt, image_bytes=None, image_mime=None):
    ge = None
    if gemini_client():
        try:
            return gemini_all(prompt, image_bytes, image_mime)
        except Exception as e:
            ge = e
            print(f"[Fallback] Gemini down: {str(e)[:120]}", flush=True)
    if groq_client() and not image_bytes:
        try:
            print("[Fallback] Trying Groq", flush=True)
            return groq_all(prompt)
        except Exception as e:
            raise RuntimeError(f"Both providers failed. G: {str(ge)[:80]} | Q: {str(e)[:80]}")
    if image_bytes and ge:
        raise RuntimeError(f"Image needs Gemini, which failed: {str(ge)[:150]}")
    raise RuntimeError("No AI provider configured")

# ---------- FILE PROCESSING ----------
def process_file(file):
    if not file or not file.filename:
        return "", None, None, ""
    name = file.filename.lower()

    if name.endswith(".pdf"):
        import PyPDF2
        try:
            reader = PyPDF2.PdfReader(file)
            total = len(reader.pages)
            n = min(total, MAX_PDF_PAGES)
            parts = []
            for i in range(n):
                try:
                    parts.append(reader.pages[i].extract_text() or "")
                except Exception as e:
                    print(f"[PDF] page {i} err: {e}", flush=True)
            text = "\n\n".join(parts)
            info = f"PDF ({n}/{total} pages)"
            del reader, parts
            gc.collect()
            return text, None, None, info
        finally:
            try:
                file.close()
            except Exception:
                pass

    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        raw = file.read()
        ext = name.rsplit(".", 1)[-1]
        mime = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp"}[ext]
        info = f"Image ({ext.upper()})"
        try:
            file.close()
        except Exception:
            pass
        return "", raw, mime, info

    raise ValueError("Unsupported file type")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/debug")
def debug():
    return jsonify({
        "gemini_key": bool(GEMINI_API_KEY),
        "groq_key": bool(GROQ_API_KEY),
        "gemini_models": GEMINI_MODELS,
        "groq_models": GROQ_MODELS,
    })

@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = (request.form.get("text") or "").strip()
        try:
            n = int(request.form.get("num_questions", 5))
        except Exception:
            n = 5
        diff = (request.form.get("difficulty") or "medium").lower()
        file = request.files.get("file")

        n = max(1, min(n, MAX_QUESTIONS))

        ftext, img_bytes, img_mime, info = process_file(file)
        if ftext:
            text = (text + "\n" + ftext).strip()

        if len(text) < 50 and not img_bytes:
            return jsonify({"error": "Need at least a paragraph or an image"}), 400

        if not gemini_client() and not groq_client():
            return jsonify({"error": "No API key configured"}), 500

        if img_bytes:
            quiz = call_ai(build_prompt(text or "Generate a quiz from this image.", n, diff),
                           img_bytes, img_mime)
            provider, chunks = "gemini (vision)", 1
        elif len(text) <= MAX_CHARS_PER_CHUNK:
            quiz = call_ai(build_prompt(text, n, diff))
            provider, chunks = "auto", 1
        else:
            chunks_list = split_text(text, MAX_CHARS_PER_CHUNK)
            per = max(1, math.ceil(n / len(chunks_list)))
            quiz = []
            for ch in chunks_list:
                if len(quiz) >= n:
                    break
                try:
                    quiz.extend(call_ai(build_prompt(ch, per, diff)))
                except Exception as e:
                    print(f"[chunk] fail: {e}", flush=True)
            if not quiz:
                raise ValueError("All chunks failed")
            provider, chunks = "auto (chunked)", len(chunks_list)

        quiz = quiz[:n]

        img_bytes = None
        text = None
        gc.collect()

        return jsonify({
            "quiz": quiz,
            "meta": {
                "chunks": chunks,
                "returned": len(quiz),
                "file_info": info,
                "provider": provider
            }
        })

    except Exception as e:
        print("[ERROR] generate route:", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
