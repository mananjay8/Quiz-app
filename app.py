import os
import sys
import json
import re
import math
import time
import traceback
import gc

from flask import Flask, render_template, request, jsonify

print("=" * 60, flush=True)
print("[BOOT] QuizGenius bilingual starting", flush=True)
print(f"[BOOT] Python {sys.version.split()[0]}", flush=True)
print(f"[BOOT] GEMINI: {'SET' if os.environ.get('GEMINI_API_KEY') else 'MISSING'}", flush=True)
print(f"[BOOT] GROQ: {'SET' if os.environ.get('GROQ_API_KEY') else 'MISSING'}", flush=True)
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
]

# Fast models specifically for the /explain endpoint (smaller tasks)
FAST_GEMINI = ["gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
FAST_GROQ = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

MAX_CHARS_PER_CHUNK = 5500
QUESTIONS_PER_CHUNK = 8
MAX_PDF_PAGES = 10
MIN_QUESTIONS = 25
MAX_QUESTIONS = 100
RETRIES_PER_MODEL = 1

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


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 15MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_all(e):
    print("[ERROR] Unhandled:", flush=True)
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


# ---------------- HELPERS ----------------
def clean_json(raw):
    if not raw:
        return "[]"
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    i = raw.find("[")
    j = raw.rfind("]")
    if i != -1 and j != -1:
        raw = raw[i:j + 1]
    else:
        i = raw.find("{")
        j = raw.rfind("}")
        if i != -1 and j != -1:
            raw = raw[i:j + 1]
    return raw.strip()


def validate_quiz(q):
    """Ensures bilingual fields exist. Falls back to English if Hindi missing."""
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

        item["question"] = str(item["question"])
        item["options"] = [str(x) for x in item["options"]]
        item["explanation"] = str(item.get("explanation", ""))

        item["question_hi"] = str(item.get("question_hi") or item["question"])
        opts_hi = item.get("options_hi")
        if not isinstance(opts_hi, list) or len(opts_hi) != 4:
            opts_hi = list(item["options"])
        item["options_hi"] = [str(x) for x in opts_hi]
        item["explanation_hi"] = str(item.get("explanation_hi") or item["explanation"])

        out.append(item)
    if not out:
        raise ValueError("No valid questions")
    return out


def split_text(text, limit):
    """Split on paragraph boundaries. Falls back to sentence splitting."""
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
    return [c for c in chunks if len(c) > 80]


def retryable(e):
    m = str(e).lower()
    return any(k in m for k in ["503", "unavailable", "overloaded", "429",
                                 "rate limit", "500", "502", "504", "timeout", "deadline"])


def recommend_questions(text_len, has_image=False):
    """Suggest a question count based on content length."""
    if has_image:
        return 25
    if text_len < 1500:
        return 25
    if text_len < 4000:
        return 30
    if text_len < 9000:
        return 40
    if text_len < 18000:
        return 55
    if text_len < 35000:
        return 70
    if text_len < 60000:
        return 85
    return 100


# ---------------- PROMPTS ----------------
def build_prompt(text, n, diff):
    dg = {
        "easy": "definitions and simple recall",
        "medium": "understanding and application",
        "hard": "deep analysis, edge cases, tricky distinctions"
    }
    return f"""You are an expert bilingual exam writer for Indian students.

Generate exactly {n} multiple-choice questions from the SOURCE MATERIAL below.

═══ LANGUAGE RULES (MOST IMPORTANT) ═══
- Every question, every option, and every explanation MUST appear in BOTH English AND Hindi (Devanagari script).
- This applies REGARDLESS of the language of the source material.
- If source is English → translate questions to Hindi.
- If source is Hindi → translate questions to English.
- Hindi MUST be in Devanagari (हिंदी), NEVER romanized ("aap kaise hain" is WRONG).
- Keep technical terms accurate: "photosynthesis" ↔ "प्रकाश संश्लेषण".

═══ QUESTION RULES ═══
- Difficulty: {diff.upper()} ({dg.get(diff, 'understanding')})
- Each question has exactly 4 options, exactly 1 is correct.
- Include a short explanation for the correct answer.
- Do NOT repeat the same concept across questions.
- Make questions test understanding, not just memorization.

═══ OUTPUT RULES ═══
- Return ONLY a valid JSON array. No markdown, no code fences, no preamble.

JSON shape (exactly this):
[
  {{
    "question": "English question?",
    "question_hi": "हिंदी में प्रश्न?",
    "options": ["A", "B", "C", "D"],
    "options_hi": ["अ", "ब", "स", "द"],
    "correct": 2,
    "explanation": "Why the correct answer is right.",
    "explanation_hi": "सही उत्तर क्यों सही है।"
  }}
]

The "correct" field is the 0-based index (0-3) — same index for English and Hindi options.

SOURCE MATERIAL:
{text}
"""


def build_explain_prompt(question_en, question_hi, wrong_en, correct_en, correct_hi):
    return f"""A student answered a quiz question incorrectly. Teach them the underlying concept from absolute basics.

Question (EN): {question_en}
Question (HI): {question_hi}
Student's wrong answer: {wrong_en}
Correct answer (EN): {correct_en}
Correct answer (HI): {correct_hi}

═══ TEACHING STYLE ═══
- Imagine explaining to a curious 10-year-old child.
- Use a real-life analogy (cricket, cooking, school, daily life, etc.).
- Break into 3-5 short simple points.
- Give one clear concrete example.
- Add a memory trick (mnemonic) if possible.
- End with a short encouraging line.
- Do NOT be condescending. Be warm and clear.

═══ LANGUAGE RULES ═══
- Provide ALL content in BOTH English AND Hindi (Devanagari).
- Hindi MUST be in Devanagari script, never romanized.

═══ OUTPUT RULES ═══
- Return ONLY valid JSON. No markdown fences. No extra text.

JSON shape (exactly this):
{{
  "topic": "Short topic name in English",
  "topic_hi": "विषय का नाम हिंदी में",
  "why_wrong": "One line explaining why the chosen answer is wrong.",
  "why_wrong_hi": "चुना गया उत्तर क्यों गलत है, एक पंक्ति में।",
  "analogy": "A simple everyday analogy that makes the concept click.",
  "analogy_hi": "सरल उदाहरण जो अवधारणा को स्पष्ट करे।",
  "basic_points": ["Point 1", "Point 2", "Point 3"],
  "basic_points_hi": ["बिंदु 1", "बिंदु 2", "बिंदु 3"],
  "example": "One concrete example.",
  "example_hi": "एक ठोस उदाहरण।",
  "memory_trick": "A simple memory trick or shortcut.",
  "memory_trick_hi": "याद रखने की सरल ट्रिक।",
  "encouragement": "Short motivating line.",
  "encouragement_hi": "प्रोत्साहन की छोटी पंक्ति।"
}}
"""


# ---------------- GEMINI ----------------
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
    return raw


def gemini_try_list(prompt, model_list, image_bytes=None, image_mime=None):
    last = None
    for m in model_list:
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


# ---------------- GROQ ----------------
def groq_call(prompt, model):
    c = groq_client()
    if not c:
        raise RuntimeError("Groq not configured")
    r = c.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert bilingual exam writer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0.7,
    )
    return r.choices[0].message.content


def groq_try_list(prompt, model_list):
    last = None
    for m in model_list:
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


def call_ai_raw(prompt, image_bytes=None, image_mime=None, fast=False):
    """Returns raw text. Tries Gemini then Groq."""
    gem_list = FAST_GEMINI if fast else GEMINI_MODELS
    groq_list = FAST_GROQ if fast else GROQ_MODELS
    ge = None

    if gemini_client():
        try:
            return gemini_try_list(prompt, gem_list, image_bytes, image_mime)
        except Exception as e:
            ge = e
            print(f"[Fallback] Gemini down: {str(e)[:120]}", flush=True)

    if groq_client() and not image_bytes:
        try:
            print("[Fallback] Trying Groq", flush=True)
            return groq_try_list(prompt, groq_list)
        except Exception as e:
            raise RuntimeError(f"Both providers failed. G: {str(ge)[:80]} | Q: {str(e)[:80]}")

    if image_bytes and ge:
        raise RuntimeError(f"Image needs Gemini, which failed: {str(ge)[:150]}")

    raise RuntimeError("No AI provider configured")


def call_ai_quiz(prompt, image_bytes=None, image_mime=None):
    raw = call_ai_raw(prompt, image_bytes, image_mime, fast=False)
    return validate_quiz(json.loads(clean_json(raw)))


# ---------------- FILE PROCESSING ----------------
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
                    print(f"[PDF] page {i}: {e}", flush=True)
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

    raise ValueError("Unsupported file type. Use PDF, PNG, JPG, or WEBP.")


# ---------------- ROUTES ----------------
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
        "min_questions": MIN_QUESTIONS,
        "max_questions": MAX_QUESTIONS,
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    """Tells frontend how many questions make sense for this content."""
    data = request.get_json(silent=True) or {}
    text_len = int(data.get("text_length", 0))
    has_image = bool(data.get("has_image", False))
    rec = recommend_questions(text_len, has_image)
    return jsonify({"recommended": rec, "min": MIN_QUESTIONS, "max": MAX_QUESTIONS})


@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = (request.form.get("text") or "").strip()
        try:
            n = int(request.form.get("num_questions", 25))
        except Exception:
            n = 25
        diff = (request.form.get("difficulty") or "medium").lower()
        file = request.files.get("file")

        n = max(MIN_QUESTIONS, min(n, MAX_QUESTIONS))

        ftext, img_bytes, img_mime, info = process_file(file)
        if ftext:
            text = (text + "\n" + ftext).strip()

        if len(text) < 50 and not img_bytes:
            return jsonify({"error": "Need at least a paragraph of text or an image"}), 400

        if not gemini_client() and not groq_client():
            return jsonify({"error": "No API key configured"}), 500

        print(f"[GEN] {n} questions | text {len(text)} chars | image {bool(img_bytes)}", flush=True)

        quiz = []
        chunks_used = 0

        if img_bytes:
            # Single shot for image
            quiz = call_ai_quiz(build_prompt(
                "Analyze the attached image and generate the quiz.",
                n, diff
            ), img_bytes, img_mime)
            chunks_used = 1

        elif len(text) <= MAX_CHARS_PER_CHUNK:
            quiz = call_ai_quiz(build_prompt(text, n, diff))
            chunks_used = 1

        else:
            chunks = split_text(text, MAX_CHARS_PER_CHUNK)
            chunks_used = len(chunks)
            print(f"[GEN] Split into {chunks_used} chunks", flush=True)

            # Distribute questions across chunks
            base = n // chunks_used
            extra = n % chunks_used
            per_chunk = [base + (1 if i < extra else 0) for i in range(chunks_used)]

            for idx, ch in enumerate(chunks):
                if len(quiz) >= n:
                    break
                want = min(per_chunk[idx], n - len(quiz))
                if want <= 0:
                    continue
                try:
                    part = call_ai_quiz(build_prompt(ch, want, diff))
                    quiz.extend(part)
                    print(f"[GEN] Chunk {idx+1}/{chunks_used} → {len(part)} Qs", flush=True)
                except Exception as e:
                    print(f"[GEN] Chunk {idx+1} failed: {str(e)[:150]}", flush=True)
                    continue

            if not quiz:
                raise ValueError("All chunks failed. Try a smaller file.")

        # De-duplicate by question text
        seen = set()
        unique = []
        for q in quiz:
            key = q["question"].strip().lower()[:100]
            if key in seen:
                continue
            seen.add(key)
            unique.append(q)

        final = unique[:n]

        # Cleanup
        img_bytes = None
        text = None
        ftext = None
        gc.collect()

        return jsonify({
            "quiz": final,
            "meta": {
                "requested": n,
                "returned": len(final),
                "chunks": chunks_used,
                "file_info": info,
            }
        })

    except Exception as e:
        print("[ERROR] /generate:", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Generates a child-level explanation for a wrong answer."""
    try:
        data = request.get_json(silent=True) or {}
        question_en = (data.get("question") or "").strip()
        question_hi = (data.get("question_hi") or "").strip()
        wrong = (data.get("wrong") or "").strip()
        correct_en = (data.get("correct") or "").strip()
        correct_hi = (data.get("correct_hi") or correct_en).strip()

        if not question_en or not correct_en:
            return jsonify({"error": "Missing question or correct answer"}), 400

        prompt = build_explain_prompt(
            question_en, question_hi or question_en,
            wrong or "(no answer)", correct_en, correct_hi
        )

        raw = call_ai_raw(prompt, fast=True)
        cleaned = clean_json(raw)
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError("Explanation not an object")
        return jsonify({"explanation": obj})

    except Exception as e:
        print("[ERROR] /explain:", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
