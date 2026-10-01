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
print("[BOOT] QuizGenius starting", flush=True)
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

FAST_GEMINI = ["gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
FAST_GROQ = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

# ---------- TUNED CONSTANTS ----------
MAX_CHARS_PER_CHUNK = 2800      # small chunks → small outputs → no truncation
PER_CALL_QUESTIONS = 4          # bilingual JSON is heavy; 4 keeps output under limits
MAX_PASSES = 5                  # loop chunks this many times if we still need more
MAX_PDF_PAGES = 8
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


# ---------- JSON CLEANUP + REPAIR ----------
def _strip_fences(raw):
    if not raw:
        return ""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _find_array(raw):
    i, j = raw.find("["), raw.rfind("]")
    if i != -1 and j != -1 and j > i:
        return raw[i:j + 1]
    return raw


def repair_truncated_json(raw):
    """
    If the model truncates mid-array, cut back to the last complete object
    and close the array. Salvages most truncated responses.
    """
    raw = raw.strip()
    if not raw.startswith("["):
        return raw
    depth = 0
    in_string = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i
    if last_complete > 0:
        return raw[:last_complete + 1] + "]"
    return raw


def parse_json_array(raw):
    """Try hard to parse. Returns list or raises."""
    cleaned = _find_array(_strip_fences(raw))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try repair
        repaired = repair_truncated_json(cleaned)
        if repaired != cleaned:
            obj = json.loads(repaired)
        else:
            raise
    if not isinstance(obj, list):
        if isinstance(obj, dict) and "quiz" in obj:
            obj = obj["quiz"]
        else:
            raise ValueError("Not a list")
    return obj


def validate_quiz(q):
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
        raise ValueError("No valid questions after validation")
    return out


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
    return [c for c in chunks if len(c) > 80]


def retryable(e):
    m = str(e).lower()
    return any(k in m for k in ["503", "unavailable", "overloaded", "429",
                                 "rate limit", "500", "502", "504", "timeout", "deadline"])


def recommend_questions(text_len, has_image=False):
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


# ---------- PROMPTS ----------
def build_prompt(text, n, diff, existing_questions=None):
    dg = {
        "easy": "definitions and simple recall",
        "medium": "understanding and application",
        "hard": "deep analysis, edge cases, tricky distinctions"
    }
    avoid_block = ""
    if existing_questions:
        avoid_block = "\n\nDO NOT REPEAT or rephrase any of these already-generated questions:\n"
        for q in existing_questions[:40]:
            avoid_block += f"- {q}\n"

    return f"""You are a bilingual exam writer for Indian students.

Generate exactly {n} multiple-choice questions from the SOURCE below.

═══ LANGUAGE RULES ═══
- Every question, option, and explanation MUST be in BOTH English AND Hindi (Devanagari).
- Hindi MUST be in Devanagari script (हिंदी), NEVER romanized.
- Keep technical terms accurate: "photosynthesis" ↔ "प्रकाश संश्लेषण".

═══ QUESTION RULES ═══
- Difficulty: {diff.upper()} ({dg.get(diff, 'understanding')})
- Exactly 4 options, exactly 1 correct.
- Short explanation for the correct answer.
- No repeated concepts.

═══ OUTPUT RULES ═══
- Return ONLY a JSON array. No markdown, no preamble, no trailing text.
- Keep explanations to one short sentence in each language.

JSON shape:
[
  {{
    "question": "English question?",
    "question_hi": "हिंदी प्रश्न?",
    "options": ["A", "B", "C", "D"],
    "options_hi": ["अ", "ब", "स", "द"],
    "correct": 0,
    "explanation": "Short reason.",
    "explanation_hi": "संक्षिप्त कारण।"
  }}
]
{avoid_block}

SOURCE:
{text}
"""


def build_explain_prompt(q_en, q_hi, wrong_en, correct_en, correct_hi):
    return f"""A student answered wrong. Teach from absolute basics to a 10-year-old.

Question (EN): {q_en}
Question (HI): {q_hi}
Student chose: {wrong_en}
Correct answer (EN): {correct_en}
Correct answer (HI): {correct_hi}

Style:
- Simple everyday analogy.
- 3-5 short points.
- One concrete example.
- A memory trick.
- One encouraging line.

Language: both English AND Hindi (Devanagari).

Return ONLY this JSON:
{{
  "topic": "...", "topic_hi": "...",
  "why_wrong": "...", "why_wrong_hi": "...",
  "analogy": "...", "analogy_hi": "...",
  "basic_points": ["...", "..."], "basic_points_hi": ["...", "..."],
  "example": "...", "example_hi": "...",
  "memory_trick": "...", "memory_trick_hi": "...",
  "encouragement": "...", "encouragement_hi": "..."
}}
"""


# ---------- GEMINI ----------
def gemini_call(prompt, model, image_bytes=None, image_mime=None):
    c = gemini_client()
    if not c:
        raise RuntimeError("Gemini not configured")
    from google.genai import types

    contents = [prompt]
    if image_bytes and image_mime:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))

    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=8000,
    )
    r = c.models.generate_content(model=model, contents=contents, config=config)
    raw = getattr(r, "text", None) or ""
    if not raw:
        try:
            raw = r.candidates[0].content.parts[0].text
        except Exception:
            pass
    if not raw:
        raise ValueError("Empty response from Gemini")
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
                print(f"[Gemini] {m} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
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
            {"role": "system", "content": "You are a bilingual exam writer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0.7,
        max_tokens=3500,
    )
    raw = r.choices[0].message.content
    if not raw:
        raise ValueError("Empty response from Groq")
    return raw


def groq_try_list(prompt, model_list):
    last = None
    for m in model_list:
        for a in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[Groq] {m} attempt {a+1}", flush=True)
                return groq_call(prompt, m)
            except Exception as e:
                last = e
                print(f"[Groq] {m} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
                if not retryable(e):
                    break
                if a < RETRIES_PER_MODEL:
                    time.sleep(1.2 ** a)
    raise RuntimeError(f"Groq exhausted: {str(last)[:150]}")


def call_ai_raw(prompt, image_bytes=None, image_mime=None, fast=False):
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
            raise RuntimeError(f"Both failed. G: {str(ge)[:80]} | Q: {str(e)[:80]}")

    if image_bytes and ge:
        raise RuntimeError(f"Image needs Gemini: {str(ge)[:150]}")

    raise RuntimeError("No AI provider configured")


def call_ai_quiz(prompt, image_bytes=None, image_mime=None):
    raw = call_ai_raw(prompt, image_bytes, image_mime, fast=False)
    print(f"[AI] Got {len(raw)} chars raw", flush=True)
    try:
        arr = parse_json_array(raw)
    except Exception as e:
        print(f"[AI] JSON parse failed: {e}", flush=True)
        print(f"[AI] Raw preview: {raw[:500]}", flush=True)
        raise
    return validate_quiz(arr)


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
        "chunk_size": MAX_CHARS_PER_CHUNK,
        "per_call_questions": PER_CALL_QUESTIONS,
    })


@app.route("/recommend", methods=["POST"])
def recommend():
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

        print(f"[GEN] target={n} | text={len(text)} chars | image={bool(img_bytes)}", flush=True)

        quiz = []
        chunk_errors = []
        chunks_used = 0
        passes_used = 0

        # ---------- IMAGE PATH (single shot) ----------
        if img_bytes:
            try:
                prompt = build_prompt(
                    "Analyze the attached image and generate the quiz.",
                    n, diff
                )
                quiz = call_ai_quiz(prompt, img_bytes, img_mime)
                chunks_used = 1
                passes_used = 1
            except Exception as e:
                print(f"[GEN] Image call failed: {e}", flush=True)
                return jsonify({"error": f"Image analysis failed: {str(e)[:200]}"}), 500

        # ---------- TEXT PATH ----------
        else:
            chunks = split_text(text, MAX_CHARS_PER_CHUNK) if len(text) > MAX_CHARS_PER_CHUNK else [text]
            chunks_used = len(chunks)
            print(f"[GEN] Using {chunks_used} chunk(s)", flush=True)

            for pass_num in range(MAX_PASSES):
                if len(quiz) >= n:
                    break
                passes_used = pass_num + 1
                print(f"[GEN] Pass {passes_used} | have {len(quiz)}/{n}", flush=True)

                for idx, ch in enumerate(chunks):
                    if len(quiz) >= n:
                        break
                    want = min(PER_CALL_QUESTIONS, n - len(quiz))
                    existing = [q["question"] for q in quiz]
                    try:
                        prompt = build_prompt(ch, want, diff, existing_questions=existing)
                        part = call_ai_quiz(prompt)

                        existing_keys = {q["question"].lower().strip()[:80] for q in quiz}
                        added = 0
                        for q in part:
                            k = q["question"].lower().strip()[:80]
                            if k not in existing_keys:
                                quiz.append(q)
                                existing_keys.add(k)
                                added += 1
                        print(f"[GEN] Pass {passes_used} chunk {idx+1}/{chunks_used}: "
                              f"got {len(part)}, added {added} unique", flush=True)
                    except Exception as e:
                        err_msg = f"Pass{passes_used}/Chunk{idx+1}: {type(e).__name__}: {str(e)[:180]}"
                        print(f"[GEN] {err_msg}", flush=True)
                        chunk_errors.append(err_msg)
                        continue

        # ---------- FINAL CHECK ----------
        if not quiz:
            detail = chunk_errors[-3:] if chunk_errors else ["No chunks attempted"]
            return jsonify({
                "error": "Could not generate questions. Details: " + " | ".join(detail)
            }), 500

        quiz = quiz[:n]

        img_bytes = None
        text = None
        ftext = None
        gc.collect()

        print(f"[GEN] Done: {len(quiz)} questions", flush=True)

        return jsonify({
            "quiz": quiz,
            "meta": {
                "requested": n,
                "returned": len(quiz),
                "chunks": chunks_used,
                "passes": passes_used,
                "file_info": info,
                "errors": chunk_errors[:5],
            }
        })

    except Exception as e:
        print("[ERROR] /generate:", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json(silent=True) or {}
        q_en = (data.get("question") or "").strip()
        q_hi = (data.get("question_hi") or "").strip()
        wrong = (data.get("wrong") or "").strip()
        c_en = (data.get("correct") or "").strip()
        c_hi = (data.get("correct_hi") or c_en).strip()

        if not q_en or not c_en:
            return jsonify({"error": "Missing question or correct answer"}), 400

        prompt = build_explain_prompt(q_en, q_hi or q_en, wrong or "(no answer)", c_en, c_hi)
        raw = call_ai_raw(prompt, fast=True)
        cleaned = _find_array(_strip_fences(raw))
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
