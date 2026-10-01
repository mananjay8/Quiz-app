import os, sys, json, re, math, time, gc, traceback
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct"]

FAST_GEMINI = ["gemini-2.0-flash", "gemini-2.5-flash"]
FAST_GROQ = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

CHUNK_SIZE = 2500
Qs_PER_CALL = 3
MAX_PASSES = 6
MAX_PDF_PAGES = 6
MIN_Q, MAX_Q = 25, 100
RETRIES = 1

_gem = None
_gro = None


def get_gem():
    global _gem
    if _gem is None and GEMINI_KEY:
        try:
            from google import genai
            _gem = genai.Client(api_key=GEMINI_KEY)
        except Exception as e:
            print(f"[INIT] Gemini: {e}", flush=True)
    return _gem


def get_gro():
    global _gro
    if _gro is None and GROQ_KEY:
        try:
            from groq import Groq
            _gro = Groq(api_key=GROQ_KEY)
        except Exception as e:
            print(f"[INIT] Groq: {e}", flush=True)
    return _gro


@app.errorhandler(413)
def e413(e): return jsonify({"error": "File too large. Max 8MB."}), 413

@app.errorhandler(Exception)
def eall(e):
    print("[ERR]", flush=True); traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


# ---------- ROBUST JSON PARSING ----------
def extract_json(text):
    """
    Extract the FIRST valid JSON value from arbitrary text.
    Uses raw_decode which stops at the end of the JSON and ignores trailing junk.
    Handles: markdown fences, preamble, trailing comments, extra data.
    """
    if not text:
        raise ValueError("Empty response")
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()
    # Try each { or [ position until we find valid JSON
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found. Preview: {text[:200]}")


def parse_quiz_array(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict) and "quiz" in obj:
        obj = obj["quiz"]
    if isinstance(obj, dict) and "questions" in obj:
        obj = obj["questions"]
    if not isinstance(obj, list):
        raise ValueError("Not a list")
    return obj


def parse_explain_object(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("Not an object")
    return obj


def validate(items):
    out = []
    for it in items:
        if not isinstance(it, dict): continue
        if not all(k in it for k in ("question","options","correct")): continue
        if not isinstance(it["options"], list) or len(it["options"]) != 4: continue
        if not isinstance(it["correct"], int) or not 0 <= it["correct"] <= 3: continue
        it["question"] = str(it["question"])
        it["options"] = [str(x) for x in it["options"]]
        it["explanation"] = str(it.get("explanation",""))
        it["question_hi"] = str(it.get("question_hi") or it["question"])
        oh = it.get("options_hi")
        if not isinstance(oh, list) or len(oh) != 4: oh = list(it["options"])
        it["options_hi"] = [str(x) for x in oh]
        it["explanation_hi"] = str(it.get("explanation_hi") or it["explanation"])
        out.append(it)
    if not out: raise ValueError("No valid questions")
    return out


def split_text(text, limit):
    if len(text) <= limit: return [text]
    chunks, cur = [], ""
    for para in re.split(r'\n\s*\n', text):
        if len(cur) + len(para) + 2 <= limit:
            cur += para + "\n\n"
        else:
            if cur: chunks.append(cur.strip())
            if len(para) > limit:
                for s in re.split(r'(?<=[.!?])\s+', para):
                    if len(cur) + len(s) + 1 <= limit:
                        cur += s + " "
                    else:
                        if cur: chunks.append(cur.strip())
                        cur = s + " "
                if cur: chunks.append(cur.strip()); cur = ""
            else:
                cur = para + "\n\n"
    if cur: chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 80]


def retryable(e):
    m = str(e).lower()
    return any(k in m for k in ["503","unavailable","overloaded","429","rate limit","500","502","504","timeout","deadline"])


def quiz_prompt(text, n, diff, existing=None):
    avoid = ""
    if existing:
        avoid = "\n\nDo NOT repeat these:\n" + "\n".join(f"- {q}" for q in existing[:30])
    return f"""Bilingual exam writer for Indian students. Generate exactly {n} MCQs.

RULES:
- Every question, option, and explanation in BOTH English AND Hindi (Devanagari).
- Hindi in Devanagari script only. No romanized Hindi.
- 4 options each, 1 correct. Short explanation (1 sentence per language).
- Difficulty: {diff.upper()}.
- Return ONLY a JSON array. No markdown, no preamble, no trailing text.

FORMAT:
[{{"question":"...","question_hi":"...","options":["A","B","C","D"],"options_hi":["अ","ब","स","द"],"correct":0,"explanation":"...","explanation_hi":"..."}}]
{avoid}

SOURCE:
{text}
"""


def explain_prompt(q_en, q_hi, wrong, c_en, c_hi):
    return f"""Teach a 10-year-old why they got this wrong.

Q (EN): {q_en}
Q (HI): {q_hi}
Chose: {wrong}
Correct (EN): {c_en}
Correct (HI): {c_hi}

Style: simple analogy + 3-5 short points + 1 example + 1 memory trick + 1 encouraging line.
Both English AND Hindi (Devanagari).

Return ONLY this JSON object. No text before or after it. No markdown.

{{"topic":"...","topic_hi":"...","why_wrong":"...","why_wrong_hi":"...","analogy":"...","analogy_hi":"...","basic_points":["...","..."],"basic_points_hi":["...","..."],"example":"...","example_hi":"...","memory_trick":"...","memory_trick_hi":"...","encouragement":"...","encouragement_hi":"..."}}
"""


def call_gem(prompt, model, img=None, mime=None):
    c = get_gem()
    if not c: raise RuntimeError("Gemini not configured")
    from google.genai import types
    parts = [prompt]
    if img and mime:
        parts.append(types.Part.from_bytes(data=img, mime_type=mime))
    cfg = types.GenerateContentConfig(temperature=0.7, max_output_tokens=6000)
    r = c.models.generate_content(model=model, contents=parts, config=cfg)
    raw = getattr(r, "text", None) or ""
    if not raw:
        try: raw = r.candidates[0].content.parts[0].text
        except Exception: pass
    if not raw: raise ValueError("Empty from Gemini")
    return raw


def call_gro(prompt, model):
    c = get_gro()
    if not c: raise RuntimeError("Groq not configured")
    r = c.chat.completions.create(
        messages=[{"role":"system","content":"Bilingual exam writer. Return ONLY JSON."},
                  {"role":"user","content":prompt}],
        model=model, temperature=0.7, max_tokens=2500,
    )
    raw = r.choices[0].message.content
    if not raw: raise ValueError("Empty from Groq")
    return raw


def ai_raw(prompt, img=None, mime=None, fast=False):
    gm = FAST_GEMINI if fast else GEMINI_MODELS
    gr = FAST_GROQ if fast else GROQ_MODELS
    last = None
    if get_gem():
        for m in gm:
            for a in range(RETRIES + 1):
                try:
                    print(f"[G] {m} try{a+1}", flush=True)
                    return call_gem(prompt, m, img, mime)
                except Exception as e:
                    last = e
                    print(f"[G] {m} fail: {str(e)[:120]}", flush=True)
                    if not retryable(e): break
                    if a < RETRIES: time.sleep(1.2**a)
    if get_gro() and not img:
        for m in gr:
            for a in range(RETRIES + 1):
                try:
                    print(f"[Q] {m} try{a+1}", flush=True)
                    return call_gro(prompt, m)
                except Exception as e:
                    last = e
                    print(f"[Q] {m} fail: {str(e)[:120]}", flush=True)
                    if not retryable(e): break
                    if a < RETRIES: time.sleep(1.2**a)
    if img and last: raise RuntimeError(f"Image needs Gemini: {str(last)[:150]}")
    raise RuntimeError(f"All providers failed: {str(last)[:150]}")


def ai_quiz(prompt, img=None, mime=None):
    raw = ai_raw(prompt, img, mime)
    print(f"[AI] {len(raw)} chars", flush=True)
    arr = parse_quiz_array(raw)
    del raw
    gc.collect()
    return validate(arr)


def process_file(f):
    if not f or not f.filename: return "", None, None, ""
    name = f.filename.lower()
    if name.endswith(".pdf"):
        import PyPDF2
        try:
            r = PyPDF2.PdfReader(f)
            total = len(r.pages); n = min(total, MAX_PDF_PAGES)
            parts = []
            for i in range(n):
                try: parts.append(r.pages[i].extract_text() or "")
                except Exception: pass
            text = "\n\n".join(parts)
            del r, parts; gc.collect()
            return text, None, None, f"PDF ({n}/{total})"
        finally:
            try: f.close()
            except Exception: pass
    if name.endswith((".png",".jpg",".jpeg",".webp")):
        raw = f.read()
        ext = name.rsplit(".",1)[-1]
        mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}[ext]
        try: f.close()
        except Exception: pass
        return "", raw, mime, f"Image ({ext.upper()})"
    raise ValueError("Use PDF, PNG, JPG, or WEBP")


@app.route("/")
def home(): return render_template("index.html")

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/debug")
def dbg():
    return jsonify({"gem": bool(GEMINI_KEY), "gro": bool(GROQ_KEY),
                    "chunk": CHUNK_SIZE, "per_call": Qs_PER_CALL,
                    "groq_models": GROQ_MODELS})

@app.route("/recommend", methods=["POST"])
def rec():
    d = request.get_json(silent=True) or {}
    tl = int(d.get("text_length",0)); hi = bool(d.get("has_image",False))
    r = 25 if hi or tl < 1500 else 30 if tl < 4000 else 40 if tl < 9000 else 55 if tl < 18000 else 70 if tl < 35000 else 85 if tl < 60000 else 100
    return jsonify({"recommended": r, "min": MIN_Q, "max": MAX_Q})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = (request.form.get("text") or "").strip()
        try: n = int(request.form.get("num_questions", 25))
        except Exception: n = 25
        diff = (request.form.get("difficulty") or "medium").lower()
        f = request.files.get("file")
        n = max(MIN_Q, min(n, MAX_Q))

        ft, img, mime, info = process_file(f)
        if ft: text = (text + "\n" + ft).strip()
        del f, ft; gc.collect()

        if len(text) < 50 and not img:
            return jsonify({"error": "Need text or image"}), 400
        if not get_gem() and not get_gro():
            return jsonify({"error": "No API key"}), 500

        print(f"[GEN] target={n} text={len(text)} img={bool(img)}", flush=True)

        quiz = []
        errors = []
        chunks_used = 0
        passes = 0

        if img:
            try:
                quiz = ai_quiz(quiz_prompt("Analyze the image and generate the quiz.", n, diff), img, mime)
                chunks_used = 1; passes = 1
            except Exception as e:
                return jsonify({"error": f"Image failed: {str(e)[:200]}"}), 500
            img = None
        else:
            chunks = split_text(text, CHUNK_SIZE)
            chunks_used = len(chunks)
            print(f"[GEN] {chunks_used} chunk(s)", flush=True)

            for p in range(MAX_PASSES):
                if len(quiz) >= n: break
                passes = p + 1
                for idx, ch in enumerate(chunks):
                    if len(quiz) >= n: break
                    want = min(Qs_PER_CALL, n - len(quiz))
                    try:
                        part = ai_quiz(quiz_prompt(ch, want, diff, [q["question"] for q in quiz]))
                        keys = {q["question"].lower().strip()[:80] for q in quiz}
                        added = 0
                        for q in part:
                            k = q["question"].lower().strip()[:80]
                            if k not in keys:
                                quiz.append(q); keys.add(k); added += 1
                        print(f"[GEN] p{passes} c{idx+1}: +{added} ({len(quiz)}/{n})", flush=True)
                        del part, keys; gc.collect()
                    except Exception as e:
                        msg = f"p{passes}c{idx+1}: {type(e).__name__}: {str(e)[:150]}"
                        print(f"[GEN] {msg}", flush=True)
                        errors.append(msg)
                gc.collect()

        if not quiz:
            d = errors[-3:] if errors else ["No chunks"]
            return jsonify({"error": "Could not generate. " + " | ".join(d)}), 500

        final = quiz[:n]
        del quiz, text; gc.collect()
        print(f"[GEN] done: {len(final)}", flush=True)

        return jsonify({"quiz": final, "meta": {
            "requested": n, "returned": len(final),
            "chunks": chunks_used, "passes": passes,
            "file_info": info, "errors": errors[:3]
        }})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    try:
        d = request.get_json(silent=True) or {}
        q_en = (d.get("question") or "").strip()
        q_hi = (d.get("question_hi") or "").strip()
        w = (d.get("wrong") or "").strip()
        c_en = (d.get("correct") or "").strip()
        c_hi = (d.get("correct_hi") or c_en).strip()
        if not q_en or not c_en:
            return jsonify({"error": "Missing data"}), 400

        print(f"[EXPLAIN] {q_en[:60]}...", flush=True)
        raw = ai_raw(explain_prompt(q_en, q_hi or q_en, w or "(none)", c_en, c_hi), fast=True)
        print(f"[EXPLAIN] got {len(raw)} chars", flush=True)

        obj = parse_explain_object(raw)
        return jsonify({"explanation": obj})
    except Exception as e:
        print(f"[EXPLAIN ERR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
