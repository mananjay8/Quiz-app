import os, sys, json, re, math, time, gc, traceback, threading, uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

GEMINI_FALLBACK = ["gemini-2.5-flash", "gemini-flash-latest"]

# Expanded Groq list — small models first (bigger free tier), big model last
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "gemma2-9b-it",
]

CHUNK_SIZE = 3500
Qs_PER_CALL = 6
MAX_PASSES = 8
MAX_PDF_PAGES = 8
MIN_Q, MAX_Q = 25, 100
RETRIES = 1
JOB_TTL_MIN = 15
MAX_CONCURRENT_JOBS = 2
GROQ_MIN_GAP = 3.0   # seconds between Groq calls to respect free-tier RPM

_gem = None
_gro = None
_gem_models_cache = None
_jobs = {}
_jobs_lock = threading.Lock()

# Groq pacing
_last_groq_call = 0
_groq_lock = threading.Lock()


# ---------- CLIENTS ----------
def get_gem():
    global _gem
    if _gem is None and GEMINI_KEY:
        try:
            from google import genai
            _gem = genai.Client(api_key=GEMINI_KEY)
            print("[INIT] Gemini ready", flush=True)
        except Exception as e:
            print(f"[INIT] Gemini failed: {e}", flush=True)
    return _gem


def get_gro():
    global _gro
    if _gro is None and GROQ_KEY:
        try:
            from groq import Groq
            _gro = Groq(api_key=GROQ_KEY)
            print("[INIT] Groq ready", flush=True)
        except Exception as e:
            print(f"[INIT] Groq failed: {e}", flush=True)
    return _gro


def discover_gemini_models():
    global _gem_models_cache
    if _gem_models_cache is not None:
        return _gem_models_cache

    client = get_gem()
    if not client:
        _gem_models_cache = (GEMINI_FALLBACK, GEMINI_FALLBACK)
        return _gem_models_cache

    try:
        available = [m.name.replace("models/", "") for m in client.models.list()]
        exclude = ["tts", "image", "live", "audio", "embedding", "robotics",
                   "computer-use", "veo", "lyria", "learnlm", "aqa",
                   "transcribe", "native-audio", "thinking"]
        text_models = [m for m in available
                       if not any(k in m.lower() for k in exclude)]
        flash = [m for m in text_models if "flash" in m.lower()]
        pro = [m for m in text_models if "pro" in m.lower() and m not in flash]

        def score(name):
            s = 0
            if "preview" in name or "exp" in name: s += 100
            if "lite" in name: s += 10
            m = re.search(r"(\d+)\.(\d+)", name)
            if m: s -= (int(m.group(1)) * 100 + int(m.group(2)))
            return s

        flash.sort(key=score)
        pro.sort(key=score)
        main = flash + pro
        fast = [m for m in main if "lite" in m.lower()] + \
               [m for m in main if "lite" not in m.lower()]
        if not main:
            main = GEMINI_FALLBACK
            fast = GEMINI_FALLBACK

        print(f"[GEMINI] main={main[:3]} fast={fast[:3]}", flush=True)
        _gem_models_cache = (main[:5], fast[:5])
        return _gem_models_cache
    except Exception as e:
        print(f"[GEMINI] discovery failed: {e}", flush=True)
        _gem_models_cache = (GEMINI_FALLBACK, GEMINI_FALLBACK)
        return _gem_models_cache


# ---------- ERROR HANDLERS ----------
@app.errorhandler(413)
def e413(e): return jsonify({"error": "File too large. Max 8MB."}), 413

@app.errorhandler(Exception)
def eall(e):
    print("[ERR]", flush=True); traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


# ---------- JOB MANAGEMENT ----------
def _cleanup_jobs():
    now = datetime.now()
    with _jobs_lock:
        dead = [jid for jid, j in _jobs.items()
                if now - j["created"] > timedelta(minutes=JOB_TTL_MIN)]
        for jid in dead:
            del _jobs[jid]


def _kill_stale_jobs():
    now = datetime.now()
    with _jobs_lock:
        for jid, j in list(_jobs.items()):
            if j["status"] in ("running", "queued"):
                last = j.get("updated", j["created"])
                age = (now - last).total_seconds()
                if age > 300:
                    print(f"[JOB {jid}] STALE {age:.0f}s → failed", flush=True)
                    j["status"] = "error"
                    j["error"] = "Timed out (no progress for 5 minutes)"
                    j["questions"] = []


def _update_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated"] = datetime.now()


# ---------- JSON HELPERS ----------
def extract_json(text):
    if not text:
        raise ValueError("Empty response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON. Preview: {text[:200]}")


def parse_quiz_array(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict) and "quiz" in obj: obj = obj["quiz"]
    if isinstance(obj, dict) and "questions" in obj: obj = obj["questions"]
    if not isinstance(obj, list): raise ValueError("Not a list")
    return obj


def parse_explain_object(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict): raise ValueError("Not an object")
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


# ---------- ERROR CLASSIFIERS ----------
def retryable(e):
    m = str(e).lower()
    return any(k in m for k in ["503","unavailable","overloaded","500","502","504","timeout","deadline"])

def is_404(e):
    m = str(e).lower()
    return "404" in m or "not_found" in m or "not found" in m

def is_quota_exhausted(e):
    m = str(e).lower()
    return any(k in m for k in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"])


# ---------- PROMPTS ----------
def quiz_prompt(text, n, diff, existing=None):
    avoid = ""
    if existing:
        avoid = "\n\nDo NOT repeat:\n" + "\n".join(f"- {q}" for q in existing[:30])
    return f"""Bilingual exam writer for Indian students. Generate exactly {n} MCQs.

RULES:
- Every question, option, and explanation in BOTH English AND Hindi (Devanagari).
- Hindi in Devanagari script only. No romanized Hindi.
- 4 options each, 1 correct. Short explanation (1 sentence per language).
- Difficulty: {diff.upper()}.
- Return ONLY a JSON array. No markdown.

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

Return ONLY this JSON. No text before or after:
{{"topic":"...","topic_hi":"...","why_wrong":"...","why_wrong_hi":"...","analogy":"...","analogy_hi":"...","basic_points":["...","..."],"basic_points_hi":["...","..."],"example":"...","example_hi":"...","memory_trick":"...","memory_trick_hi":"...","encouragement":"...","encouragement_hi":"..."}}
"""


# ---------- AI CALLS ----------
def call_gem(prompt, model, img=None, mime=None):
    c = get_gem()
    if not c: raise RuntimeError("Gemini not configured")
    from google.genai import types
    parts = [prompt]
    if img and mime:
        parts.append(types.Part.from_bytes(data=img, mime_type=mime))
    cfg = types.GenerateContentConfig(temperature=0.7, max_output_tokens=8000)
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
        messages=[
            {"role":"system","content":"Bilingual exam writer. Return ONLY valid JSON."},
            {"role":"user","content":prompt}
        ],
        model=model, temperature=0.7, max_tokens=4000,
    )
    raw = r.choices[0].message.content
    if not raw: raise ValueError("Empty from Groq")
    return raw


def ai_raw(prompt, img=None, mime=None, fast=False):
    """Groq FIRST with pacing. Gemini fallback. Images only via Gemini."""
    global _last_groq_call
    main_models, fast_models = discover_gemini_models()
    gm = fast_models if fast else main_models
    last = None

    # ---------- IMAGE: must use Gemini ----------
    if img:
        if get_gem():
            for m in gm:
                for a in range(RETRIES + 1):
                    try:
                        print(f"[G-img] {m} try{a+1}", flush=True)
                        return call_gem(prompt, m, img, mime)
                    except Exception as e:
                        last = e
                        print(f"[G-img] {m} fail: {str(e)[:120]}", flush=True)
                        if is_404(e) or is_quota_exhausted(e):
                            break
                        if not retryable(e):
                            break
                        if a < RETRIES:
                            time.sleep(1.2 ** a)
        raise RuntimeError(f"Image needs Gemini: {str(last)[:150]}")

    # ---------- TEXT: Groq FIRST with pacing ----------
    if get_gro():
        for m in GROQ_MODELS:
            # Enforce minimum gap between Groq calls
            with _groq_lock:
                elapsed = time.time() - _last_groq_call
                if elapsed < GROQ_MIN_GAP:
                    time.sleep(GROQ_MIN_GAP - elapsed)
                _last_groq_call = time.time()

            for a in range(RETRIES + 1):
                try:
                    print(f"[Q] {m} try{a+1}", flush=True)
                    return call_gro(prompt, m)
                except Exception as e:
                    last = e
                    print(f"[Q] {m} fail: {str(e)[:130]}", flush=True)
                    if is_quota_exhausted(e):
                        time.sleep(2)
                        break
                    if is_404(e):
                        break
                    if not retryable(e):
                        break
                    if a < RETRIES:
                        time.sleep(1.5 ** a)

    # ---------- TEXT: Gemini fallback ----------
    if get_gem():
        gem_exhausted = False
        for m in gm:
            if gem_exhausted:
                break
            for a in range(RETRIES + 1):
                try:
                    print(f"[G] {m} try{a+1}", flush=True)
                    return call_gem(prompt, m)
                except Exception as e:
                    last = e
                    print(f"[G] {m} fail: {str(e)[:120]}", flush=True)
                    if is_quota_exhausted(e):
                        print("[G] Gemini exhausted", flush=True)
                        gem_exhausted = True
                        break
                    if is_404(e):
                        break
                    if not retryable(e):
                        break
                    if a < RETRIES:
                        time.sleep(1.2 ** a)

    raise RuntimeError(f"All providers failed: {str(last)[:150]}")


def ai_quiz(prompt, img=None, mime=None):
    raw = ai_raw(prompt, img, mime)
    print(f"[AI] {len(raw)} chars", flush=True)
    arr = parse_quiz_array(raw)
    del raw; gc.collect()
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


# ---------- BACKGROUND WORKER ----------
def _run_generation(job_id, text, n, diff, img_bytes, img_mime):
    print(f"[JOB {job_id}] start target={n}", flush=True)
    try:
        quiz = []; errors = []; chunks_used = 0; passes = 0
        _update_job(job_id, status="running")

        if img_bytes:
            try:
                part = ai_quiz(quiz_prompt(
                    "Analyze the attached image and generate the quiz.", n, diff
                ), img_bytes, img_mime)
                quiz.extend(part)
                chunks_used = 1; passes = 1
                _update_job(job_id, generated=len(quiz), questions=list(quiz))
            except Exception as e:
                print(f"[JOB {job_id}] image failed: {e}", flush=True)
                _update_job(job_id, status="error", error=f"Image failed: {str(e)[:200]}")
                return
            img_bytes = None
        else:
            chunks = split_text(text, CHUNK_SIZE)
            chunks_used = len(chunks)
            _update_job(job_id, chunks=chunks_used)
            print(f"[JOB {job_id}] {chunks_used} chunk(s)", flush=True)

            for p in range(MAX_PASSES):
                if len(quiz) >= n: break
                passes = p + 1
                for idx, ch in enumerate(chunks):
                    if len(quiz) >= n: break
                    want = min(Qs_PER_CALL, n - len(quiz))
                    try:
                        part = ai_quiz(quiz_prompt(
                            ch, want, diff, [q["question"] for q in quiz]
                        ))
                        keys = {q["question"].lower().strip()[:80] for q in quiz}
                        added = 0
                        for q in part:
                            k = q["question"].lower().strip()[:80]
                            if k not in keys:
                                quiz.append(q); keys.add(k); added += 1
                        print(f"[JOB {job_id}] p{passes}c{idx+1}: +{added} ({len(quiz)}/{n})", flush=True)
                        _update_job(job_id, generated=len(quiz),
                                    questions=list(quiz), passes=passes)
                        del part, keys; gc.collect()
                    except Exception as e:
                        msg = f"p{passes}c{idx+1}: {type(e).__name__}: {str(e)[:120]}"
                        print(f"[JOB {job_id}] {msg}", flush=True)
                        errors.append(msg)
                gc.collect()

        if not quiz:
            _update_job(job_id, status="error",
                        error="Could not generate. " + " | ".join(errors[-2:]))
            return

        final = quiz[:n]
        _update_job(job_id, status="done", questions=final,
                    generated=len(final), passes=passes,
                    errors=errors[:3], finished=datetime.now())
        print(f"[JOB {job_id}] done: {len(final)}", flush=True)
    except Exception as e:
        traceback.print_exc()
        _update_job(job_id, status="error", error=f"{type(e).__name__}: {str(e)[:200]}")


# ---------- ROUTES ----------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/debug")
def dbg():
    try:
        main, fast = discover_gemini_models()
        with _jobs_lock:
            active = sum(1 for j in _jobs.values() if j["status"] in ("running","queued"))
        return jsonify({
            "gem": bool(GEMINI_KEY), "gro": bool(GROQ_KEY),
            "gemini_main": main, "gemini_fast": fast,
            "groq_models": GROQ_MODELS,
            "chunk": CHUNK_SIZE, "per_call": Qs_PER_CALL,
            "groq_min_gap": GROQ_MIN_GAP,
            "active_jobs": active,
        })
    except Exception as e:
        return jsonify({"error": f"Debug failed: {str(e)[:200]}"}), 500


@app.route("/recommend", methods=["POST"])
def rec():
    try:
        d = request.get_json(silent=True) or {}
        tl = int(d.get("text_length",0)); hi = bool(d.get("has_image",False))
        r = 25 if hi or tl < 1500 else 30 if tl < 4000 else 40 if tl < 9000 else 55 if tl < 18000 else 70 if tl < 35000 else 85 if tl < 60000 else 100
        return jsonify({"recommended": r, "min": MIN_Q, "max": MAX_Q})
    except Exception:
        return jsonify({"recommended": 25, "min": MIN_Q, "max": MAX_Q})


@app.route("/generate", methods=["POST"])
def generate():
    try:
        _cleanup_jobs()
        _kill_stale_jobs()

        with _jobs_lock:
            active = sum(1 for j in _jobs.values() if j["status"] in ("running","queued"))
            if active >= MAX_CONCURRENT_JOBS:
                return jsonify({
                    "error": "Server is busy with other quizzes. Wait 1-2 min and retry."
                }), 429

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

        job_id = uuid.uuid4().hex[:12]
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "queued", "total": n, "generated": 0,
                "questions": [], "chunks": 0, "passes": 0,
                "created": datetime.now(), "updated": datetime.now(),
                "file_info": info, "error": None,
            }

        t = threading.Thread(target=_run_generation,
                             args=(job_id, text, n, diff, img, mime), daemon=True)
        t.start()
        print(f"[GEN] job {job_id} target={n}", flush=True)
        return jsonify({"job_id": job_id, "total": n, "file_info": info})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


@app.route("/progress/<job_id>")
def progress(job_id):
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return jsonify({"error": "Job not found or expired"}), 404
            return jsonify({
                "status": job["status"], "total": job["total"],
                "generated": job["generated"], "chunks": job.get("chunks", 0),
                "passes": job.get("passes", 0), "error": job.get("error"),
                "file_info": job.get("file_info", ""),
                "questions": job["questions"] if job["status"] == "done" else [],
            })
    except Exception as e:
        return jsonify({"error": f"Progress failed: {str(e)[:150]}"}), 500


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
        print(f"[EXPLAIN] {q_en[:60]}", flush=True)
        raw = ai_raw(explain_prompt(q_en, q_hi or q_en, w or "(none)", c_en, c_hi), fast=True)
        print(f"[EXPLAIN] {len(raw)} chars", flush=True)
        return jsonify({"explanation": parse_explain_object(raw)})
    except Exception as e:
        print(f"[EXPLAIN ERR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
