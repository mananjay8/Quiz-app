import os
import json
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set.")


def clean_json(raw):
    """Strip markdown fences and whitespace from Gemini's response."""
    raw = raw.strip()
    # Remove ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Find the first [ and last ] to isolate the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return raw.strip()


def build_prompt(text, num_questions, difficulty):
    """Build a strict prompt that returns clean JSON."""

    difficulty_guide = {
        "easy": "Focus on definitions, basic facts, and simple recall. Warm-up level.",
        "medium": "Test understanding and application. Mix recall with reasoning.",
        "hard": "Test deep understanding, edge cases, and tricky distinctions. Include one question that requires multi-step reasoning."
    }

    return f"""You are an expert exam writer creating a Telegram-style quiz.

Generate exactly {num_questions} multiple-choice questions from the text below.

DIFFICULTY: {difficulty.upper()}
{difficulty_guide.get(difficulty, difficulty_guide["medium"])}

STRICT RULES:
1. Each question must have exactly 4 options.
2. Exactly ONE option is correct.
3. Include a short explanation (1-2 sentences) for why the correct answer is right.
4. Vary the position of the correct answer (don't always put it first).
5. Questions must be self-contained — no "according to the passage" references.
6. Avoid trick questions unless difficulty is "hard".
7. Do not repeat concepts across questions.

Return ONLY a valid JSON array. No markdown, no code fences, no preamble.

Format:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 2,
    "explanation": "Short explanation of why this is correct."
  }}
]

The "correct" field is the 0-based index of the right answer (0, 1, 2, or 3).

TEXT TO BASE QUESTIONS ON:
{text}
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    num_questions = int(data.get("num_questions", 5))
    difficulty = (data.get("difficulty") or "medium").lower()

    # Clamp question count
    num_questions = max(1, min(num_questions, 15))

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < 80:
        return jsonify({"error": "Please paste at least a paragraph (80+ characters)."}), 400
    if not GEMINI_API_KEY:
        return jsonify({"error": "Server is missing its API key."}), 500

    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
            },
        )
        prompt = build_prompt(text, num_questions, difficulty)
        response = model.generate_content(prompt)
        raw = clean_json(response.text)
        quiz = json.loads(raw)

        # Validate structure
        if not isinstance(quiz, list) or len(quiz) == 0:
            raise ValueError("AI returned an empty quiz.")

        for q in quiz:
            if not all(k in q for k in ("question", "options", "correct")):
                raise ValueError("AI returned a malformed question.")
            if len(q["options"]) != 4:
                raise ValueError("A question doesn't have 4 options.")
            if not isinstance(q["correct"], int) or not (0 <= q["correct"] <= 3):
                raise ValueError("A question has an invalid correct index.")

        return jsonify({"quiz": quiz})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned invalid JSON. Try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
