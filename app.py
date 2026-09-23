import os
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Read the API key from Render's environment variables.
# Locally, you can set this in your terminal before running.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set. Quiz generation will fail.")


def generate_quiz(text, num_questions=5):
    """
    Sends the user's text to Gemini and asks for a JSON quiz.
    Returns a list of question dicts.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a university professor writing an exam.
    Generate exactly {num_questions} multiple-choice questions from the following text.

    Each question must have 4 options (A, B, C, D) and one correct answer.
    Include one tricky question and one easy warm-up.
    Test understanding, not just memorization.

    Format your response as a JSON array with this exact structure:
    [
      {{
        "question": "The question text here",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct": 0
      }}
    ]

    The "correct" field must be the index (0-3) of the right answer.
    Only return valid JSON. No extra text, no markdown, no code fences.

    Here is the text:
    {text}
    """

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Clean up if Gemini wraps the JSON in markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    quiz_data = json.loads(raw)
    return quiz_data


@app.route("/")
def home():
    """Serves the main page."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Receives text, returns quiz questions as JSON."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if len(text) < 50:
        return jsonify({"error": "Please provide more text (at least a paragraph)."}), 400

    try:
        quiz = generate_quiz(text)
        return jsonify({"quiz": quiz})
    except Exception as e:
        return jsonify({"error": f"AI generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
