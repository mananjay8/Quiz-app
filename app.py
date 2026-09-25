import os
import io
import json
import re
import PyPDF2
import PIL.Image
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set.")

# UPDATED MODEL: gemini-2.5-flash replaces the deprecated 1.5 model
MODEL_NAME = "gemini-2.5-flash"

def clean_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return raw.strip()

def build_prompt(text, num_questions, difficulty):
    difficulty_guide = {
        "easy": "Focus on definitions and simple recall.",
        "medium": "Test understanding and application.",
        "hard": "Test deep understanding, edge cases, and tricky distinctions."
    }
    return f"""You are an expert exam writer creating a quiz.
Generate exactly {num_questions} multiple-choice questions from the provided material.
DIFFICULTY: {difficulty.upper()} - {difficulty_guide.get(difficulty, difficulty_guide["medium"])}
STRICT RULES:
1. Each question must have exactly 4 options.
2. Exactly ONE option is correct.
3. Include a short explanation for why the correct answer is right.
4. Return ONLY a valid JSON array with this format:
[
  {{
    "question": "Question text?",
    "options": ["A", "B", "C", "D"],
    "correct": 2,
    "explanation": "Reason."
  }}
]
The "correct" field is the 0-based index of the right answer.
Here is the material:
{text}
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # 1. Get data from the form
    text = request.form.get("text", "").strip()
    num_questions = int(request.form.get("num_questions", 5))
    difficulty = request.form.get("difficulty", "medium").lower()
    file = request.files.get("file")

    # Clamp question count (max 50 to avoid API timeouts)
    num_questions = max(1, min(num_questions, 50))

    file_is_image = False
    if file and file.filename:
        filename = file.filename.lower()
        try:
            # Handle PDF
            if filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() or ""
                text = (text + "\n" + pdf_text).strip()
            
            # Handle Images (send image directly to Gemini)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                file_is_image = True
                
        except Exception as e:
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 400

    # 3. Validate input
    if len(text) < 50 and not file_is_image:
        return jsonify({"error": "Please provide at least a paragraph of text or an image."}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "Server is missing its API key."}), 500

    # 4. Call Gemini
    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config={"temperature": 0.7, "response_mime_type": "application/json"}
        )
        
        prompt = build_prompt(text if text else "Analyze the attached image and generate quiz questions.", num_questions, difficulty)
        
        # Prepare content parts (text + optional image)
        content_parts = [prompt]
        if file_is_image:
            # Read image file directly into Gemini
            img = PIL.Image.open(io.BytesIO(file.read()))
            content_parts.append(img)

        response = model.generate_content(content_parts)
        raw = clean_json(response.text)
        quiz = json.loads(raw)

        # 5. Validate structure
        if not isinstance(quiz, list) or len(quiz) == 0:
            raise ValueError("AI returned an empty quiz.")
        for q in quiz:
            if not all(k in q for k in ("question", "options", "correct")):
                raise ValueError("Malformed question structure.")
            if len(q["options"]) != 4:
                raise ValueError("Question does not have 4 options.")

        return jsonify({"quiz": quiz})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned invalid JSON. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
