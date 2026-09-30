import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured. Please add it to the .env file."
        }), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{SYSTEM_PROMPT}\n\nFriend's message:\n{message}",
        )

        reply = (response.text or "").strip()

        if not reply:
            reply = "Sorry, I couldn't think of a reply right now. Try again."

        return jsonify({"reply": reply})

    except Exception as exc:
        return jsonify({
            "error": f"Unable to get a response from Gemini: {exc}"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
