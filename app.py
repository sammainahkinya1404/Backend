from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for React frontend

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Define system message for context
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an expert assistant who provides advice and answers questions "
        "strictly about business and investment opportunities in Kenya. "
        "You should not provide information or engage in discussions on any other topics. "
        "If asked about unrelated topics, politely redirect the user back to business inquiries."
    )
}

@app.route("/api/query", methods=["POST"])
def query_openai():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"error": "Empty input"}), 400

        # Construct the messages payload
        messages = [
            SYSTEM_MESSAGE,
            {"role": "user", "content": user_message}
        ]

        # Main Part: API Call to OpenAI using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",  # Use the GPT-4 model
            messages=messages
        )

        # Extract assistant's response
        bot_message = response.choices[0].message.content
        return jsonify({"response": bot_message})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process the request"}), 500

if __name__ == "__main__":
    app.run(debug=True)
