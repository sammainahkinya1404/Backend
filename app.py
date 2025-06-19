from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_memory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# System prompt for GPT-4
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a smart and realistic business assistant focused on Kenya. Your role is to:\n"
        "1. Suggest viable business and investment opportunities in Kenya, tailored to local demand.\n"
        "2. Generate practical startup budgets for business ideas — breaking down costs like licenses, rent, inventory, salaries, marketing, and contingencies.\n"
        "3. Estimate realistic profit and loss projections for businesses over 6 to 12 months, showing assumptions (e.g., daily revenue, margins, seasonal trends).\n"
        "4. Recommend the most suitable business based on user inputs like available capital, location (urban/rural), interests, and risk appetite.\n"
        "5. Provide optional extras: startup timelines, legal steps (e.g., licensing in Kenya), digital tools (e.g., POS, inventory software), and links to government or support resources.\n"
        "6. If the user input is too vague, politely ask for more information (like capital or location) to give better guidance.\n\n"
        "⚠️ Only give advice grounded in the Kenyan market. If the user asks about international business, gently redirect back to local relevance.\n"
        "Be polite, realistic, and helpful. Think like a business consultant who understands everyday challenges."
    )
}

# Database model for storing messages
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Chat route
@app.route("/api/query", methods=["POST"])
def query_openai():
    try:
        data = request.json
        session_id = data.get("session_id")
        user_message = data.get("message", "").strip()

        if not user_message or not session_id:
            return jsonify({"error": "Missing message or session_id"}), 400

        # Save user message to DB
        db.session.add(ChatMessage(session_id=session_id, role="user", content=user_message))
        db.session.commit()

        # Retrieve conversation history
        history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        messages = [SYSTEM_MESSAGE] + [{"role": msg.role, "content": msg.content} for msg in history]

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        bot_message = response.choices[0].message.content

        # Save assistant response
        db.session.add(ChatMessage(session_id=session_id, role="assistant", content=bot_message))
        db.session.commit()

        return jsonify({"response": bot_message})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to process the request"}), 500

# Optional reset route
@app.route("/api/reset", methods=["POST"])
def reset_session():
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        ChatMessage.query.filter_by(session_id=session_id).delete()
        db.session.commit()
        return jsonify({"message": "Session reset successfully."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to reset session"}), 500
# Get session history (JSON)
@app.route("/api/history", methods=["GET"])
def get_history():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
    formatted = [{
        "timestamp": msg.timestamp.isoformat(),
        "role": msg.role,
        "content": msg.content
    } for msg in history]

    return jsonify({"messages": formatted})


# Export session as plain text
@app.route("/api/export", methods=["POST"])
def export_session():
    data = request.json
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()

    if not history:
        return jsonify({"error": "No messages found for session"}), 404

    export_lines = []
    for msg in history:
        prefix = "🧑 You" if msg.role == "user" else "🤖 Assistant"
        export_lines.append(f"{prefix} ({msg.timestamp.strftime('%Y-%m-%d %H:%M')}):\n{msg.content}\n")

    export_text = "\n".join(export_lines)

    return jsonify({"text": export_text})


# Automatically create DB tables at boot
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database table 'chat_message' ensured.")
    app.run(debug=True)
