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

# System message to guide GPT-4 behavior
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a smart and realistic business assistant focused on Kenya. Your role is to:\n"
        "1. Suggest viable business and investment opportunities in Kenya.\n"
        "2. Generate startup budgets with details like rent, licenses, staff, and marketing.\n"
        "3. Estimate profits/losses with local pricing assumptions.\n"
        "4. Recommend ideas based on user's capital, location, and interests.\n"
        "5. Optionally offer timelines, digital tools, and registration steps in Kenya.\n"
        "6. Redirect any international queries back to local context.\n"
        "Be clear, practical, and user-focused."
    )
}

# Database model
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Handle GPT-4 chat query
@app.route("/api/query", methods=["POST"])
def query_openai():
    try:
        data = request.json
        session_id = data.get("session_id")
        user_message = data.get("message", "").strip()

        if not user_message or not session_id:
            return jsonify({"error": "Missing message or session_id"}), 400

        # Store user message
        db.session.add(ChatMessage(session_id=session_id, role="user", content=user_message))
        db.session.commit()

        # Retrieve full session history
        history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        messages = [SYSTEM_MESSAGE] + [{"role": msg.role, "content": msg.content} for msg in history]

        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        bot_message = response.choices[0].message.content

        # Store assistant message
        db.session.add(ChatMessage(session_id=session_id, role="assistant", content=bot_message))
        db.session.commit()

        return jsonify({"response": bot_message})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to process the request"}), 500

# Reset session memory
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

# Return message history
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

# Export history as plain text
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

    return jsonify({"text": "\n".join(export_lines)})

# Optional: manual DB init (Render debug)
@app.route("/init-db")
def init_db():
    with app.app_context():
        db.create_all()
    return "✅ Tables created."

# Auto-create tables on boot
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database table 'chat_message' ensured.")
    app.run(debug=True)
