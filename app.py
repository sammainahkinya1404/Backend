from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

# Load environment variables
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_memory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize DeepSeek client (OpenAI-compatible)
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# ============================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================

class BudgetItem(BaseModel):
    """Single item in a budget breakdown"""
    item: str = Field(description="Name of the budget item")
    description: str = Field(description="Brief description of what this covers")
    cost_low: int = Field(description="Lower estimate in KES")
    cost_high: int = Field(description="Upper estimate in KES")
    priority: Literal["essential", "recommended", "optional"] = Field(description="Priority level")

class LicenseStep(BaseModel):
    """A step in obtaining a license"""
    step_number: int = Field(description="Step number in sequence")
    action: str = Field(description="What to do")
    where: str = Field(description="Where to do it (e.g., eCitizen, County Office)")
    cost: str = Field(description="Cost in KES or 'Free'")
    duration: str = Field(description="How long it takes")

class License(BaseModel):
    """License or permit requirement"""
    name: str = Field(description="Name of the license/permit")
    authority: str = Field(description="Issuing authority")
    total_cost: str = Field(description="Total cost range in KES")
    processing_time: str = Field(description="Total processing time")
    steps: List[LicenseStep] = Field(description="Steps to obtain this license")

class Supplier(BaseModel):
    """Supplier or market information"""
    name: str = Field(description="Supplier/market name")
    location: str = Field(description="Physical location")
    what_to_buy: str = Field(description="What you can source here")
    price_range: str = Field(description="Typical price range")
    tips: str = Field(description="Negotiation or buying tips")

class MonthlyProjection(BaseModel):
    """Monthly P&L projection"""
    month: int = Field(description="Month number (1-12)")
    revenue: int = Field(description="Expected revenue in KES")
    expenses: int = Field(description="Expected expenses in KES")
    profit: int = Field(description="Net profit in KES")
    notes: str = Field(description="Key assumptions or events")

class BusinessRecommendation(BaseModel):
    """A recommended business opportunity"""
    name: str = Field(description="Business name/type")
    tagline: str = Field(description="Short catchy description")
    capital_required_low: int = Field(description="Minimum startup capital in KES")
    capital_required_high: int = Field(description="Maximum startup capital in KES")
    monthly_profit_low: int = Field(description="Minimum expected monthly profit in KES")
    monthly_profit_high: int = Field(description="Maximum expected monthly profit in KES")
    risk_level: Literal["low", "medium", "high"] = Field(description="Risk level")
    time_commitment: Literal["part-time", "full-time", "flexible"] = Field(description="Time needed")
    skills_needed: List[str] = Field(description="Key skills required")
    why_suitable: str = Field(description="Why this suits the user")
    quick_start_steps: List[str] = Field(description="3-5 steps to start quickly")

class FollowUpQuestion(BaseModel):
    """A follow-up question to gather more info"""
    question: str = Field(description="The question to ask")
    why_asking: str = Field(description="Brief explanation of why this helps")
    options: Optional[List[str]] = Field(description="Suggested options if applicable", default=None)

class ActionStep(BaseModel):
    """A step in the action plan"""
    week: str = Field(description="Week or timeframe (e.g., 'Week 1', 'Week 2-3')")
    title: str = Field(description="Step title")
    tasks: List[str] = Field(description="Specific tasks to complete")
    estimated_cost: str = Field(description="Cost for this phase in KES")

class AssistantResponse(BaseModel):
    """Main structured response from the assistant"""
    message: str = Field(description="Main conversational message to user")
    response_type: Literal[
        "greeting",
        "gathering_info",
        "recommendations",
        "budget_breakdown",
        "license_guide",
        "supplier_guide",
        "profit_projection",
        "action_plan",
        "general_advice",
        "clarification"
    ] = Field(description="Type of response being given")

    # Optional structured data based on response type
    follow_up_questions: Optional[List[FollowUpQuestion]] = Field(default=None)
    business_recommendations: Optional[List[BusinessRecommendation]] = Field(default=None)
    budget_items: Optional[List[BudgetItem]] = Field(default=None)
    budget_total_low: Optional[int] = Field(default=None)
    budget_total_high: Optional[int] = Field(default=None)
    licenses: Optional[List[License]] = Field(default=None)
    suppliers: Optional[List[Supplier]] = Field(default=None)
    monthly_projections: Optional[List[MonthlyProjection]] = Field(default=None)
    action_steps: Optional[List[ActionStep]] = Field(default=None)

    # Conversation context
    user_profile_summary: Optional[str] = Field(default=None, description="Summary of what we know about user")
    next_suggested_topic: Optional[str] = Field(default=None, description="What to explore next")

# ============================================
# KENYA BUSINESS KNOWLEDGE BASE
# ============================================

KENYA_BUSINESS_KNOWLEDGE = """
## KENYA BUSINESS REGISTRATION COSTS (2025)
- Business Name Registration: KES 950
- Private Limited Company: KES 10,650
- Public Limited Company: KES 10,650
- Limited Liability Partnership: KES 25,000
- Single Business Permit: KES 5,000 - 50,000 (varies by county and business size)

## REGISTRATION PROCESS
1. Create eCitizen account (https://accounts.ecitizen.go.ke)
2. Search and reserve business name (KES 150 for search, valid 30 days)
3. Prepare documents: ID/Passport copy, KRA PIN certificate
4. Submit registration via Business Registration Service (BRS)
5. Receive certificate (3-14 days depending on business type)
6. Register for KRA PIN (automatic with CR1 form)
7. Apply for Single Business Permit from County Government

## COMMON LICENSES BY BUSINESS TYPE
- Food Business: Health Certificate (KES 3,000-10,000), Food Handler Certificate (KES 500/person)
- Retail Shop: Single Business Permit only
- Salon/Barber: Single Business Permit + Health Permit
- M-PESA Agent: Safaricom Agent Registration (float: KES 20,000+)
- Manufacturing: NEMA License (KES 10,000+), KEBS certification if applicable
- Transport (Boda/Taxi): NTSA compliance, Insurance, County permits

## POPULAR BUSINESS IDEAS BY CAPITAL
### KES 10,000 - 30,000
- Vegetable/Fruit hawking: KES 10-20K startup, KES 500-2,000/day profit
- Mandazi/Chapati business: KES 15-25K startup, KES 800-2,500/day profit
- Phone accessories: KES 20-30K startup, KES 500-1,500/day profit
- Social media management: KES 5-15K startup, KES 10-50K/month profit

### KES 30,000 - 100,000
- M-PESA/Airtel Money agent: KES 30-50K startup, KES 500-3,000/day profit
- Mitumba (second-hand clothes): KES 30-80K startup, 100-300% markup
- Fast food kiosk: KES 40-80K startup, KES 3,000-15,000/day profit
- Cleaning services: KES 30-60K startup, KES 50-150K/month profit
- Poultry farming (layers/broilers): KES 50-100K startup, variable returns

### KES 100,000 - 500,000
- Salon/Barbershop: KES 100-300K startup, KES 3,000-10,000/day profit
- Boutique shop: KES 150-400K startup, KES 5,000-20,000/day profit
- Cyber cafe/printing: KES 150-350K startup, KES 3,000-8,000/day profit
- Boda boda spare parts: KES 100-200K startup, KES 3,000-10,000/day profit
- Hardware shop: KES 200-500K startup, KES 5,000-15,000/day profit

## POPULAR MARKETS & SUPPLIERS IN KENYA
### Nairobi
- Gikomba Market: Mitumba, fabrics, shoes (largest open-air market)
- Nyamakima: Electronics, phone accessories, wholesale goods
- Kamukunji: Hardware, tools, machinery
- Muthurwa: Groceries, cereals, household goods wholesale
- Industrial Area: Manufacturing supplies, packaging materials

### Mombasa
- Kongowea Market: General wholesale
- Marikiti Market: Fresh produce wholesale

### Other Sources
- Alibaba/1688.com: Import directly (minimum orders apply)
- Kenya China Trade Center: Electronics, household goods
- Jiji.co.ke / Facebook Marketplace: Second-hand equipment

## COUNTY-SPECIFIC PERMIT COSTS (Approximate)
- Nairobi: KES 10,000 - 100,000 (UBP - Unified Business Permit)
- Mombasa: KES 8,000 - 80,000
- Kisumu: KES 5,000 - 50,000
- Nakuru: KES 5,000 - 40,000
- Smaller counties: KES 3,000 - 20,000

## COMMON MISTAKES TO AVOID
1. Not registering the business (risk of fines up to KES 50,000)
2. Ignoring county permits (business can be shut down)
3. Not keeping financial records (tax issues with KRA)
4. Underestimating working capital needs
5. Poor location choice
6. Not researching competition
7. Mixing personal and business finances
"""

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_MESSAGE = {
    "role": "system",
    "content": f"""You are "Biashara Buddy" - Kenya's most helpful AI business consultant. You help Kenyans start and grow successful businesses.

## YOUR PERSONALITY
- Warm, encouraging, and practical
- You speak like a knowledgeable Kenyan friend who genuinely wants to see users succeed
- Use simple language, avoid jargon
- Be realistic about challenges but always solution-oriented
- Occasionally use Swahili phrases naturally (e.g., "Sawa!", "Poa sana!", "Hakuna shida")

## YOUR CAPABILITIES
1. **Business Discovery**: Help users find the perfect business based on their capital, skills, location, and goals
2. **Budget Planning**: Create detailed, realistic startup budgets with itemized costs
3. **License Guidance**: Explain exactly which licenses/permits are needed and how to get them
4. **Supplier Connections**: Tell users WHERE to buy inventory/equipment and at what prices
5. **Profit Projections**: Create realistic 6-12 month P&L projections
6. **Action Plans**: Provide week-by-week launch timelines

## CONVERSATION APPROACH
1. **Start by understanding the user**: If they're new, warmly welcome them and ask about:
   - Available capital (be specific: "How much money can you invest? E.g., KES 20,000, 50,000, 100,000?")
   - Location (county and whether urban/rural)
   - Time availability (full-time or side hustle)
   - Skills or interests
   - Risk tolerance

2. **Give structured, actionable responses**:
   - When recommending businesses, give 2-4 options with clear comparisons
   - When giving budgets, itemize EVERYTHING (licenses, rent, inventory, contingency)
   - When explaining licenses, give step-by-step with costs and timelines
   - When suggesting suppliers, give specific names and locations

3. **Always provide next steps**: End responses with clear guidance on what to do next

## IMPORTANT RULES
- ONLY give advice relevant to Kenya
- Use REAL, CURRENT prices and costs (2024-2025 data)
- Be honest about risks and challenges
- If user's capital is very low, suggest realistic options (don't promise unrealistic returns)
- Always recommend registering the business legally
- Suggest starting small and scaling up

## KENYA BUSINESS DATA
{KENYA_BUSINESS_KNOWLEDGE}

## RESPONSE FORMAT
You MUST respond with valid JSON. Always include these required fields:
{{
  "message": "Your friendly response message here",
  "response_type": "one of: greeting, gathering_info, recommendations, budget_breakdown, license_guide, supplier_guide, profit_projection, action_plan, general_advice, clarification",
  "user_profile_summary": "Brief summary of what you know about the user (optional)",
  "next_suggested_topic": "What to discuss next (optional)"
}}

Include these OPTIONAL fields based on what you're providing:

For business recommendations (response_type: "recommendations"):
"business_recommendations": [{{
  "name": "Business name",
  "tagline": "Short catchy description",
  "capital_required_low": 10000,
  "capital_required_high": 50000,
  "monthly_profit_low": 5000,
  "monthly_profit_high": 20000,
  "risk_level": "low/medium/high",
  "time_commitment": "part-time/full-time/flexible",
  "skills_needed": ["skill1", "skill2"],
  "why_suitable": "Why this business suits the user",
  "quick_start_steps": ["Step 1", "Step 2", "Step 3"]
}}]

For budget breakdowns (response_type: "budget_breakdown"):
"budget_items": [{{
  "item": "Item name",
  "description": "What this covers",
  "cost_low": 1000,
  "cost_high": 5000,
  "priority": "essential/recommended/optional"
}}],
"budget_total_low": 50000,
"budget_total_high": 100000

For license guides (response_type: "license_guide"):
"licenses": [{{
  "name": "License name",
  "authority": "Issuing authority",
  "total_cost": "KES 5,000 - 10,000",
  "processing_time": "5-10 days",
  "steps": [{{
    "step_number": 1,
    "action": "What to do",
    "where": "Where to do it",
    "cost": "KES 1,000",
    "duration": "1-2 days"
  }}]
}}]

For supplier guides (response_type: "supplier_guide"):
"suppliers": [{{
  "name": "Market/Supplier name",
  "location": "Physical location",
  "what_to_buy": "Products available",
  "price_range": "KES 100 - 5,000",
  "tips": "Negotiation tips"
}}]

For profit projections (response_type: "profit_projection"):
"monthly_projections": [{{
  "month": 1,
  "revenue": 50000,
  "expenses": 30000,
  "profit": 20000,
  "notes": "Key assumptions"
}}]

For action plans (response_type: "action_plan"):
"action_steps": [{{
  "week": "Week 1",
  "title": "Step title",
  "tasks": ["Task 1", "Task 2"],
  "estimated_cost": "KES 10,000"
}}]

For follow-up questions (when gathering info):
"follow_up_questions": [{{
  "question": "Your question?",
  "why_asking": "Why this helps",
  "options": ["Option 1", "Option 2", "Option 3"]
}}]
"""
}

# ============================================
# DATABASE MODELS
# ============================================

class ChatMessage(db.Model):
    """Stores chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    structured_data = db.Column(db.Text, nullable=True)  # JSON string of structured response
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UserProfile(db.Model):
    """Stores user profile information gathered during conversation"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    capital_available = db.Column(db.String(50), nullable=True)
    location_county = db.Column(db.String(50), nullable=True)
    location_type = db.Column(db.String(20), nullable=True)  # urban/rural
    time_commitment = db.Column(db.String(20), nullable=True)  # full-time/part-time
    skills = db.Column(db.Text, nullable=True)
    interests = db.Column(db.Text, nullable=True)
    risk_tolerance = db.Column(db.String(20), nullable=True)
    selected_business = db.Column(db.String(100), nullable=True)
    conversation_stage = db.Column(db.String(30), default="discovery")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Ensure tables exist
with app.app_context():
    db.create_all()
    print("Database tables created successfully.")

# ============================================
# API ROUTES
# ============================================

@app.route("/api/query", methods=["POST"])
def query_openai():
    """Main chat endpoint with structured outputs"""
    try:
        data = request.json
        session_id = data.get("session_id")
        user_message = data.get("message", "").strip()

        if not user_message or not session_id:
            return jsonify({"error": "Missing message or session_id"}), 400

        # Save user message
        db.session.add(ChatMessage(
            session_id=session_id,
            role="user",
            content=user_message
        ))
        db.session.commit()

        # Get or create user profile
        profile = UserProfile.query.filter_by(session_id=session_id).first()
        if not profile:
            profile = UserProfile(session_id=session_id)
            db.session.add(profile)
            db.session.commit()

        # Build conversation history
        history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()

        # Include profile context in system message
        profile_context = f"""
## CURRENT USER PROFILE
- Capital: {profile.capital_available or 'Unknown'}
- Location: {profile.location_county or 'Unknown'} ({profile.location_type or 'Unknown'})
- Time: {profile.time_commitment or 'Unknown'}
- Skills: {profile.skills or 'Unknown'}
- Interests: {profile.interests or 'Unknown'}
- Risk Tolerance: {profile.risk_tolerance or 'Unknown'}
- Selected Business: {profile.selected_business or 'None yet'}
- Conversation Stage: {profile.conversation_stage}
"""

        system_with_context = {
            "role": "system",
            "content": SYSTEM_MESSAGE["content"] + profile_context
        }

        messages = [system_with_context] + [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]

        # Call DeepSeek with JSON mode
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=4096
        )

        # Parse the JSON response
        response_text = response.choices[0].message.content

        try:
            response_dict = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            response_dict = {
                "message": response_text,
                "response_type": "general_advice"
            }

        # Validate with Pydantic (optional, for data consistency)
        try:
            validated_response = AssistantResponse(**response_dict)
            response_dict = validated_response.model_dump()
        except Exception as validation_error:
            print(f"Validation warning: {validation_error}")
            # Continue with unvalidated response if validation fails
            if "message" not in response_dict:
                response_dict["message"] = response_text

        # Save assistant message with structured data
        db.session.add(ChatMessage(
            session_id=session_id,
            role="assistant",
            content=response_dict.get("message", ""),
            structured_data=json.dumps(response_dict)
        ))
        db.session.commit()

        return jsonify({
            "response": response_dict,
            "profile": {
                "stage": profile.conversation_stage,
                "capital": profile.capital_available,
                "location": profile.location_county,
                "selected_business": profile.selected_business
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500


@app.route("/api/profile", methods=["GET"])
def get_profile():
    """Get user profile for a session"""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    profile = UserProfile.query.filter_by(session_id=session_id).first()
    if not profile:
        return jsonify({"profile": None})

    return jsonify({
        "profile": {
            "capital_available": profile.capital_available,
            "location_county": profile.location_county,
            "location_type": profile.location_type,
            "time_commitment": profile.time_commitment,
            "skills": profile.skills,
            "interests": profile.interests,
            "risk_tolerance": profile.risk_tolerance,
            "selected_business": profile.selected_business,
            "conversation_stage": profile.conversation_stage
        }
    })


@app.route("/api/profile", methods=["POST"])
def update_profile():
    """Update user profile"""
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        profile = UserProfile.query.filter_by(session_id=session_id).first()
        if not profile:
            profile = UserProfile(session_id=session_id)
            db.session.add(profile)

        # Update fields if provided
        updatable_fields = [
            'capital_available', 'location_county', 'location_type',
            'time_commitment', 'skills', 'interests', 'risk_tolerance',
            'selected_business', 'conversation_stage'
        ]

        for field in updatable_fields:
            if field in data:
                setattr(profile, field, data[field])

        db.session.commit()
        return jsonify({"message": "Profile updated successfully"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to update profile"}), 500


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset session - clear messages and profile"""
    try:
        session_id = request.json.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        ChatMessage.query.filter_by(session_id=session_id).delete()
        UserProfile.query.filter_by(session_id=session_id).delete()
        db.session.commit()

        return jsonify({"message": "Session reset successfully"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to reset session"}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get chat history with structured data"""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()

    formatted = []
    for msg in history:
        entry = {
            "timestamp": msg.timestamp.isoformat(),
            "role": msg.role,
            "content": msg.content
        }
        if msg.structured_data:
            entry["structured_data"] = json.loads(msg.structured_data)
        formatted.append(entry)

    return jsonify({"messages": formatted})


@app.route("/api/export", methods=["POST"])
def export_session():
    """Export conversation as formatted text"""
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
    if not history:
        return jsonify({"error": "No messages found"}), 404

    profile = UserProfile.query.filter_by(session_id=session_id).first()

    export_lines = ["=" * 50]
    export_lines.append("BIASHARA BUDDY - BUSINESS CONSULTATION EXPORT")
    export_lines.append("=" * 50)
    export_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    export_lines.append("")

    if profile:
        export_lines.append("USER PROFILE:")
        export_lines.append(f"  Capital: {profile.capital_available or 'Not specified'}")
        export_lines.append(f"  Location: {profile.location_county or 'Not specified'}")
        export_lines.append(f"  Selected Business: {profile.selected_business or 'Not selected'}")
        export_lines.append("")

    export_lines.append("CONVERSATION:")
    export_lines.append("-" * 50)

    for msg in history:
        prefix = "YOU" if msg.role == "user" else "BIASHARA BUDDY"
        export_lines.append(f"\n[{prefix}] ({msg.timestamp.strftime('%H:%M')})")
        export_lines.append(msg.content)

    return jsonify({"text": "\n".join(export_lines)})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Biashara Buddy API",
        "version": "2.0.0"
    })


# Run locally
if __name__ == "__main__":
    app.run(debug=True)
