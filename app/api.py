from fastapi import FastAPI
from pydantic import BaseModel

from app.chatbot import analyze_message, memory

# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Dynamic AI Chatbot API",
    description="API backend for the Dynamic AI Chatbot",
    version="1.0.0"
)

# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    message: str

# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Dynamic AI Chatbot API is running.",
        "status": "success"
    }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):
    result = analyze_message(
        user_message=request.message,
        memory=memory,
        conversation_id=request.conversation_id
    )

    return {
        "user_id": request.user_id,
        "conversation_id": request.conversation_id,
        "message": result["message"],
        "response": result["response"],
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "entities": result["entities"],
        "sentiment": result["sentiment"],
        "sentiment_score": result["sentiment_score"],
        "response_source": result["response_source"],
        "response_time": result["response_time"],
        "message_id": result["message_id"],
        "user_message_id": result["user_message_id"],
        "error": result["error"]
    }