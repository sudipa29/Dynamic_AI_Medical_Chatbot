
import time

from flask import Flask, jsonify, redirect, render_template, request

from app.chatbot_pipeline import process_message
from config import Config


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB request limit

APP_NAME = "Medical AI Assistant"


# ============================================================
# 2. DASHBOARD PAGES
# ============================================================

@app.route("/")
def home():
    """Open the Medical AI Assistant by default."""
    return redirect("/assistant")


@app.route("/assistant")
def assistant_page():
    """Render the medical chatbot interface."""
    return render_template(
        "assistant.html",
        app_name=APP_NAME,
    )


@app.route("/overview")
def overview_page():
    """Render chatbot performance analytics."""
    return render_template(
        "overview.html",
        app_name=APP_NAME,
    )


@app.route("/insights")
def insights_page():
    """Render the latest conversation insights."""
    return render_template(
        "insights.html",
        app_name=APP_NAME,
    )


# ============================================================
# 3. API HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Return the status expected by the dashboard."""
    return jsonify({
        "status": "healthy",
        "api_status": "Online",
        "application": APP_NAME,
    }), 200


# ============================================================
# 4. CHAT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a message using the existing medical pipeline."""

    started_at = time.perf_counter()

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "success": False,
                "error": "Please send a valid JSON request.",
                "api_status": "Online",
            }), 400

        user_message = str(payload.get("message") or "").strip()

        conversation_history = str(
            payload.get("conversation_history") or ""
        )

        if not user_message:
            return jsonify({
                "success": False,
                "error": "Please enter a message.",
                "api_status": "Online",
            }), 400

        max_length = 4000

        if len(user_message) > max_length:
            return jsonify({
                "success": False,
                "error": (
                    f"Your message must be no longer than "
                    f"{max_length} characters."
                ),
                "api_status": "Online",
            }), 400

        # Use the existing, tested medical chatbot pipeline.
        result = process_message(
            user_message=user_message,
            conversation_history=conversation_history,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "The chatbot pipeline did not return a dictionary."
            )

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

        response_text = str(result.get("response") or "").strip()

        if not response_text:
            raise ValueError(
                "The chatbot pipeline returned an empty response."
            )

        # Keep confidence values between 0 and 1 for the frontend.
        confidence = float(result.get("confidence", 0.0))
        second_confidence = float(
            result.get("second_confidence", 0.0)
        )

        confidence = max(0.0, min(1.0, confidence))
        second_confidence = max(
            0.0,
            min(1.0, second_confidence),
        )

        return jsonify({
            "success": True,
            "message": user_message,
            "response": response_text,

            "intent": result.get("intent", "unknown"),
            "confidence": confidence,

            "second_intent": result.get(
                "second_intent", "unknown"
            ),
            "second_confidence": second_confidence,

            "confidence_margin": (
                confidence - second_confidence
            ),

            "sentiment": result.get("sentiment", "Neutral"),
            "topic": result.get("topic", "Medical Inquiry"),

            "response_time_ms": elapsed_ms,
            "api_status": "Online",
        }), 200

    except Exception:
        app.logger.exception("Medical chatbot request failed.")

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

        return jsonify({
            "success": False,
            "error": (
                "The assistant could not process this message. "
                "Please check the server terminal for details "
                "and try again."
            ),
            "api_status": "Error",
            "response_time_ms": elapsed_ms,
        }), 500


# ============================================================
# 5. OPTIONAL ROOT-LEVEL ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "success": False,
        "error": "The request is too large.",
    }), 413


# ============================================================
# 6. START THE WEB SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Medical AI Assistant — Web Dashboard")
    print("Open: http://127.0.0.1:5000")
    print("API health: http://127.0.0.1:5000/api/health")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=Config.FLASK_DEBUG,
    )
