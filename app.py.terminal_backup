
import re

from app.intent_detection import predict_intent_details
from app.response_generator import generate_response


# ============================================================
# 1. BASIC CONVERSATION PHRASES
# ============================================================

GREETING_PHRASES = {
    "hi", "hello", "hey", "hi there", "hello there",
    "good morning", "good afternoon", "good evening",
}

THANK_YOU_PHRASES = {
    "thanks", "thank you", "thanks a lot",
    "thank you so much", "many thanks",
}

GOODBYE_PHRASES = {
    "bye", "goodbye", "see you", "see you later",
    "take care",
}


# ============================================================
# 2. MEDICAL TOPIC ALIASES
# ============================================================

MEDICAL_TOPIC_ALIASES = {
    "COVID-19": ["covid-19", "covid 19", "covid", "coronavirus"],
    "Diabetes": [
        "type 1 diabetes", "type 2 diabetes",
        "type one diabetes", "type two diabetes",
        "type i diabetes", "type ii diabetes",
        "diabetes", "diabetic",
    ],
    "Hypertension": [
        "high blood pressure", "blood pressure", "hypertension",
    ],
    "Heart Disease": ["heart disease", "cardiovascular disease"],
    "UTI": ["urinary tract infection", "urinary infection", "uti"],
    "Food Poisoning": ["food poisoning", "foodborne illness"],
    "Acid Reflux / GERD": [
        "acid reflux", "gerd",
        "gastroesophageal reflux disease",
        "gastro-oesophageal reflux disease",
    ],
    "Common Cold": ["common cold", "cold"],
    "Migraine": ["migraine", "migraines"],
    "Asthma": ["asthma"],
    "Dehydration": ["dehydration"],
    "Pneumonia": ["pneumonia"],
    "Arthritis": ["arthritis"],
    "Anemia": ["anemia", "anaemia"],
    "Influenza": ["influenza", "flu"],
    "Allergy": ["allergy", "allergies", "allergic reaction"],
    "Anxiety": ["anxiety", "anxiety disorder"],
    "Depression": ["depression", "depressive disorder"],
    "Obesity": ["obesity", "obese"],
    "High Cholesterol": ["high cholesterol", "cholesterol"],
    "Thyroid Disorders": [
        "thyroid disorder", "thyroid disorders",
        "thyroid disease", "thyroid problem", "thyroid problems",
    ],
    "Gastritis": ["gastritis"],
    "Vitamin D Deficiency": [
        "vitamin d deficiency", "low vitamin d",
    ],
}

TOPIC_CANONICAL_NAMES = {
    topic: topic for topic in MEDICAL_TOPIC_ALIASES
}


# ============================================================
# 3. FOLLOW-UP QUESTION PATTERNS
# ============================================================

FOLLOW_UP_PATTERNS = [
    r"\bwhat are (?:the )?symptoms\b",
    r"\bwhat symptoms\b",
    r"\bwhat causes\b",
    r"\bwhat are the causes\b",
    r"\bhow can i prevent\b",
    r"\bhow to prevent\b",
    r"\bhow is it prevented\b",
    r"\bhow can it be prevented\b",
    r"\bwhat about prevention\b",
    r"\bhow about prevention\b",
    r"\bhow is it diagnosed\b",
    r"\bwhat tests\b",
    r"\bwhich tests\b",
    r"\bhow is it treated\b",
    r"\bhow can it be managed\b",
    r"\bwhat treatment\b",
    r"\bwhat should i do\b",
    r"\bshould i see a doctor\b",
    r"\bwhen should i see a doctor\b",
    r"\bwhen should i seek medical care\b",
    r"\bis it serious\b",
    r"\bwhat are the complications\b",
    r"\bcan it be cured\b",
    r"\btell me more\b",
    r"\bexplain more\b",
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\bcan you explain that\b",
    r"\bwhy does it happen\b",
    r"\bhow does it happen\b",
]

SHORT_FOLLOW_UPS = {
    "why", "how", "what about it", "what about that",
    "what about this", "how about it", "how about that",
    "how about this", "tell me more", "explain more",
    "more details", "what are its symptoms",
    "what are the symptoms", "what causes it",
    "what causes that", "how to prevent it",
    "how can i prevent it", "how is it treated",
    "what should i do", "is it serious", "can it be cured",
    "what tests are needed", "what about prevention",
    "how about prevention", "what about symptoms",
    "what about treatment", "what about causes",
    "what about diagnosis",
}


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_message(message):
    """Normalize text for phrase matching."""
    message = str(message or "").lower().strip()
    message = re.sub(r"[.!?,;:]+", " ", message)
    message = re.sub(r"\s+", " ", message)
    return message.strip()


def _compile_phrase_pattern(phrase):
    """Match a phrase without matching it inside another word."""
    escaped = re.escape(phrase.lower())
    escaped = escaped.replace(r"\ ", r"\s+")

    return re.compile(
        r"(?<!\w)" + escaped + r"(?!\w)",
        re.IGNORECASE,
    )


_COMPILED_TOPIC_ALIASES = [
    (topic, alias, _compile_phrase_pattern(alias))
    for topic, aliases in MEDICAL_TOPIC_ALIASES.items()
    for alias in aliases
]


# ============================================================
# 5. BASIC INTENT DETECTION
# ============================================================

def detect_basic_intent(user_message):
    """Detect exact greetings, thanks, and goodbye phrases."""
    normalized = normalize_message(user_message)

    if normalized in GREETING_PHRASES:
        return "greeting"

    if normalized in THANK_YOU_PHRASES:
        return "thank_you"

    if normalized in GOODBYE_PHRASES:
        return "goodbye"

    return None


# ============================================================
# 6. SENTIMENT DETECTION
# ============================================================

POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "helpful",
    "thanks", "thank", "better", "wonderful", "useful",
    "relieved",
}

NEGATIVE_WORDS = {
    "bad", "pain", "painful", "worried", "scared", "afraid",
    "sad", "angry", "terrible", "severe", "difficult",
    "worse", "unwell", "sick", "anxious",
}


def detect_sentiment(user_message):
    """Return a lightweight keyword-based sentiment label."""
    words = set(
        re.findall(r"\b[\w'-]+\b", normalize_message(user_message))
    )

    positive_count = len(words.intersection(POSITIVE_WORDS))
    negative_count = len(words.intersection(NEGATIVE_WORDS))

    if positive_count > negative_count:
        return "Positive"

    if negative_count > positive_count:
        return "Negative"

    return "Neutral"


# ============================================================
# 7. MEDICAL TOPIC DETECTION
# ============================================================

def find_medical_topic_in_text(text):
    """Find the latest recognized medical topic, preferring longer aliases."""
    normalized = normalize_message(text)

    if not normalized:
        return None

    matches = []

    for topic, alias, pattern in _COMPILED_TOPIC_ALIASES:
        for match in pattern.finditer(normalized):
            matches.append({
                "topic": topic,
                "start": match.start(),
                "end": match.end(),
                "length": len(alias),
            })

    if not matches:
        return None

    matches.sort(
        key=lambda item: (item["end"], item["length"]),
        reverse=True,
    )

    return matches[0]["topic"]


def detect_topic(user_message, detected_intent):
    """Return a dashboard-friendly topic label."""
    topic = find_medical_topic_in_text(user_message)

    if topic:
        return TOPIC_CANONICAL_NAMES.get(topic, topic)

    if detected_intent in {"greeting", "thank_you", "goodbye"}:
        return "General Conversation"

    if detected_intent == "unknown":
        return "General Inquiry"

    return "Medical Inquiry"


# ============================================================
# 8. CONTEXTUAL FOLLOW-UP DETECTION
# ============================================================

def is_contextual_follow_up(user_message):
    """Determine whether the message depends on a previous topic."""
    normalized = normalize_message(user_message)

    if not normalized:
        return False

    if find_medical_topic_in_text(normalized):
        return False

    if normalized in SHORT_FOLLOW_UPS:
        return True

    return any(
        re.search(pattern, normalized)
        for pattern in FOLLOW_UP_PATTERNS
    )


# ============================================================
# 9. CONVERSATION HISTORY AND CONTEXT
# ============================================================

def _extract_history_lines(conversation_history):
    """Extract User and Assistant messages from stored history."""
    user_messages = []
    assistant_messages = []

    if not conversation_history:
        return user_messages, assistant_messages

    for line in str(conversation_history).splitlines():
        stripped = line.strip()

        user_match = re.match(r"^User:\s*(.*)$", stripped, re.I)
        assistant_match = re.match(
            r"^Assistant:\s*(.*)$", stripped, re.I
        )

        if user_match and user_match.group(1).strip():
            user_messages.append(user_match.group(1).strip())

        elif assistant_match and assistant_match.group(1).strip():
            assistant_messages.append(assistant_match.group(1).strip())

    return user_messages, assistant_messages


def extract_medical_topic(conversation_history):
    """Find the most recent recognized topic in the conversation."""
    user_messages, assistant_messages = _extract_history_lines(
        conversation_history
    )

    for message in reversed(user_messages):
        topic = find_medical_topic_in_text(message)
        if topic:
            return topic

    for message in reversed(assistant_messages):
        topic = find_medical_topic_in_text(message)
        if topic:
            return topic

    return None


def build_contextual_message(user_message, conversation_history):
    """Expand a follow-up question using the previous medical topic."""
    if not is_contextual_follow_up(user_message):
        return user_message

    topic = extract_medical_topic(conversation_history)

    if not topic:
        return None

    return f"Regarding {topic}: {user_message.strip()}"


# ============================================================
# 10. INTENT CONFIDENCE HELPERS
# ============================================================

def _safe_float(value, default=0.0):
    """Convert a value to float safely."""
    try:
        result = float(value)

        if result != result:
            return default

        return result

    except (TypeError, ValueError, OverflowError):
        return default


def _parse_intent_details(details):
    """Normalize the intent model result and recalculate its margin."""
    default_result = {
        "intent": "unknown",
        "confidence": 0.0,
        "second_intent": "unknown",
        "second_confidence": 0.0,
        "confidence_margin": 0.0,
    }

    if details is None:
        return default_result.copy()

    if isinstance(details, dict):
        intent = details.get("intent", "unknown")
        confidence = _safe_float(details.get("confidence", 0.0))
        second_intent = details.get("second_intent", "unknown")
        second_confidence = _safe_float(
            details.get("second_confidence", 0.0)
        )

    elif isinstance(details, (tuple, list)):
        values = list(details)

        if not values:
            return default_result.copy()

        intent = values[0]
        confidence = _safe_float(values[1]) if len(values) > 1 else 0.0
        second_intent = values[2] if len(values) > 2 else "unknown"
        second_confidence = (
            _safe_float(values[3]) if len(values) > 3 else 0.0
        )

    else:
        print(
            "[Intent Detection] Unsupported result type: "
            f"{type(details).__name__}"
        )
        return default_result.copy()

    confidence = max(0.0, min(1.0, confidence))
    second_confidence = max(0.0, min(1.0, second_confidence))

    return {
        "intent": str(intent or "unknown"),
        "confidence": confidence,
        "second_intent": str(second_intent or "unknown"),
        "second_confidence": second_confidence,
        "confidence_margin": confidence - second_confidence,
    }


# ============================================================
# 11. MAIN CHATBOT PIPELINE
# ============================================================

def process_message(user_message, conversation_history=""):
    """Process one message and return its response and analytics."""

    original_message = str(user_message or "").strip()

    if not original_message:
        return {
            "user_message": "",
            "intent": "unknown",
            "confidence": 0.0,
            "second_intent": "unknown",
            "second_confidence": 0.0,
            "confidence_margin": 0.0,
            "sentiment": "Neutral",
            "topic": "General Inquiry",
            "response": "Please enter a question so I can help you.",
        }

    response_message = build_contextual_message(
        original_message,
        conversation_history,
    )

    if response_message is None:
        return {
            "user_message": original_message,
            "intent": "unknown",
            "confidence": 0.0,
            "second_intent": "unknown",
            "second_confidence": 0.0,
            "confidence_margin": 0.0,
            "sentiment": detect_sentiment(original_message),
            "topic": "Medical Inquiry",
            "response": (
                "Which medical condition are you asking about? "
                "Please mention a condition such as asthma or diabetes "
                "first. Then I can help with its symptoms, causes, "
                "prevention, diagnosis, or general management."
            ),
        }

    # Exact basic conversation phrases bypass the trained model.
    basic_intent = detect_basic_intent(response_message)

    if basic_intent:
        detected_intent = basic_intent
        confidence = 1.0
        second_intent = "unknown"
        second_confidence = 0.0
    else:
        # Medical questions use the trained intent model.
        raw_details = predict_intent_details(response_message)
        details = _parse_intent_details(raw_details)

        detected_intent = details["intent"]
        confidence = details["confidence"]
        second_intent = details["second_intent"]
        second_confidence = details["second_confidence"]

    confidence = max(0.0, min(1.0, _safe_float(confidence)))
    second_confidence = max(
        0.0,
        min(1.0, _safe_float(second_confidence)),
    )

    # Confidence values are fractions between 0 and 1.
    # The frontend converts these to percentages for display.
    confidence_margin = confidence - second_confidence

    sentiment = detect_sentiment(original_message)
    topic = detect_topic(response_message, detected_intent)

    response = generate_response(
        user_message=response_message,
        intent=detected_intent,
        conversation_history=conversation_history,
    )

    return {
        "user_message": original_message,
        "intent": detected_intent,
        "confidence": round(confidence, 4),
        "second_intent": second_intent,
        "second_confidence": round(second_confidence, 4),
        "confidence_margin": round(confidence_margin, 4),
        "sentiment": sentiment,
        "topic": topic,
        "response": response,
    }


# ============================================================
# 12. COMMAND-LINE TESTING
# ============================================================

def start_chatbot():
    """Run the medical chatbot in the terminal."""
    print("=" * 60)
    print("Medical AI Assistant")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    conversation_history = ""

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if normalize_message(user_message) in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        result = process_message(
            user_message=user_message,
            conversation_history=conversation_history,
        )

        print("\nAssistant:", result["response"])
        print("Intent:", result["intent"])
        print("Confidence:", result["confidence"])
        print("Second Intent:", result["second_intent"])
        print("Second Confidence:", result["second_confidence"])
        print("Confidence Margin:", result["confidence_margin"])
        print("Sentiment:", result["sentiment"])
        print("Topic:", result["topic"])

        conversation_history += f"\nUser: {user_message}"
        conversation_history += f"\nAssistant: {result['response']}"


if __name__ == "__main__":
    start_chatbot()
