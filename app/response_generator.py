
import re

from app.faq_response import (
    get_faq_response,
    detect_medical_topic,
)
from app.gpt_client import generate_gpt_response


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Keep local medical retrieval as the default.
# Gemini is used when local retrieval cannot provide an answer.
USE_GEMINI_FOR_MEDICAL_ANSWERS = False

MEDICAL_DISCLAIMER = (
    "This is general educational information, not a personal "
    "diagnosis or treatment plan. Consult a qualified healthcare "
    "professional for individual medical advice."
)


# ============================================================
# 2. MEDICAL ASSISTANT INSTRUCTIONS
# ============================================================

MEDICAL_SYSTEM_INSTRUCTIONS = """
You are a responsible Medical AI Assistant.

Provide general educational information about medical conditions,
symptoms, prevention, diagnosis, and general management.

Rules:
1. Never claim to diagnose a person.
2. Do not prescribe medication or personalized dosages.
3. Encourage professional evaluation for individual medical advice.
4. Never invent facts, sources, or medical references.
5. Do not present unsupported information as verified.
6. For possible emergencies, advise immediate emergency assistance.
7. Politely redirect questions unrelated to medicine or health.
8. Use conversation history to understand follow-up questions.
9. Ask for clarification when the medical condition is unclear.
10. Use clear, readable language.
11. Do not claim to have examined the user.
12. Do not append the application's standard disclaimer yourself.
13. Never delay emergency advice while asking follow-up questions.
14. Say "local emergency services" rather than assuming a country.
15. Finish every answer with a complete sentence.
16. Do not leave unfinished headings, quotations, bullets, or lists.
17. Use normal spaces between all words.
18. Prefer concise, readable answers.
"""


# ============================================================
# 3. EMERGENCY DETECTION
# ============================================================

EMERGENCY_PATTERNS = [
    r"\b(?:cannot|can't)\s+breathe\b",
    r"\b(?:cannot|can't)\s+breath(?:e|ing)?\b",
    r"\b(?:cannot|can't)\s+get\s+(?:any\s+)?air\b",
    r"\b(?:not breathing|stopped breathing)\b",
    r"\b(?:severe|serious|extreme)\s+breathing difficulty\b",
    r"\b(?:struggling|unable)\s+to\s+breathe\b",
    r"\b(?:difficulty breathing|trouble breathing)\b",
    r"\bsevere shortness of breath\b",
    r"\b(?:severe|crushing|heavy|intense)\s+chest\s+pain\b",
    r"\b(?:chest pain|chest pressure)\b",
    r"\b(?:stroke|possible stroke|suspected stroke)\b",
    r"\b(?:facial weakness|face weakness)\b",
    r"\b(?:face drooping|facial drooping|drooping face)\b",
    r"\b(?:sudden weakness|sudden numbness)\b",
    r"\b(?:slurred speech|sudden speech difficulty)\b",
    r"\b(?:difficulty speaking|trouble speaking)\b",
    r"\b(?:difficulty talking|trouble talking)\b",
    r"\b(?:cannot speak|can't speak|unable to speak)\b",
    r"\b(?:difficulty understanding speech)\b",
    r"\b(?:unconscious|unresponsive)\b",
    r"\b(?:severe bleeding|uncontrolled bleeding)\b",
    r"\b(?:prolonged seizure|seizure lasting)\b",
    r"\bworst headache of (?:my|his|her|their) life\b",
    r"\bsudden severe headache\b",
    r"\b(?:suspected|possible) sepsis\b",
    (
        r"\b(?:i think i'm having|i think i am having|"
        r"i may be having|i might be having)\s+(?:a )?heart attack\b"
    ),
    r"\b(?:i am having|i'm having|im having)\s+(?:a )?heart attack\b",
    (
        r"\b(?:i have|i'm having|im having)\s+"
        r"(?:severe )?chest pain\b"
    ),
    (
        r"\bchest pain\b.*\b"
        r"(?:shortness of breath|cold sweat|fainting)\b"
    ),
]

EMERGENCY_RESPONSE = (
    "Your message may describe a medical emergency. Contact "
    "emergency services or go to the nearest emergency department "
    "now. If you are in India, call 112; otherwise, call your "
    "local emergency number. Do not delay urgent care while "
    "chatting with an AI assistant. If possible, ask someone "
    "nearby to help. Do not drive yourself if you may be having "
    "a heart attack."
)


def is_emergency_message(user_message):
    """Detect possible emergencies before retrieval or Gemini."""

    if not user_message:
        return False

    text = re.sub(
        r"\s+",
        " ",
        str(user_message).lower(),
    ).strip()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    chest_pain = bool(
        re.search(
            r"\b(chest pain|chest pressure|chest tightness)\b",
            text,
        )
    )

    breathing_problem = bool(
        re.search(
            r"\b(breathing difficulty|difficulty breathing|"
            r"shortness of breath|can't breathe|cannot breathe|"
            r"trouble breathing|struggling to breathe)\b",
            text,
        )
    )

    if chest_pain and breathing_problem:
        return True

    facial_problem = bool(
        re.search(
            r"\b(face|facial)\b.{0,60}\b("
            r"weakness|weak|drooping|droop|numbness|numb"
            r")\b",
            text,
        )
    )

    speech_problem = bool(
        re.search(
            r"\b("
            r"difficulty speaking|trouble speaking|"
            r"difficulty talking|trouble talking|"
            r"slurred speech|unclear speech|garbled speech|"
            r"cannot speak|can't speak|unable to speak|"
            r"difficulty understanding speech"
            r")\b",
            text,
        )
    )

    sudden_onset = bool(
        re.search(
            r"\b(sudden|suddenly|all of a sudden)\b",
            text,
        )
    )

    if facial_problem and speech_problem:
        return True

    if sudden_onset and (facial_problem or speech_problem):
        return True

    one_sided_weakness = bool(
        re.search(
            r"\b(one side|left side|right side)\b.{0,50}\b("
            r"weakness|weak|numbness|numb|drooping"
            r")\b",
            text,
        )
    )

    if one_sided_weakness and speech_problem:
        return True

    return False


# ============================================================
# 4. RESPONSE CLEANING AND VALIDATION
# ============================================================

def clean_medical_response(response):
    """Clean spacing and formatting without changing medical facts."""

    if response is None:
        return ""

    response = str(response)
    response = response.replace("\r\n", "\n").replace("\r", "\n")

    spacing_fixes = {
        "andshould": "and should",
        "andshouldbe": "and should be",
        "atwork": "at work",
        "forindividual": "for individual",
        "medicaladvice": "medical advice",
        "feelingof": "feeling of",
        "professionalfor": "professional for",
        "healthcareprofessional": "healthcare professional",
        "qualifiedhealthcare": "qualified healthcare",
        "symptomsand": "symptoms and",
        "severityand": "severity and",
        "shouldbe": "should be",
        "substancesatwork": "substances at work",
        "risk factorsiggers": "risk factors and triggers",
        "triggers.The": "triggers. The",
        "personaldiagnosis": "personal diagnosis",
        "experiencesthe": "experience the",
    }

    for incorrect, corrected in spacing_fixes.items():
        response = re.sub(
            re.escape(incorrect),
            corrected,
            response,
            flags=re.IGNORECASE,
        )

    response = re.sub(
        r"([.!?])(?=[A-Z])",
        r"\1 ",
        response,
    )

    cleaned_lines = []

    for line in response.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})", line):
            continue

        cleaned_lines.append(line)

    response = "\n".join(cleaned_lines)

    response = re.sub(
        r"\n[ \t]*\n(?:[ \t]*\n)+",
        "\n\n",
        response,
    )

    response = response.strip()
    response = re.sub(r"\s+\*\s*$", "", response).strip()

    return response


def is_obviously_incomplete_response(response):
    """Reject empty responses and obvious unfinished endings."""

    if not response or not str(response).strip():
        return True

    text = str(response).strip()

    if text.endswith(":"):
        return True

    unfinished_endings = [
        r"\band$",
        r"\bor$",
        r"\bbecause$",
        r"\bsuch as$",
        r"\bfor example$",
        r"\bincluding$",
        r"\bfor instance$",
    ]

    for pattern in unfinished_endings:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    last_line = text.splitlines()[-1].strip()

    if re.fullmatch(r"(?:[-*+]|\d+[.)])\s*", last_line):
        return True

    if last_line in ('"', "'", "“", "‘", "*"):
        return True

    return False


def add_medical_disclaimer(response):
    """Append the standard disclaimer no more than once."""

    response = clean_medical_response(response)

    if not response:
        response = "I could not generate a reliable medical response."

    disclaimer_patterns = [
        (
            r"This is general educational information,\s*not a personal "
            r"diagnosis or treatment plan\.\s*Consult a qualified "
            r"healthcare professional for individual medical advice\.?"
        ),
        r"This information is not a substitute for professional medical advice\.?",
        r"This is not medical advice\.?",
        r"Please consult a doctor for medical advice\.?",
    ]

    for pattern in disclaimer_patterns:
        response = re.sub(
            pattern,
            "",
            response,
            flags=re.IGNORECASE,
        )

    response = clean_medical_response(response)

    if response:
        return f"{response}\n\n{MEDICAL_DISCLAIMER}"

    return MEDICAL_DISCLAIMER


# ============================================================
# 5. BASIC CONVERSATION
# ============================================================

BASIC_RESPONSES = {
    "hi": (
        "Hello! I'm your Medical AI Assistant. You can ask me "
        "general questions about health, medical conditions, "
        "symptoms, and prevention."
    ),
    "hello": (
        "Hello! I'm your Medical AI Assistant. How can I help "
        "you with a medical or health-related question?"
    ),
    "hey": (
        "Hi! You can ask me general questions about health "
        "and medical conditions."
    ),
    "thanks": (
        "You're welcome! I'm here to help with general medical "
        "and health information."
    ),
    "thank you": (
        "You're welcome! A qualified healthcare professional "
        "can advise you about your individual situation."
    ),
    "bye": "Goodbye! Take care of your health.",
    "goodbye": "Goodbye! Take care of your health.",
}


def get_basic_response(user_message):
    """Return a response for greetings and simple conversation."""

    cleaned = re.sub(
        r"[^\w\s]",
        "",
        user_message.lower(),
    ).strip()

    return BASIC_RESPONSES.get(cleaned)


# ============================================================
# 6. CHATBOT META-QUESTIONS
# ============================================================

CHATBOT_META_PATTERNS = [
    r"\bconfirm\b.{0,60}\b(assistant|chatbot|bot|condition|clarif)",
    r"\b(check|verify|test)\b.{0,60}\b(assistant|chatbot|bot|response)",
    r"\bwhy did you ask\b",
    r"\bwhy are you asking\b",
    r"\bwhy do you need\b.{0,50}\b(condition|topic|clarification)\b",
    r"\bhow does\b.{0,40}\b(assistant|chatbot|bot)\b.{0,30}\bwork\b",
    r"\bwhat can you do\b",
    r"\bwhat are you designed to do\b",
    r"\bwhat kind of questions can i ask\b",
    r"\bhow do you handle follow[- ]up questions\b",
    r"\bdo you remember\b.{0,40}\b(previous|last)\b.{0,30}\b(question|message|topic)\b",
]


def is_chatbot_meta_message(user_message):
    """Identify questions about the assistant itself."""

    if not user_message:
        return False

    return any(
        re.search(
            pattern,
            user_message.lower().strip(),
            re.IGNORECASE,
        )
        for pattern in CHATBOT_META_PATTERNS
    )


def generate_chatbot_meta_response():
    return (
        "I'm designed to answer general medical and health-related "
        "questions. I can use recent conversation history to understand "
        "follow-up questions. If you ask something like "
        "'How is it treated?' without an identifiable medical "
        "condition in the current conversation, I'll ask you to "
        "clarify the condition rather than guess. You can also start "
        "a new chat to discuss a different topic."
    )


# ============================================================
# 7. UNRELATED-TOPIC FILTER
# ============================================================

OFF_TOPIC_PATTERNS = [
    r"\b(tell|write|make|create)\b.{0,30}\b(joke|funny story)\b",
    r"\b(write|compose|generate)\b.{0,30}\b(poem|song|lyrics)\b",
    r"\b(write|create)\b.{0,30}\b(essay|fiction|short story)\b",
    r"\b(quantum physics|quantum mechanics)\b",
    r"\b(general relativity|special relativity)\b",
    r"\b(black holes?|planetary motion)\b",
    r"\b(weather forecast|weather today|temperature outside)\b",
    r"\b(football|soccer|cricket match|basketball score)\b",
    r"\b(write|debug|fix|explain)\b.{0,30}\bpython code\b",
    r"\b(solve|calculate)\b.{0,30}\b(equation|integral|derivative)\b",
]


def is_clearly_off_topic(user_message):
    """Detect common requests unrelated to medical information."""

    if not user_message:
        return False

    return any(
        re.search(
            pattern,
            user_message.lower().strip(),
            re.IGNORECASE,
        )
        for pattern in OFF_TOPIC_PATTERNS
    )


def generate_off_topic_response():
    return (
        "I'm designed to help with medical and health-related "
        "information. I can't assist with that unrelated topic, "
        "but you're welcome to ask about health, symptoms, "
        "medical terminology, or prevention."
    )


# ============================================================
# 8. FOLLOW-UP TOPIC RESOLUTION
# ============================================================

def resolve_context_topic(user_message, conversation_history=""):
    """
    Resolve the topic from the current question first.

    If the current message is a follow-up such as
    'What about prevention?', search recent history from newest
    to oldest for a recognizable medical topic.
    """

    try:
        current_topic = detect_medical_topic(user_message)
        if current_topic:
            return current_topic
    except Exception as error:
        print(
            "[Medical Context] Current-topic detection failed: "
            f"{type(error).__name__}: {error}"
        )

    if not conversation_history:
        return None

    # Search individual history lines in reverse order so that
    # recent conversation context is preferred.
    history_lines = [
        line.strip()
        for line in str(conversation_history).splitlines()
        if line.strip()
    ]

    for line in reversed(history_lines):
        # Avoid using a generic assistant disclaimer as context.
        if MEDICAL_DISCLAIMER.lower() in line.lower():
            continue

        try:
            topic = detect_medical_topic(line)
            if topic:
                return topic
        except Exception:
            continue

    return None


# ============================================================
# 9. LOCAL MEDICAL RESPONSE
# ============================================================

def build_local_medical_response(result):
    """Build a response only from a successfully retrieved answer."""

    if not isinstance(result, dict) or result.get("found") is not True:
        return None

    answer = result.get("answer")

    if not answer or not str(answer).strip():
        return None

    answer = clean_medical_response(answer)

    if not answer:
        return None

    source = result.get("source")

    if source and str(source).strip():
        answer += (
            "\n\nSource listed in the local knowledge base: "
            + str(source).strip()
        )

    print("[Medical Response] Using local knowledge base.")

    return add_medical_disclaimer(answer)


# ============================================================
# 10. GEMINI MEDICAL FALLBACK
# ============================================================

def generate_gemini_medical_fallback(
    user_message,
    conversation_history="",
):
    """Use the existing Gemini client and reject incomplete answers."""

    prompt = f"""
{MEDICAL_SYSTEM_INSTRUCTIONS}

Recent conversation:
{conversation_history or "No previous conversation provided."}

Latest user question:
{user_message}

Instructions:
- Answer the latest question.
- Resolve follow-up questions using the conversation when possible.
- Ask for clarification if the medical condition is unclear.
- Do not invent references or claim an unverified source.
- Do not diagnose or prescribe medication.
- For possible emergencies, advise immediate emergency assistance.
- Say "local emergency services" rather than assuming a country.
- Use clear headings and bullet points when helpful.
- Use normal spacing between words.
- Finish the answer with a complete sentence.
- Do not end with a colon or unfinished bullet.
- Do not include separator lines made from repeated hyphens.
- Do not append the application's standard disclaimer.
- If you cannot answer reliably, acknowledge the limitation.
"""

    try:
        print(
            "[Medical Response] Local answer unavailable. "
            "Trying Gemini fallback."
        )

        # Keep the existing Gemini-compatible client, model,
        # endpoint, and chat.completions implementation unchanged.
        response = generate_gpt_response(user_message=prompt)

        if not response or not str(response).strip():
            print("[Medical Response] Gemini returned an empty response.")
            return None

        response = clean_medical_response(response)

        if is_obviously_incomplete_response(response):
            print(
                "[Medical Response] Gemini response appears incomplete. "
                "Rejecting it."
            )
            return None

        print("[Medical Response] Gemini response accepted.")

        return add_medical_disclaimer(response)

    except Exception as error:
        print(
            "[Medical Response] Gemini fallback failed: "
            f"{type(error).__name__}: {error}"
        )
        return None


# ============================================================
# 11. UNKNOWN / UNAVAILABLE RESPONSE
# ============================================================

def generate_unknown_response():
    return (
        "I couldn't find sufficiently relevant information in the "
        "local medical knowledge base, and the AI fallback is "
        "currently unavailable or returned an incomplete answer. "
        "Please try again shortly or consult a qualified healthcare "
        "professional. If this may be an emergency, seek immediate "
        "medical care."
    )


# ============================================================
# 12. MEDICAL RESPONSE PIPELINE
# ============================================================

def generate_medical_response(
    user_message,
    intent=None,
    conversation_history="",
):
    """
    Retrieve local information using the detected intent and topic.

    Use Gemini only when local retrieval cannot provide a reliable
    answer, unless the configuration explicitly enables Gemini for
    all medical answers.
    """

    detected_topic = resolve_context_topic(
        user_message=user_message,
        conversation_history=conversation_history,
    )

    local_result = None

    try:
        # Important: pass both topic and intent to retrieval.
        # This helps distinguish a definition from symptoms,
        # causes, prevention, diagnosis, and management.
        local_result = get_faq_response(
            user_message,
            detected_topic=detected_topic,
            detected_intent=intent,
        )
    except Exception as error:
        print(
            "[Medical Retrieval] Retrieval failed: "
            f"{type(error).__name__}: {error}"
        )

    local_response = build_local_medical_response(local_result)

    if local_response and not USE_GEMINI_FOR_MEDICAL_ANSWERS:
        return local_response

    gemini_response = generate_gemini_medical_fallback(
        user_message=user_message,
        conversation_history=conversation_history,
    )

    if gemini_response:
        return gemini_response

    if local_response:
        print(
            "[Medical Response] Gemini unavailable or incomplete; "
            "using the available local response."
        )
        return local_response

    print("[Medical Response] No reliable answer is available.")

    return add_medical_disclaimer(generate_unknown_response())


# ============================================================
# 13. MAIN RESPONSE FUNCTION
# ============================================================

def generate_response(
    user_message,
    intent=None,
    conversation_history="",
):
    """Main response pipeline with emergency detection first."""

    if user_message is None:
        return "Please enter a message."

    user_message = str(user_message).strip()

    if not user_message:
        return "Please enter a message."

    # Step 1: Emergencies always take priority.
    if is_emergency_message(user_message):
        print("[Medical Safety] Potential emergency detected.")
        return EMERGENCY_RESPONSE

    # Step 2: Basic conversation.
    basic_response = get_basic_response(user_message)

    if basic_response:
        return basic_response

    # Step 3: Questions about the assistant.
    if is_chatbot_meta_message(user_message):
        return generate_chatbot_meta_response()

    # Step 4: Filter clearly unrelated requests.
    if is_clearly_off_topic(user_message):
        return generate_off_topic_response()

    # Step 5: Medical retrieval and optional Gemini fallback.
    return generate_medical_response(
        user_message=user_message,
        intent=intent,
        conversation_history=conversation_history,
    )


# ============================================================
# 14. DIRECT TESTS
# ============================================================

if __name__ == "__main__":

    print("\nSPACING TESTS")

    spacing_samples = [
        "severity and should be determined",
        "substances at work",
        "professionalfor individual medical advice",
        "a feelingof tightness",
        "Symptoms may vary, and not everyone experiencesthe same signs.",
        "triggers.The appropriate treatment depends on severity.",
    ]

    for sample in spacing_samples:
        cleaned = clean_medical_response(sample)

        passed = not any(
            bad in cleaned.lower()
            for bad in [
                "andshould",
                "atwork",
                "professionalfor",
                "feelingof",
                "experiencesthe",
                "triggers.the",
            ]
        )

        print(f"{'PASS' if passed else 'FAIL'}: {cleaned}")

    print("\nDISCLAIMER TEST")

    sample_response = add_medical_disclaimer(
        "This is a test response."
    )

    print(sample_response)
    print(
        "Disclaimer appears once:",
        sample_response.count(MEDICAL_DISCLAIMER) == 1,
    )

    print("\nEMERGENCY TESTS")

    emergency_tests = [
        "I have severe chest pain and difficulty breathing.",
        "A person suddenly has facial weakness and difficulty speaking.",
        "Someone has slurred speech and one side of their face is drooping.",
        "I cannot breathe properly.",
        "I think I am having a heart attack.",
    ]

    for message in emergency_tests:
        detected = is_emergency_message(message)
        print(f"{'PASS' if detected else 'FAIL'}: {message}")

    print("\nLOCAL KNOWLEDGE TESTS")

    test_cases = [
        ("Tell me about asthma", "definition"),
        ("What are the symptoms of asthma?", "symptoms"),
        ("What causes asthma?", "causes_risk_factors"),
        ("How can asthma be prevented?", "prevention"),
        ("What is diabetes?", "definition"),
        ("What are the symptoms of diabetes?", "symptoms"),
    ]

    for question, expected_intent in test_cases:
        print(f"\nQuestion: {question}")
        result = get_faq_response(
            question,
            detected_intent=expected_intent,
        )

        found = bool(result and result.get("found"))
        actual_type = (
            result.get("information_type")
            if isinstance(result, dict)
            else None
        )

        print("Found:", found)
        print("Expected intent:", expected_intent)
        print("Information type:", actual_type)

        if not found:
            print("Result: CHECK REQUIRED")
        else:
            print("Answer:", result.get("answer", ""))

    print("\nTESTS COMPLETED")
