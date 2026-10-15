
"""
Medical AI Chatbot - Intent Detection

Training data:
    data/medical/medical_intents.csv

Knowledge source:
    data/medical/medical_knowledge.csv

Supports:
    predict_intent(text)
    predict_intent_details(text)

Returns:
    (intent, confidence, second_intent, second_confidence, margin)
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INTENTS_DATA_PATH = (
    BASE_DIR / "data" / "medical" / "medical_intents.csv"
)

MEDICAL_DATA_PATH = (
    BASE_DIR / "data" / "medical" / "medical_knowledge.csv"
)

EXPECTED_INTENTS = [
    "definition",
    "symptoms",
    "causes_risk_factors",
    "prevention",
    "diagnosis_tests",
    "general_management",
    "when_to_seek_care",
]

MIN_CONFIDENCE = 0.25
MIN_CONFIDENCE_MARGIN = 0.05

vectorizer = None
intent_model = None
training_error = None


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text while preserving medically meaningful terms."""

    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. BASIC CONVERSATION INTENTS
# ============================================================

BASIC_PHRASES = {
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    ],
    "thank_you": [
        "thank you",
        "thanks",
        "thank you so much",
        "thanks a lot",
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later",
    ],
}


def detect_basic_intent(text: str) -> Optional[str]:
    """Recognize exact greetings, thanks, and goodbyes."""

    cleaned = normalize_text(text)

    if not cleaned:
        return "unknown"

    for intent, phrases in BASIC_PHRASES.items():
        for phrase in phrases:
            if cleaned == phrase:
                return intent

    return None


# ============================================================
# 4. MEDICAL TOPICS AND ALIASES
# ============================================================

TOPIC_ALIASES = {
    "acid reflux": [
        "gerd",
        "gastroesophageal reflux disease",
        "gastro-oesophageal reflux disease",
    ],
    "allergy": [
        "allergies",
        "allergic reaction",
    ],
    "anemia": [
        "anaemia",
        "low hemoglobin",
        "low haemoglobin",
    ],
    "anxiety": ["anxiety disorder"],
    "arthritis": ["joint inflammation"],
    "asthma": ["asthmatic condition"],
    "covid-19": [
        "covid",
        "covid 19",
        "coronavirus",
        "corona virus",
    ],
    "common cold": [
        "cold",
        "runny nose and cold",
    ],
    "dehydration": [
        "dehydrated",
        "lack of body fluids",
    ],
    "depression": ["depressive disorder"],
    "diabetes": [
        "diabetes mellitus",
        "high blood sugar",
        "type 1 diabetes",
        "type 2 diabetes",
        "type one diabetes",
        "type two diabetes",
    ],
    "food poisoning": [
        "foodborne illness",
        "food-borne illness",
    ],
    "gastritis": ["stomach inflammation"],
    "heart disease": ["cardiovascular disease"],
    "heart attack": ["myocardial infarction"],
    "high cholesterol": ["hypercholesterolemia"],
    "hypertension": [
        "high blood pressure",
        "elevated blood pressure",
    ],
    "influenza": ["flu", "seasonal flu"],
    "migraine": ["migraine headache"],
    "obesity": ["overweight"],
    "pneumonia": ["lung infection"],
    "thyroid disorders": [
        "thyroid disease",
        "thyroid problem",
        "thyroid problems",
    ],
    "uti": [
        "urinary tract infection",
        "urine infection",
    ],
    "vitamin d deficiency": ["low vitamin d"],
}

MEDICAL_CUES = {
    "disease", "condition", "symptom", "symptoms",
    "medical", "health", "treatment", "managed",
    "management", "diagnosis", "diagnosed",
    "prevention", "prevented", "cause", "causes",
    "risk", "risks", "doctor", "physician",
    "medicine", "medication", "pain", "fever",
    "cough", "headache", "breathing", "infection",
    "illness", "sick", "suffering", "blood pressure",
    "lifestyle", "diagnose", "prevent", "manage",
    "treat", "treatment", "tests", "test",
    "complications", "warning signs",
}


def load_supported_topics() -> list:
    """Load topic names from the medical knowledge base."""

    topics = set(TOPIC_ALIASES.keys())

    try:
        if MEDICAL_DATA_PATH.exists():
            df = pd.read_csv(MEDICAL_DATA_PATH)

            if "topic" in df.columns:
                topics.update(
                    str(topic).strip().lower()
                    for topic in df["topic"].dropna()
                    if str(topic).strip()
                )

    except Exception as error:
        print(
            "[Intent Detection] Could not load medical topics: "
            f"{error}"
        )

    return sorted(topics, key=len, reverse=True)


SUPPORTED_TOPICS = load_supported_topics()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Check for a phrase without matching inside another word."""

    pattern = (
        r"(?<!\w)"
        + re.escape(phrase.lower()).replace(r"\ ", r"\s+")
        + r"(?!\w)"
    )

    return bool(re.search(pattern, text))


def detect_medical_topic(text: str) -> Optional[str]:
    """Identify a supported medical topic or its known alias."""

    cleaned = normalize_text(text)

    if not cleaned:
        return None

    # Match the longest supported topic first.
    for topic in SUPPORTED_TOPICS:
        if _contains_phrase(cleaned, topic):
            return topic.title()

    # Check alternate names.
    for topic, aliases in TOPIC_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if _contains_phrase(cleaned, alias):
                return topic.title()

    return None


def has_medical_context(text: str) -> bool:
    """Determine whether a message appears medically relevant."""

    cleaned = normalize_text(text)

    if not cleaned:
        return False

    if detect_medical_topic(cleaned):
        return True

    return any(
        _contains_phrase(cleaned, cue)
        for cue in MEDICAL_CUES
    )


# ============================================================
# 5. EXPLICIT MEDICAL INTENT RULES
# ============================================================

INTENT_PATTERNS = {
    "when_to_seek_care": [
        r"\bwhen should i see a doctor\b",
        r"\bwhen should i seek medical care\b",
        r"\bwhen should someone seek medical care\b",
        r"\bwhen to seek medical care\b",
        r"\bwhen should i get medical help\b",
        r"\bwhen is it an emergency\b",
        r"\bwhen should i go to hospital\b",
        r"\bwhen should i go to the hospital\b",
        r"\bwhat are the warning signs\b",
        r"\bwhen should i worry\b",
        r"\bwhen is medical attention needed\b",
        r"\bwhen should .* seek medical care\b",
        r"\bwhen to get medical attention\b",
        r"\bwhen do i need medical help\b",
    ],

    "diagnosis_tests": [
        r"\bhow is .{1,80}\bdiagnosed\b",
        r"\bhow do doctors diagnose\b",
        r"\bhow can .{1,80}\bbe diagnosed\b",
        r"\bwhat tests are used\b",
        r"\bwhat tests diagnose\b",
        r"\bwhich tests are needed\b",
        r"\bdiagnosis of\b",
        r"\btests for\b",
        r"\bhow do doctors test for\b",
        r"\bwhat tests are needed\b",
        r"\bwhat test is used\b",
        r"\bhow can doctors confirm\b",
        r"\bhow is .{1,80}\bidentified\b",
    ],

    "causes_risk_factors": [
        r"\bwhat causes\b",
        r"\bwhat are the causes\b",
        r"\bwhat causes it\b",
        r"\bwhat causes this\b",
        r"\bwhat are its causes\b",
        r"\bwhy does .{1,100}\bhappen\b",
        r"\bwhy does .{1,100}\bbecome\b",
        r"\bwhy does .{1,100}\bget\b",
        r"\bwhy does .{1,100}\brise\b",
        r"\bwhy does .{1,100}\bincrease\b",
        r"\bwhy do people get\b",
        r"\bwhat are the risk factors\b",
        r"\brisk factors for\b",
        r"\bwhat increases the risk\b",
        r"\bwhy do i get\b",
        r"\bwhat triggers\b",
        r"\bwhat are the triggers\b",
        r"\bwhat leads to\b",
        r"\bwhat makes .{1,80}\bhigh\b",
        r"\bwhat makes .{1,80}\blow\b",
        r"\bwhat is the reason for\b",
        r"\bwhat might cause\b",
        r"\bwhat can cause\b",
        r"\bwhat are common causes\b",
        r"\bwhat are the main causes\b",
        r"\bwhy is .{1,80}\bhigh\b",
        r"\bwhy is .{1,80}\belevated\b",
        r"\bwhat factors contribute to\b",
    ],

    "general_management": [
        r"\bhow can it be managed\b",
        r"\bhow can this be managed\b",
        r"\bhow can .{1,80}\bbe managed\b",
        r"\bhow is it managed\b",
        r"\bhow is this managed\b",
        r"\bhow is .{1,80}\bmanaged\b",
        r"\bhow can i manage it\b",
        r"\bhow do i manage it\b",
        r"\bhow to manage it\b",
        r"\bhow can .{1,80}\bmanage\b",
        r"\bhow do .{1,80}\bmanage\b",
        r"\bmanagement of\b",
        r"\bgeneral management of\b",
        r"\bhow to treat\b",
        r"\bhow is it treated\b",
        r"\bhow can it be treated\b",
        r"\bhow do you treat it\b",
        r"\bwhat is its treatment\b",
        r"\bwhat is the treatment\b",
        r"\bwhat are the treatment options\b",
        r"\bwhat about treatment\b",
        r"\btreatment for\b",
        r"\bhow can i relieve\b",
        r"\bways to relieve\b",
        r"\bwhat helps with\b",
        r"\bwhat can help with\b",
        r"\bhow to deal with\b",
        r"\bcare for .{1,80}\b",
        r"\bwhat lifestyle changes\b",
        r"\bwhich lifestyle changes\b",
        r"\blifestyle changes .*manage\b",
        r"\blifestyle changes .*help\b",
        r"\bwhat helps manage\b",
        r"\bwhat can help manage\b",
        r"\bwhat may help manage\b",
        r"\bwhat might help manage\b",
        r"\bhow can i control\b",
        r"\bhow to control\b",
        r"\bways to manage\b",
        r"\bways to control\b",
        r"\bhow can .* improve\b",
        r"\bhow can .* relieve\b",
        r"\bwhat should i do to manage\b",
        r"\bwhat can i do to manage\b",
        r"\bhow can .* symptoms be relieved\b",
        r"\bhow can .* symptoms be managed\b",
        r"\bwhat are the management options\b",
    ],

    "prevention": [
        r"\bhow can it be prevented\b",
        r"\bhow can .{1,80}\bbe prevented\b",
        r"\bhow to prevent\b",
        r"\bhow can i prevent\b",
        r"\bways to prevent\b",
        r"\bhow do i avoid\b",
        r"\bhow can .{1,80}\bbe avoided\b",
        r"\bhow to avoid\b",
        r"\bprevention of\b",
        r"\bhow can i reduce the risk\b",
        r"\bhow to lower the risk\b",
        r"\bwhat about prevention\b",
        r"\bhow about prevention\b",
        r"\bwhat about preventing\b",
        r"\bhow can i protect myself\b",
        r"\bwhat can i do to prevent\b",
        r"\bhow do i prevent\b",
        r"\bhow can .* reduce the risk\b",
        r"\bwhat measures prevent\b",
        r"\bways to reduce the risk\b",
        r"\bcan .* be prevented\b",
        r"\bhow can someone prevent\b",
    ],

    "symptoms": [
        r"\bhow can i tell if .{1,80}\bcausing symptoms\b",
        r"\bhow can i tell whether .{1,80}\bcausing symptoms\b",
        r"\bhow do i tell if .{1,80}\bcausing symptoms\b",
        r"\bwhat are its symptoms\b",
        r"\bwhat are the symptoms\b",
        r"\bwhat symptoms does .{1,80}\bcause\b",
        r"\bwhat are the signs and symptoms\b",
        r"\bwhat symptoms can it cause\b",
        r"\bsymptoms of\b",
        r"\bsigns of\b",
        r"\bhow does it feel\b",
        r"\bwhat does .{1,80}\bfeel like\b",
        r"\bi am suffering from\b",
        r"\bi have symptoms of\b",
        r"\bi am experiencing\b",
        r"\bi feel\b",
        r"\bi have\b.{1,80}\bpain\b",
        r"\bwhat are common symptoms\b",
        r"\bwhat signs should i look for\b",
        r"\bhow can i recognize\b",
        r"\bhow do i know if i have\b",
    ],

    "definition": [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bdefine\b",
        r"\bdefinition of\b",
        r"\bmeaning of\b",
        r"\bexplain\b",
        r"\btell me about\b",
        r"\bwhat does .{1,80}\bmean\b",
    ],
}


def detect_explicit_medical_intent(
    text: str,
) -> Optional[str]:
    """Detect intent from clear question patterns."""

    cleaned = normalize_text(text)

    if not cleaned or not has_medical_context(cleaned):
        return None

    # More specific questions must be checked before general ones.
    intent_priority = [
        "when_to_seek_care",
        "diagnosis_tests",
        "causes_risk_factors",
        "general_management",
        "prevention",
        "symptoms",
        "definition",
    ]

    for intent in intent_priority:
        for pattern in INTENT_PATTERNS.get(intent, []):
            if re.search(pattern, cleaned):
                return intent

    return None


# ============================================================
# 6. TRAIN THE MODEL FROM MEDICAL INTENT EXAMPLES
# ============================================================

def train_intent_model() -> None:
    """Train TF-IDF and Logistic Regression from question examples."""

    global vectorizer, intent_model, training_error

    vectorizer = None
    intent_model = None
    training_error = None

    try:
        if not INTENTS_DATA_PATH.exists():
            training_error = (
                "Intent dataset not found: "
                f"{INTENTS_DATA_PATH}"
            )
            print(f"[Intent Detection] {training_error}")
            return

        df = pd.read_csv(INTENTS_DATA_PATH)

        required_columns = {"text", "intent"}

        if not required_columns.issubset(df.columns):
            training_error = (
                "medical_intents.csv must contain "
                "'text' and 'intent' columns."
            )
            print(f"[Intent Detection] {training_error}")
            return

        df = df[["text", "intent"]].dropna().copy()

        df["text"] = df["text"].map(normalize_text)
        df["intent"] = (
            df["intent"].astype(str).str.strip().str.lower()
        )

        df = df[df["text"] != ""]
        df = df[df["intent"].isin(EXPECTED_INTENTS)]

        df = df.drop_duplicates(
            subset=["text", "intent"]
        ).reset_index(drop=True)

        if df.empty:
            training_error = (
                "No valid question examples were found."
            )
            print(f"[Intent Detection] {training_error}")
            return

        missing_intents = (
            set(EXPECTED_INTENTS) - set(df["intent"].unique())
        )

        if missing_intents:
            training_error = (
                "Training dataset is missing these intents: "
                + ", ".join(sorted(missing_intents))
            )
            print(f"[Intent Detection] {training_error}")
            return

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

        X = vectorizer.fit_transform(df["text"])
        y = df["intent"]

        intent_model = LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )

        intent_model.fit(X, y)

        print(
            "[Intent Detection] Model trained from "
            "medical_intents.csv: "
            f"{len(df)} unique examples, "
            f"{y.nunique()} intents."
        )

    except Exception as error:
        training_error = str(error)
        vectorizer = None
        intent_model = None

        print(
            "[Intent Detection] Model training failed: "
            f"{error}"
        )


# Train once when this module is imported.
train_intent_model()


# ============================================================
# 7. MODEL PROBABILITIES
# ============================================================

def get_model_probabilities(text: str) -> dict:
    """Return model probabilities for every trained intent."""

    if vectorizer is None or intent_model is None:
        return {}

    cleaned = normalize_text(text)

    if not cleaned:
        return {}

    try:
        X = vectorizer.transform([cleaned])
        probabilities = intent_model.predict_proba(X)[0]

        return {
            str(intent): float(probability)
            for intent, probability in zip(
                intent_model.classes_,
                probabilities,
            )
        }

    except Exception as error:
        print(
            "[Intent Detection] Probability calculation failed: "
            f"{error}"
        )
        return {}


def build_prediction(
    selected_intent: str,
    probabilities: dict,
) -> Tuple[str, float, str, float, float]:
    """
    Build a result for a selected intent.

    Confidence is the model's probability for that intent.
    Explicit rules can select the intent even when its probability
    is low. The returned score must not be interpreted as certainty.
    """

    if not probabilities:
        return (
            selected_intent,
            0.0,
            "unknown",
            0.0,
            0.0,
        )

    confidence = float(
        probabilities.get(selected_intent, 0.0)
    )

    alternatives = [
        (intent, probability)
        for intent, probability in probabilities.items()
        if intent != selected_intent
    ]

    alternatives.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    if alternatives:
        second_intent, second_confidence = alternatives[0]
    else:
        second_intent = "unknown"
        second_confidence = 0.0

    margin = confidence - second_confidence

    return (
        selected_intent,
        confidence,
        second_intent,
        float(second_confidence),
        float(margin),
    )


# ============================================================
# 8. INTENT PREDICTION
# ============================================================

def predict_intent_details(
    text: str,
) -> Tuple[str, float, str, float, float]:
    """
    Return:
        (intent, confidence, second_intent,
         second_confidence, confidence_margin)

    Confidence values are fractions between 0 and 1.
    Multiply by 100 when displaying percentages.
    """

    cleaned = normalize_text(text)

    if not cleaned:
        return (
            "unknown",
            0.0,
            "unknown",
            0.0,
            0.0,
        )

    # Basic conversation intents.
    basic_intent = detect_basic_intent(cleaned)

    if basic_intent:
        return (
            basic_intent,
            1.0,
            "unknown",
            0.0,
            1.0,
        )

    # Reject clearly unsupported non-medical questions.
    if not has_medical_context(cleaned):
        return (
            "unknown",
            0.0,
            "unknown",
            0.0,
            0.0,
        )

    # Calculate model probabilities.
    probabilities = get_model_probabilities(cleaned)

    # Clear question patterns take priority over uncertain ML guesses.
    explicit_intent = detect_explicit_medical_intent(cleaned)

    if explicit_intent:
        return build_prediction(
            explicit_intent,
            probabilities,
        )

    if not probabilities:
        return (
            "unknown",
            0.0,
            "unknown",
            0.0,
            0.0,
        )

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_intent, best_confidence = ranked[0]

    if len(ranked) > 1:
        second_intent, second_confidence = ranked[1]
    else:
        second_intent = "unknown"
        second_confidence = 0.0

    margin = best_confidence - second_confidence

    # Keep rejection for uncertain questions that