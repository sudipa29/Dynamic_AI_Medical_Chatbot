
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "medical",
    "medical_knowledge.csv",
)

# Used only when a supported topic has no explicit information
# category and a text-similarity fallback is required.
SIMILARITY_THRESHOLD = 0.15


# ============================================================
# 2. SPECIAL HEADACHE KNOWLEDGE
# ============================================================

HEADACHE_DEFINITION = (
    "A headache is pain or discomfort in the head, scalp, or neck. "
    "Its characteristics and associated symptoms vary depending on "
    "the underlying cause. A headache alone cannot establish a diagnosis."
)

HEADACHE_SYMPTOMS = (
    "Headache commonly involves pain or discomfort in the head, scalp, "
    "or neck. The pain may feel dull, aching, pressure-like, or throbbing. "
    "Associated symptoms vary with the cause and may include nausea, "
    "sensitivity to light or sound, or other symptoms. A headache alone "
    "cannot establish its cause."
)

HEADACHE_CAUSES = (
    "Common headache triggers and causes can include stress, inadequate "
    "sleep, dehydration, skipped meals, eye strain, infections, and "
    "changes in caffeine intake. Migraines and tension-type headaches "
    "are also common headache disorders. Some headaches occur with "
    "other medical conditions, so the pattern, severity, duration, and "
    "associated symptoms matter. These possibilities do not establish "
    "the cause of an individual's headache."
)

HEADACHE_CARE = (
    "Seek emergency medical help if a headache begins suddenly and is "
    "extremely severe, is the worst headache you have experienced, or "
    "occurs with weakness, difficulty speaking, confusion, loss of "
    "consciousness, a seizure, fever with a stiff neck, or a recent "
    "significant head injury. Seek medical advice for a new, persistent, "
    "worsening, or recurring headache, or one associated with vision "
    "changes or repeated vomiting. These warning signs do not identify "
    "a specific diagnosis, but they can indicate a need for urgent assessment."
)

HEADACHE_PREVENTION = (
    "Some headache triggers may be reduced by maintaining regular sleep, "
    "drinking enough fluids, eating regular meals, managing stress, "
    "taking breaks from screens, and keeping a record of possible triggers. "
    "Prevention depends on the headache type and the individual's circumstances."
)

GENERAL_EVALUATION_ANSWER = (
    "A medical evaluation may include questions about symptoms, when they "
    "started, their severity, medical history, medicines, and possible "
    "triggers. A healthcare professional may perform a physical or "
    "neurological examination and recommend tests when indicated. Not "
    "everyone needs the same tests; the evaluation depends on the "
    "individual's symptoms and clinical findings."
)


# ============================================================
# 3. TOPIC ALIASES
# ============================================================

# Alias keys are used to recognize topics in the user's message.
# They do not have to match the CSV's capitalization.
# After a topic is recognized, we resolve it to the actual CSV topic.

TOPIC_ALIASES = {
    "diabetes": [
        "diabetes",
        "diabetic",
        "blood sugar",
        "type 1 diabetes",
        "type 2 diabetes",
        "type one diabetes",
        "type two diabetes",
    ],
    "hypertension": [
        "hypertension",
        "high blood pressure",
        "blood pressure",
    ],
    "asthma": [
        "asthma",
        "asthmatic",
    ],
    "migraine": [
        "migraine",
        "migraines",
    ],
    "common cold": [
        "common cold",
        "cold",
    ],
    "influenza": [
        "influenza",
        "flu",
    ],
    "dehydration": [
        "dehydration",
        "dehydrated",
    ],
    "anemia": [
        "anemia",
        "anaemia",
        "low hemoglobin",
        "low haemoglobin",
    ],
    "obesity": [
        "obesity",
        "obese",
    ],
    "high cholesterol": [
        "high cholesterol",
        "cholesterol",
    ],
    "acid reflux / gerd": [
        "acid reflux",
        "gerd",
        "heartburn",
        "reflux",
        "gastroesophageal reflux disease",
        "gastro-oesophageal reflux disease",
    ],
    "gastritis": [
        "gastritis",
    ],
    "food poisoning": [
        "food poisoning",
        "foodborne illness",
    ],
    "allergy": [
        "allergy",
        "allergies",
        "allergic reaction",
    ],
    "arthritis": [
        "arthritis",
        "joint inflammation",
    ],
    "pneumonia": [
        "pneumonia",
    ],
    "urinary tract infection": [
        "urinary tract infection",
        "urinary infection",
        "uti",
        "urine infection",
    ],
    "vitamin d deficiency": [
        "vitamin d deficiency",
        "low vitamin d",
    ],
    "covid-19": [
        "covid-19",
        "covid 19",
        "covid",
        "coronavirus",
    ],
    "heart disease": [
        "heart disease",
        "cardiac disease",
        "heart condition",
        "cardiovascular disease",
    ],
    "thyroid disorders": [
        "thyroid disorder",
        "thyroid disorders",
        "thyroid disease",
        "thyroid problem",
        "thyroid problems",
    ],
    "depression": [
        "depression",
        "depressed",
        "depressive disorder",
    ],
    "anxiety": [
        "anxiety",
        "anxious",
        "anxiety disorder",
    ],
    "insomnia": [
        "insomnia",
        "difficulty sleeping",
        "trouble sleeping",
    ],
    "kidney disease": [
        "kidney disease",
        "chronic kidney disease",
    ],
    "kidney stones": [
        "kidney stone",
        "kidney stones",
    ],
    "fever": [
        "fever",
        "high temperature",
        "raised temperature",
    ],
    "heart attack": [
        "heart attack",
        "heart attacks",
        "myocardial infarction",
    ],
    "headache": [
        "headache",
        "headaches",
    ],
}


# ============================================================
# 4. INFORMATION CATEGORY ALIASES
# ============================================================

# Maps both the user's language and ML intent labels to the
# information_type values used in medical_knowledge.csv.

INTENT_TO_INFORMATION_TYPE = {
    "definition": "Definition",
    "symptoms": "Symptoms",
    "causes_risk_factors": "Causes / Risk Factors",
    "causes / risk factors": "Causes / Risk Factors",
    "prevention": "Prevention",
    "diagnosis_tests": "Diagnosis / Tests",
    "diagnosis / tests": "Diagnosis / Tests",
    "general_management": "General Management",
    "when_to_seek_care": "When to Seek Medical Care",
    "when_to_seek_medical_care": "When to Seek Medical Care",
    "when to seek medical care": "When to Seek Medical Care",
}


# ============================================================
# 5. LOAD AND PREPARE KNOWLEDGE BASE
# ============================================================

def normalize_text(text):
    """Normalize text for phrase matching."""

    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_knowledge_base():
    """Load and validate the local medical knowledge CSV."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Medical knowledge file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "topic",
        "information_type",
        "content",
        "source",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Medical knowledge CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in required_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[
        (df["topic"] != "")
        & (df["information_type"] != "")
        & (df["content"] != "")
    ].copy()

    df = df.reset_index(drop=True)

    df["topic_key"] = df["topic"].map(normalize_text)
    df["information_type_key"] = (
        df["information_type"].map(normalize_text)
    )

    df["search_text"] = (
        df["topic"].astype(str)
        + " "
        + df["information_type"].astype(str)
        + " "
        + df["content"].astype(str)
    )

    return df


knowledge_df = load_knowledge_base()

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

knowledge_matrix = vectorizer.fit_transform(
    knowledge_df["search_text"]
)


# ============================================================
# 6. PHRASE MATCHING AND TOPIC RESOLUTION
# ============================================================

def contains_phrase(text, phrase):
    """Check a complete phrase without matching inside larger words."""

    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)

    if not normalized_phrase:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(normalized_phrase).replace(r"\ ", r"\s+")
        + r"(?!\w)"
    )

    return re.search(pattern, normalized_text) is not None


def resolve_csv_topic(topic_name):
    """
    Resolve a topic or alias to the actual topic name in the CSV.

    This handles differences such as 'UTI' versus
    'Urinary Tract Infection (UTI)'.
    """

    if not topic_name:
        return None

    normalized_topic = normalize_text(topic_name)

    # First try an exact normalized topic match.
    exact_matches = knowledge_df[
        knowledge_df["topic_key"] == normalized_topic
    ]

    if not exact_matches.empty:
        return exact_matches.iloc[0]["topic"]

    # Next try known aliases against actual CSV topic names.
    for alias_group, aliases in TOPIC_ALIASES.items():
        group_names = [alias_group] + aliases

        if normalized_topic not in {
            normalize_text(name) for name in group_names
        }:
            continue

        for alias in group_names:
            matching_rows = knowledge_df[
                knowledge_df["topic_key"] == normalize_text(alias)
            ]

            if not matching_rows.empty:
                return matching_rows.iloc[0]["topic"]

        # Some CSV names include extra wording, for example
        # "Urinary Tract Infection (UTI)".
        for csv_topic in knowledge_df["topic"].drop_duplicates():
            if any(
                contains_phrase(csv_topic, alias)
                for alias in group_names
            ):
                return csv_topic

    return None


def detect_medical_topic(user_message):
    """Detect a supported medical topic explicitly mentioned by the user."""

    message = normalize_text(user_message)

    if not message:
        return None

    candidates = []

    for alias_group, aliases in TOPIC_ALIASES.items():
        for alias in set([alias_group] + aliases):
            if contains_phrase(message, alias):
                candidates.append(
                    (len(normalize_text(alias)), alias_group, alias)
                )

    # Prefer the longest matching phrase. This helps distinguish
    # "type 2 diabetes" from the broader "diabetes" alias.

    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, alias_group, _ in candidates:
        csv_topic = resolve_csv_topic(alias_group)

        if csv_topic:
            return csv_topic

    return None


# ============================================================
# 7. INFORMATION CATEGORY DETECTION
# ============================================================

def detect_requested_information_type(user_message, detected_intent=None):
    """
    Identify the information category requested by the user.

    Explicit wording in the question takes priority. If no category
    can be inferred from the wording, use the ML intent if provided.
    """

    message = normalize_text(user_message)

    # 1. Emergency and care-seeking questions
    care_phrases = [
        "when should i see a doctor",
        "when should someone seek medical care",
        "when should someone seek care",
        "when should i seek medical care",
        "when to seek medical care",
        "when should i seek medical help",
        "when to seek medical help",
        "when should i get medical help",
        "when to get medical help",
        "seek medical attention",
        "seek medical care",
        "warning signs",
        "emergency symptoms",
        "when to see a doctor",
        "when to contact a doctor",
        "when to get medical attention",
        "should i go to the hospital",
    ]

    if any(contains_phrase(message, phrase) for phrase in care_phrases):
        return "When to Seek Medical Care"

    # 2. Causes and risk factors
    cause_phrases = [
        "causes",
        "cause of",
        "caused by",
        "risk factors",
        "what causes",
        "why does",
        "why do i",
        "triggers",
        "trigger",
    ]

    if any(contains_phrase(message, phrase) for phrase in cause_phrases):
        return "Causes / Risk Factors"

    # 3. Symptoms
    symptom_phrases = [
        "symptom",
        "symptoms",
        "signs of",
        "what does it feel like",
        "what are its symptoms",
        "what are the symptoms",
    ]

    if any(contains_phrase(message, phrase) for phrase in symptom_phrases):
        return "Symptoms"

    # 4. Diagnosis and tests
    diagnosis_phrases = [
        "diagnosis",
        "diagnosed",
        "diagnose",
        "tests",
        "test for",
        "medical evaluation",
        "how is it detected",
        "how is it diagnosed",
        "which examination",
    ]

    if any(contains_phrase(message, phrase) for phrase in diagnosis_phrases):
        return "Diagnosis / Tests"

    # 5. General management
    management_phrases = [
        "managed",
        "manage",
        "management",
        "treatment",
        "treat",
        "treated",
        "treating",
        "relieve",
        "relief",
        "how can i control",
        "how to control",
        "how to manage",
        "how is it treated",
        "what can i do",
        "help with",
    ]

    if any(contains_phrase(message, phrase) for phrase in management_phrases):
        return "General Management"

    # 6. Prevention
    prevention_phrases = [
        "prevent",
        "prevention",
        "preventing",
        "avoid",
        "reduce the risk",
    ]

    if any(contains_phrase(message, phrase) for phrase in prevention_phrases):
        return "Prevention"

    # 7. Definition and overview
    definition_phrases = [
        "what is",
        "what are",
        "define",
        "definition",
        "meaning of",
        "explain",
        "tell me about",
        "give me an overview",
        "describe",
        "information about",
        "information on",
    ]

    if any(contains_phrase(message, phrase) for phrase in definition_phrases):
        return "Definition"

    # 8. Fall back to the ML intent when the question wording
    # doesn't independently identify the requested category.
    if detected_intent:
        normalized_intent = normalize_text(detected_intent)

        if normalized_intent in INTENT_TO_INFORMATION_TYPE:
            return INTENT_TO_INFORMATION_TYPE[normalized_intent]

    return None


# ============================================================
# 8. SPECIAL MEDICAL RESPONSES
# ============================================================

def get_special_medical_response(
    user_message,
    detected_topic=None,
    detected_intent=None,
):
    """Handle headache questions and general evaluation questions."""

    message = normalize_text(user_message)

    # Do not let headache-specific handling override another
    # supported medical topic.

    if detected_topic and normalize_text(detected_topic) != "headache":
        return None

    is_headache_question = (
        normalize_text(detected_topic) == "headache"
        or bool(re.search(r"\bheadaches?\b", message))
    )

    if is_headache_question:
        requested_type = detect_requested_information_type(
            user_message,
            detected_intent,
        )

        # A question can ask about both causes and emergency warning signs.
        asks_causes = bool(
            re.search(
                r"\b(cause|causes|caused|trigger|triggers|risk factors)\b",
                message,
            )
        )

        asks_care = bool(
            re.search(
                r"\b("
                r"when should|when do i|when to|"
                r"seek medical care|seek medical help|"
                r"see a doctor|medical attention|"
                r"warning signs|emergency|urgent"
                r")\b",
                message,
            )
        )

        if asks_causes and asks_care:
            return {
                "found": True,
                "answer": (
                    "**Common causes and triggers of headaches**\n\n"
                    + HEADACHE_CAUSES
                    + "\n\n**When to seek medical care**\n\n"
                    + HEADACHE_CARE
                ),
                "topic": "Headache",
                "information_type": (
                    "Causes / Risk Factors; When to Seek Medical Care"
                ),
                "source": "MedlinePlus; NHS",
                "similarity": 1.0,
            }

        if requested_type == "Definition":
            answer = HEADACHE_DEFINITION
            information_type = "Definition"

        elif requested_type == "Symptoms":
            answer = HEADACHE_SYMPTOMS
            information_type = "Symptoms"

        elif requested_type == "Causes / Risk Factors":
            answer = HEADACHE_CAUSES
            information_type = "Causes / Risk Factors"

        elif requested_type == "When to Seek Medical Care":
            answer = HEADACHE_CARE
            information_type = "When to Seek Medical Care"

        elif requested_type == "Prevention":
            answer = HEADACHE_PREVENTION
            information_type = "Prevention"

        else:
            # A bare statement like "I have a headache" should
            # not be treated as a definition request.
            answer = HEADACHE_SYMPTOMS
            information_type = "Symptoms"

        return {
            "found": True,
            "answer": answer,
            "topic": "Headache",
            "information_type": information_type,
            "source": "MedlinePlus; NHS",
            "similarity": 1.0,
        }

    evaluation_phrases = [
        "medical evaluation",
        "medical examination",
        "what tests might",
        "tests might be needed",
        "how will a doctor check",
        "how is a condition diagnosed",
    ]

    if any(phrase in message for phrase in evaluation_phrases):
        return {
            "found": True,
            "answer": GENERAL_EVALUATION_ANSWER,
            "topic": "General Medical Evaluation",
            "information_type": "Diagnosis / Tests",
            "source": "General educational information",
            "similarity": 1.0,
        }

    return None


# ============================================================
# 9. NOT-FOUND RESULT
# ============================================================

def not_found_result(similarity=0.0):
    """Return a consistent result when local retrieval is unsuitable."""

    return {
        "found": False,
        "answer": (
            "I couldn't find sufficiently relevant information "
            "in the local medical knowledge base to answer that "
            "question reliably. I'll try the configured AI fallback "
            "when available."
        ),
        "topic": None,
        "information_type": None,
        "source": None,
        "similarity": float(similarity),
    }


# ============================================================
# 10. KNOWLEDGE-BASE RETRIEVAL
# ============================================================

def get_faq_response(
    user_message,
    detected_topic=None,
    detected_intent=None,
):
    """
    Retrieve an answer from the local medical knowledge base.

    Parameters:
        user_message:
            The user's question, including contextual expansion
            when applicable.

        detected_topic:
            Optional topic already identified by the chatbot pipeline.

        detected_intent:
            Optional ML intent, such as 'definition' or 'symptoms'.

    Returns:
        found, answer, topic, information_type, source, similarity

    Compatibility:
        Existing calls to get_faq_response(user_message) continue
        to work. Callers can optionally pass detected_topic and
        detected_intent as keyword arguments.
    """

    message = str(user_message or "").strip()

    if not message:
        return {
            "found": False,
            "answer": "Please enter a medical question.",
            "topic": None,
            "information_type": None,
            "source": None,
            "similarity": 0.0,
        }

    # --------------------------------------------------------
    # Step 1: Resolve the topic to an actual CSV topic.
    # --------------------------------------------------------

    resolved_topic = resolve_csv_topic(detected_topic)

    if resolved_topic is None:
        resolved_topic = detect_medical_topic(message)

    # Headache has its own special knowledge even when it is
    # not included in medical_knowledge.csv.

    if (
        resolved_topic is None
        and re.search(r"\bheadaches?\b", normalize_text(message))
    ):
        resolved_topic = "Headache"

    # --------------------------------------------------------
    # Step 2: Detect the requested information category.
    # --------------------------------------------------------

    requested_type = detect_requested_information_type(
        message,
        detected_intent,
    )

    # --------------------------------------------------------
    # Step 3: Try the special response handlers.
    # --------------------------------------------------------

    special_response = get_special_medical_response(
        message,
        detected_topic=resolved_topic,
        detected_intent=detected_intent,
    )

    if special_response is not None:
        return special_response

    normalized_message = normalize_text(message)

    if not normalized_message:
        return not_found_result()

    # --------------------------------------------------------
    # Step 4: Reject unsupported or unrecognized topics.
    # --------------------------------------------------------

    if not resolved_topic:
        query_vector = vectorizer.transform([normalized_message])

        scores = cosine_similarity(
            query_vector,
            knowledge_matrix,
        ).flatten()

        best_score = float(scores.max()) if len(scores) else 0.0

        print(
            "[Medical Retrieval] No supported topic detected. "
            f"Best similarity={best_score:.3f}. "
            "Rejecting unverified local match."
        )

        return not_found_result(best_score)

    # --------------------------------------------------------
    # Step 5: Restrict retrieval to the recognized topic.
    # --------------------------------------------------------

    topic_key = normalize_text(resolved_topic)

    topic_rows = knowledge_df[
        knowledge_df["topic_key"] == topic_key
    ]

    if topic_rows.empty:
        # Headache was handled above. Other missing topics must
        # not be silently replaced with an unrelated condition.
        return not_found_result()

    # --------------------------------------------------------
    # Step 6: Select the requested category FIRST.
    # --------------------------------------------------------
    #
    # This is the important accuracy fix:
    # if the user asks for symptoms, select the Symptoms row.
    # Do not allow TF-IDF similarity to replace it with Definition
    # or another category just because its text scores higher.
    # --------------------------------------------------------

    if requested_type:
        matching_rows = topic_rows[
            topic_rows["information_type_key"]
            == normalize_text(requested_type)
        ]

        if not matching_rows.empty:
            row = matching_rows.iloc[0]

            query_vector = vectorizer.transform([normalized_message])

            row_vector = knowledge_matrix[
                [int(row.name)]
            ]

            similarity = float(
                cosine_similarity(
                    query_vector,
                    row_vector,
                )[0][0]
            )

            return {
                "found": True,
                "answer": row["content"],
                "topic": row["topic"],
                "information_type": row["information_type"],
                "source": row["source"],
                "similarity": similarity,
            }

    # --------------------------------------------------------
    # Step 7: If category is unspecified or absent in the CSV,
    # use similarity only within the recognized topic.
    # --------------------------------------------------------

    query_vector = vectorizer.transform([normalized_message])

    scores = cosine_similarity(
        query_vector,
        knowledge_matrix,
    ).flatten()

    topic_indices = topic_rows.index.to_list()

    topic_scores = scores[topic_indices]

    best_position = int(topic_scores.argmax())
    best_index = topic_indices[best_position]
    best_score = float(topic_scores[best_position])

    if best_score < SIMILARITY_THRESHOLD:
        print(
            "[Medical Retrieval] Recognized topic, but the "
            f"local match is too weak ({best_score:.3f})."
        )

        return not_found_result(best_score)

    best_row = knowledge_df.loc[best_index]

    return {
        "found": True,
        "answer": best_row["content"],
        "topic": best_row["topic"],
        "information_type": best_row["information_type"],
        "source": best_row["source"],
        "similarity": best_score,
    }


# ============================================================
# 11. REGRESSION TESTS
# ============================================================

def run_tests():
    """Run local retrieval tests for topic and category accuracy."""

    test_cases = [
        ("Tell me about asthma", "definition"),
        ("What are the symptoms of asthma?", "symptoms"),
        ("What causes asthma?", "causes_risk_factors"),
        ("How can asthma be prevented?", "prevention"),
        ("How is asthma diagnosed?", "diagnosis_tests"),
        ("How is asthma managed?", "general_management"),
        (
            "When should I seek medical care for asthma?",
            "when_to_seek_care",
        ),
        ("What is diabetes?", "definition"),
        ("What are the symptoms of diabetes?", "symptoms"),
        ("What is hypertension?", "definition"),
        ("What are the symptoms of migraine?", "symptoms"),
        ("What causes headaches?", "causes_risk_factors"),
        (
            "When should someone seek medical care for a headache?",
            "when_to_seek_care",
        ),
        (
            "What tests might be needed during a medical evaluation?",
            "diagnosis_tests",
        ),
        ("What is sepsis?", "definition"),
        ("Explain quantum physics in simple terms.", "definition"),
    ]

    print("=" * 80)
    print("MEDICAL KNOWLEDGE BASE REGRESSION TEST")
    print("=" * 80)

    for question, expected_intent in test_cases:
        result = get_faq_response(
            question,
            detected_intent=expected_intent,
        )

        print("\n" + "-" * 80)
        print("Question:", question)
        print("Expected intent:", expected_intent)
        print("Found:", result["found"])
        print("Topic:", result["topic"])
        print("Information type:", result["information_type"])
        print("Similarity:", round(result["similarity"], 3))
        print("Source:", result["source"])
        print("Answer:", result["answer"])

        # Check that the selected category agrees with the
        # requested intent when a local answer is available.

        expected_type = INTENT_TO_INFORMATION_TYPE.get(
            expected_intent
        )

        if (
            result["found"]
            and expected_type
            and result["information_type"]
            and ";" not in result["information_type"]
        ):
            if (
                normalize_text(result["information_type"])
                == normalize_text(expected_type)
            ):
                print("CATEGORY CHECK: PASS")
            else:
                print(
                    "CATEGORY CHECK: REVIEW - "
                    "returned category differs from expected category."
                )

        if "sepsis" in question.lower() and result["found"]:
            print(
                "TOPIC CHECK: REVIEW - verify that the CSV contains "
                "a sepsis record before accepting this answer."
            )

    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
