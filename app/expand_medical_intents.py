import csv
import shutil
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = Path("data/medical/medical_intents.csv")
TOPICS_PATH = Path("data/medical/medical_topics.csv")
BACKUP_PATH = Path("data/medical/medical_intents_before_topic_fix.csv")


# ============================================================
# QUESTION PATTERNS FOR THE SEVEN MEDICAL INTENTS
# ============================================================

INTENT_PATTERNS = {
    "definition": [
        "What is {topic}?",
        "Can you explain {topic} in simple terms?",
        "I want to understand what {topic} means.",
        "Give me an overview of {topic}.",
        "Could you describe {topic}?",
        "What does the term {topic} mean?",
        "Help me understand {topic}.",
        "Please explain the meaning of {topic}.",
        "Can you tell me about {topic}?",
        "I need basic information about {topic}.",
        "How would you describe {topic}?",
        "What should I know about {topic}?",
        "Explain {topic} for a beginner.",
        "Can I get an introduction to {topic}?",
        "Tell me the basics of {topic}.",
        "What exactly is meant by {topic}?",
        "I would like an explanation of {topic}.",
        "Describe {topic} in simple language.",
        "Could I learn the basics of {topic}?",
        "What is meant by the condition called {topic}?",
    ],

    "symptoms": [
        "What are the symptoms of {topic}?",
        "How can someone recognize {topic}?",
        "What signs may indicate {topic}?",
        "How does {topic} usually feel?",
        "What symptoms can occur with {topic}?",
        "What might a person experience with {topic}?",
        "How can {topic} show up in the body?",
        "What are common warning signs of {topic}?",
        "What changes might indicate {topic}?",
        "Can you list symptoms associated with {topic}?",
        "What should I look out for with {topic}?",
        "How might someone know they have symptoms of {topic}?",
        "What physical signs can occur with {topic}?",
        "What are the usual signs of {topic}?",
        "Could you describe the symptoms linked to {topic}?",
        "What symptoms are often reported with {topic}?",
        "How might {topic} affect how someone feels?",
        "What signs are commonly associated with {topic}?",
        "What symptoms can be related to {topic}?",
        "Tell me about the signs of {topic}.",
    ],

    "causes_risk_factors": [
        "What causes {topic}?",
        "What are the risk factors for {topic}?",
        "Why might someone develop {topic}?",
        "What can lead to {topic}?",
        "How does {topic} develop?",
        "What factors increase the chance of {topic}?",
        "What may contribute to {topic}?",
        "What are the possible causes of {topic}?",
        "Why do people get {topic}?",
        "What makes someone more likely to develop {topic}?",
        "Can lifestyle factors contribute to {topic}?",
        "What underlying factors are associated with {topic}?",
        "What may increase someone's risk of {topic}?",
        "How can a person become affected by {topic}?",
        "What is linked to the development of {topic}?",
        "Which factors are associated with {topic}?",
        "What could make {topic} more likely?",
        "What are common contributors to {topic}?",
        "Can you explain why {topic} occurs?",
        "What should I know about the causes of {topic}?",
    ],

    "prevention": [
        "How can {topic} be prevented?",
        "What can help reduce the risk of {topic}?",
        "How can someone lower their chances of developing {topic}?",
        "What steps may help prevent {topic}?",
        "How can I protect myself from {topic}?",
        "Are there ways to reduce the chance of {topic}?",
        "What habits may help prevent {topic}?",
        "How can people reduce their risk of {topic}?",
        "What preventive measures are useful for {topic}?",
        "Can lifestyle changes help prevent {topic}?",
        "What can someone do to avoid certain risk factors for {topic}?",
        "How can the risk of {topic} be lowered?",
        "What steps can help protect against {topic}?",
        "How might someone reduce their likelihood of {topic}?",
        "What are general prevention strategies for {topic}?",
        "How can I lower my risk of {topic}?",
        "What preventive habits are associated with {topic}?",
        "How do people help prevent {topic}?",
        "Can you share prevention tips for {topic}?",
        "What should people consider to reduce the risk of {topic}?",
    ],

    "diagnosis_tests": [
        "How is {topic} diagnosed?",
        "Which tests may help identify {topic}?",
        "How do healthcare professionals check for {topic}?",
        "What tests are used when checking for {topic}?",
        "How can a doctor determine whether someone has {topic}?",
        "What medical examinations may be used for {topic}?",
        "How do doctors investigate possible {topic}?",
        "Can a laboratory test help detect {topic}?",
        "What assessments may be involved in diagnosing {topic}?",
        "How is suspected {topic} evaluated?",
        "What should I know about testing for {topic}?",
        "Which diagnostic methods are used for {topic}?",
        "How might a healthcare professional confirm {topic}?",
        "What happens during an evaluation for {topic}?",
        "What checks might a doctor perform for {topic}?",
        "How do clinicians assess possible {topic}?",
        "Can you explain the diagnostic process for {topic}?",
        "What types of tests might be considered for {topic}?",
        "How can healthcare workers identify {topic}?",
        "What is involved in checking someone for {topic}?",
    ],

    "general_management": [
        "How is {topic} generally managed?",
        "What are the general management options for {topic}?",
        "How can someone manage {topic} with professional guidance?",
        "What approaches are used to manage {topic}?",
        "How do healthcare professionals help people with {topic}?",
        "What general care may help with {topic}?",
        "What should someone know about managing {topic}?",
        "Can you explain the usual management of {topic}?",
        "What support may be available for people with {topic}?",
        "How is care for {topic} usually approached?",
        "What are common ways to manage {topic}?",
        "What general steps may help someone living with {topic}?",
        "How can a person discuss management of {topic} with a doctor?",
        "What care options might be discussed for {topic}?",
        "How do professionals support patients with {topic}?",
        "What does managing {topic} usually involve?",
        "Can you describe general care for {topic}?",
        "What approaches might a clinician recommend for {topic}?",
        "Where can someone learn about managing {topic}?",
        "What should I ask my healthcare professional about managing {topic}?",
    ],

    "when_to_seek_care": [
        "When should someone seek medical care for {topic}?",
        "What warning signs of {topic} require medical attention?",
        "When should a person contact a healthcare professional about {topic}?",
        "What symptoms related to {topic} should not be ignored?",
        "When is it important to get help for {topic}?",
        "What signs mean someone should see a doctor about {topic}?",
        "When should I get medical advice about {topic}?",
        "What concerning changes associated with {topic} need attention?",
        "When might {topic} require urgent medical assessment?",
        "What warning signs should prompt someone to seek help for {topic}?",
        "When is professional advice needed for {topic}?",
        "What symptoms of {topic} may need prompt evaluation?",
        "When should someone get checked for possible {topic}?",
        "What are reasons to contact a doctor about {topic}?",
        "When should symptoms associated with {topic} be assessed?",
        "What signs of {topic} may require urgent attention?",
        "How can someone tell when to seek help for {topic}?",
        "When should a person get medical support for {topic}?",
        "What situations involving {topic} call for medical advice?",
        "When should I speak to a healthcare professional about {topic}?",
    ],
}


# ============================================================
# CREATE A CLEAN DATASET
# ============================================================

def expand_dataset():
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(
            f"Medical topics file not found: {TOPICS_PATH}"
        )

    # Save the current dataset before replacing it.
    # An existing backup will not be overwritten.
    if OUTPUT_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(OUTPUT_PATH, BACKUP_PATH)
        print(f"Previous dataset backed up to: {BACKUP_PATH}")

    topics_df = pd.read_csv(TOPICS_PATH)

    # Use the correct column explicitly.
    if "topic_name" not in topics_df.columns:
        raise ValueError(
            "The topics file must contain a 'topic_name' column."
        )

    topics = (
        topics_df["topic_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not topics:
        raise ValueError("No medical topic names were found.")

    rows = []
    seen_questions = set()

    # Generate all questions from scratch.
    # Do not reuse the corrupted numeric-topic dataset.
    for intent, patterns in INTENT_PATTERNS.items():
        for topic in topics:
            for pattern in patterns:
                question = pattern.format(topic=topic).strip()
                question_key = question.casefold()

                if question_key not in seen_questions:
                    rows.append({
                        "text": question,
                        "intent": intent,
                    })
                    seen_questions.add(question_key)

    # Safety check: numeric placeholders must not appear.
    invalid_rows = [
        row for row in rows
        if row["text"].strip().split()[-1].isdigit()
    ]

    if invalid_rows:
        raise ValueError(
            "Invalid numeric-topic questions were detected. "
            "Dataset was not written."
        )

    # Confirm the expected balanced dataset before saving.
    counts = {}

    for row in rows:
        counts[row["intent"]] = counts.get(row["intent"], 0) + 1

    expected_per_intent = len(topics) * 20

    if len(counts) != len(INTENT_PATTERNS):
        raise ValueError("Unexpected number of intent categories.")

    if any(
        counts.get(intent, 0) != expected_per_intent
        for intent in INTENT_PATTERNS
    ):
        raise ValueError(
            "Dataset is not balanced. "
            f"Expected {expected_per_intent} questions per intent."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "intent"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 65)
    print("MEDICAL INTENT DATASET REBUILT")
    print("=" * 65)
    print(f"Medical topics: {len(topics)}")
    print(f"Question patterns per intent: 20")
    print(f"Total questions: {len(rows)}")
    print(f"Unique intents: {len(counts)}")
    print(f"Questions per intent: {expected_per_intent}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample questions:")
    for row in rows[:10]:
        print(f"{row['text']} -> {row['intent']}")

    print("\nFinal examples:")
    for row in rows[-5:]:
        print(f"{row['text']} -> {row['intent']}")

    print("\nDataset expansion completed successfully.")


if __name__ == "__main__":
    expand_dataset()