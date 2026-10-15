
import csv
from pathlib import Path

# ============================================================
# CREATE A SEPARATE MEDICAL INTENT DATASET
# ============================================================

OUTPUT_PATH = Path("data/medical/medical_intents.csv")

medical_topics = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Migraine",
    "Common Cold",
    "Influenza",
    "Dehydration",
    "Anemia",
    "Obesity",
    "High Cholesterol",
    "Acid Reflux",
    "Gastritis",
    "Food Poisoning",
    "Allergy",
    "Arthritis",
    "Pneumonia",
    "Urinary Tract Infection",
    "Vitamin D Deficiency",
    "COVID-19",
    "Heart Disease",
    "Thyroid Disorders",
    "Depression",
    "Anxiety",
    "Insomnia",
    "Kidney Stones",
]

# Each topic receives one question for each intent.
# Different question patterns help introduce some variation.

intent_templates = {
    "definition": [
        "What is {topic}?",
        "Can you explain {topic}?",
        "Tell me about {topic}.",
    ],
    "symptoms": [
        "What are the symptoms of {topic}?",
        "How can someone recognize {topic}?",
        "What signs may indicate {topic}?",
    ],
    "causes_risk_factors": [
        "What causes {topic}?",
        "What are the risk factors for {topic}?",
        "Why might someone develop {topic}?",
    ],
    "prevention": [
        "How can {topic} be prevented?",
        "What can help reduce the risk of {topic}?",
        "How can someone lower their chances of developing {topic}?",
    ],
    "diagnosis_tests": [
        "How is {topic} diagnosed?",
        "Which tests may help identify {topic}?",
        "How do healthcare professionals check for {topic}?",
    ],
    "general_management": [
        "How is {topic} generally managed?",
        "What are the general management options for {topic}?",
        "How can someone manage {topic} with professional guidance?",
    ],
    "when_to_seek_care": [
        "When should someone seek medical care for {topic}?",
        "What warning signs of {topic} require medical attention?",
        "When should a person contact a healthcare professional about {topic}?",
    ],
}


def create_dataset():
    rows = []

    for topic in medical_topics:
        for intent, templates in intent_templates.items():
            # Rotate the wording patterns across topics.
            template_index = medical_topics.index(topic) % len(templates)
            question = templates[template_index].format(topic=topic)

            rows.append({
                "text": question,
                "intent": intent,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "intent"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("MEDICAL INTENT DATASET CREATED")
    print("=" * 60)
    print(f"File: {OUTPUT_PATH}")
    print(f"Total questions: {len(rows)}")
    print(f"Unique intents: {len(intent_templates)}")
    print(f"Medical topics: {len(medical_topics)}")
    print("\nQuestions per intent:")

    for intent in intent_templates:
        count = sum(row["intent"] == intent for row in rows)
        print(f"{intent}: {count}")

    print("\nSample questions:")
    for row in rows[:7]:
        print(f"{row['text']} -> {row['intent']}")

    print("\nDataset creation completed.")


if __name__ == "__main__":
    create_dataset()