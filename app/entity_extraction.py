import re
import spacy


# --------------------------------------------------
# Load spaCy English model
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# Custom Order ID pattern
# Example:
# ORD12345
# ORD56789
# --------------------------------------------------

ORDER_ID_PATTERN = r"\bORD\d{5,}\b"


def extract_entities(text):
    """
    Extract entities from user input using
    spaCy + custom rules.
    """

    doc = nlp(text)

    entities = []

    # --------------------------------------------------
    # 1. spaCy entities
    # --------------------------------------------------

    for ent in doc.ents:

        # Ignore spaCy's incorrect classification
        # when the entity is actually an order ID.
         if re.fullmatch(ORDER_ID_PATTERN, ent.text.upper()):
            continue

         entities.append({
        "text": ent.text,
        "label": ent.label_,
        "description": spacy.explain(ent.label_)
    })

    # --------------------------------------------------
    # 2. Custom Order ID extraction
    # --------------------------------------------------

    order_ids = re.findall(
        ORDER_ID_PATTERN,
        text.upper()
    )

    for order_id in order_ids:

        entities.append({
            "text": order_id,
            "label": "ORDER_ID",
            "description": "Customer order ID"
        })

    return entities


# --------------------------------------------------
# Test NER
# --------------------------------------------------

if __name__ == "__main__":

    test_messages = [
        "Please check order ORD12345.",
        "My order ORD56789 has not arrived.",
        "John placed order ORD98765 yesterday.",
        "I need help with my order.",
        "My package should arrive tomorrow.",
        "The order was placed on January 15."
    ]

    print("=" * 60)
    print("AI CHATBOT - NAMED ENTITY RECOGNITION")
    print("=" * 60)

    for message in test_messages:

        print("\nUser:", message)

        entities = extract_entities(message)

        if entities:

            print("Entities:")

            for entity in entities:

                print(
                    f"  {entity['text']} "
                    f"→ {entity['label']} "
                    f"({entity['description']})"
                )

        else:

            print("  No entities detected.")

    print("\n" + "=" * 60)
    print("NER TEST COMPLETED")
    print("=" * 60)