
import re


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sentences = [
        "Hello!!!",
        "Where IS my order???",
        "  I WANT A REFUND  ",
        "My payment failed!!!",
        "Can you tell me about this product?"
    ]

    print("=" * 60)
    print("TEXT PREPROCESSING TEST")
    print("=" * 60)

    for sentence in test_sentences:

        cleaned = clean_text(sentence)

        print(
            f"\nOriginal : {sentence}"
        )

        print(
            f"Cleaned  : {cleaned}"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING TEST COMPLETED"
    )

    print(
        "=" * 60
    )