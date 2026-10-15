import re

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean and normalize user text.

    The preprocessing keeps:
    - letters
    - numbers
    - spaces
    - hyphens

    Hyphens are important for medical terms such as:
    COVID-19
    25-hydroxyvitamin
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace slash with a space
    text = text.replace("/", " ")

    # Keep letters, numbers, spaces, and hyphens
    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    # Normalize repeated hyphens
    text = re.sub(r"-{2,}", "-", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Convert cleaned text into a list of tokens.

    Hyphenated medical terms remain as one token.
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    tokens = re.findall(
        r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b",
        cleaned_text
    )

    return tokens


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Return normalized text using the cleaning pipeline.
    """

    return clean_text(text)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    test_messages = [
        "What are the symptoms of Diabetes?",
        "What is COVID-19?",
        "Tell me about Acid Reflux / GERD.",
        "I have a headache!!!",
        "How can I prevent kidney stones?",
        "What are the symptoms of Vitamin D deficiency?",
        "Can anxiety affect my sleep?"
    ]

    print("=" * 60)
    print("MEDICAL NLP PREPROCESSING TEST")
    print("=" * 60)

    for message in test_messages:

        cleaned = clean_text(message)
        tokens = tokenize_text(message)

        print("\nOriginal:")
        print(message)

        print("Cleaned:")
        print(cleaned)

        print("Tokens:")
        print(tokens)

    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETED")
    print("=" * 60)