from textblob import TextBlob


def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.

    Returns:
        Positive, Negative, or Neutral
    """

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity


if __name__ == "__main__":

    test_messages = [
        "Thank you, your support was excellent.",
        "I really love this product.",
        "Where is my order?",
        "Can you tell me the delivery date?",
        "My order is extremely late.",
        "I am very frustrated with this service.",
        "The product is terrible."
    ]

    print("=" * 60)
    print("AI CHATBOT - SENTIMENT ANALYSIS")
    print("=" * 60)

    for message in test_messages:

        sentiment, polarity = analyze_sentiment(message)

        print(f"\nUser: {message}")
        print(f"Sentiment: {sentiment}")
        print(f"Polarity : {polarity:.3f}")

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS TEST COMPLETED")
    print("=" * 60)