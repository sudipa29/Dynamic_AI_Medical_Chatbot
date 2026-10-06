import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from app.preprocessing import clean_text


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/raw/intents.csv"

df = pd.read_csv(DATA_PATH)

df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

X = df["clean_text"]
y = df["intent"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 5. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# 7. INTENT PREDICTION WITH DETAILS
# ============================================================

def predict_intent_details(text):
    """
    Predict the user's intent and provide confidence details.

    Runtime fallback rules:

    1. If confidence margin < 5%, return unknown.
    2. If confidence < 15% AND margin < 8%, return unknown.
    3. Otherwise, keep the model's top prediction.
    """

    cleaned_text = clean_text(text)

    text_vector = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(text_vector)[0]

    # Sort probabilities from highest to lowest
    sorted_indexes = probabilities.argsort()[::-1]

    # --------------------------------------------------------
    # Top prediction
    # --------------------------------------------------------

    top_index = sorted_indexes[0]

    original_top_intent = model.classes_[top_index]

    top_confidence = probabilities[top_index]

    # --------------------------------------------------------
    # Second prediction
    # --------------------------------------------------------

    second_index = sorted_indexes[1]

    second_intent = model.classes_[second_index]

    second_confidence = probabilities[second_index]

    # --------------------------------------------------------
    # Confidence margin
    # --------------------------------------------------------

    confidence_margin = top_confidence - second_confidence

    # --------------------------------------------------------
    # Runtime fallback
    # --------------------------------------------------------

    final_intent = original_top_intent

    # Rule 1:
    # Very small difference between top two intents
    if confidence_margin < 0.05:

        final_intent = "unknown"

    # Rule 2:
    # Low confidence and small difference between intents
    elif top_confidence < 0.15 and confidence_margin < 0.08:

        final_intent = "unknown"

    return (
        final_intent,
        top_confidence,
        second_intent,
        second_confidence,
        confidence_margin
    )


# ============================================================
# 8. SIMPLE PREDICTION FUNCTION
# ============================================================

def predict_intent(text):
    """
    Simple intent prediction function.

    This returns the final intent after applying
    the runtime fallback rules.
    """

    (
        final_intent,
        top_confidence,
        _,
        _,
        _
    ) = predict_intent_details(text)

    return final_intent, top_confidence


# ============================================================
# 9. CONFIDENCE ANALYSIS
# ============================================================

def analyze_test_confidence():

    probabilities = model.predict_proba(X_test_tfidf)

    top_confidences = probabilities.max(axis=1)

    predicted_labels = model.classes_[probabilities.argmax(axis=1)]

    confidence_df = pd.DataFrame({
        "actual_intent": y_test.values,
        "predicted_intent": predicted_labels,
        "confidence": top_confidences
    })

    confidence_df["correct"] = (
        confidence_df["actual_intent"]
        == confidence_df["predicted_intent"]
    )

    correct_confidence = confidence_df.loc[
        confidence_df["correct"],
        "confidence"
    ]

    incorrect_confidence = confidence_df.loc[
        ~confidence_df["correct"],
        "confidence"
    ]

    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    print(
        f"\nMinimum Confidence: "
        f"{confidence_df['confidence'].min():.2%}"
    )

    print(
        f"Maximum Confidence: "
        f"{confidence_df['confidence'].max():.2%}"
    )

    print(
        f"Average Confidence: "
        f"{confidence_df['confidence'].mean():.2%}"
    )

    print(
        f"\nAverage Confidence - Correct Predictions: "
        f"{correct_confidence.mean():.2%}"
    )

    if len(incorrect_confidence) > 0:

        print(
            f"Average Confidence - Incorrect Predictions: "
            f"{incorrect_confidence.mean():.2%}"
        )

        print(
            f"\nLowest Confidence - Incorrect Predictions: "
            f"{incorrect_confidence.min():.2%}"
        )

        print(
            f"Highest Confidence - Incorrect Predictions: "
            f"{incorrect_confidence.max():.2%}"
        )


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INTENT CLASSIFICATION MODEL")
    print("=" * 60)

    print(f"\nDataset Records: {len(df)}")
    print(f"Unique Intents: {df['intent'].nunique()}")
    print(f"Training Records: {len(X_train)}")
    print(f"Testing Records: {len(X_test)}")

    print(f"\nTraining Features: {X_train_tfidf.shape}")
    print(f"Testing Features: {X_test_tfidf.shape}")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    analyze_test_confidence()


    # ========================================================
    # MANUAL TESTING
    # ========================================================

    test_messages = [

        "Where is my package?",

        "I need my money back",

        "My card payment did not work",

        "Hello there",

        "I want to cancel my order",

        "Thank you for your help",

        "Tell me about this item",

        "Can you help me?",

        "What is quantum physics?",

        "What is the weather today?",

        "Tell me a joke",

        "How are you?",

        "Who are you?",

        "I have a problem",

        "My order is late",

        "My card isn't working",

        "I want my money back",

        "What products do you sell?"
    ]


    print("\n" + "=" * 60)
    print("MANUAL INTENT TESTS")
    print("=" * 60)


    for message in test_messages:

        (
            final_intent,
            confidence,
            second_intent,
            second_confidence,
            margin
        ) = predict_intent_details(message)

        print(f"\nMessage: {message}")

        print(f"Final Intent: {final_intent}")

        print(
            f"Top Confidence: "
            f"{confidence:.2%}"
        )

        print(f"Second Intent: {second_intent}")

        print(
            f"Second Confidence: "
            f"{second_confidence:.2%}"
        )

        print(
            f"Confidence Margin: "
            f"{margin:.2%}"
        )