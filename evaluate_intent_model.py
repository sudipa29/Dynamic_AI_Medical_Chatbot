
"""
Medical AI Chatbot - Intent Model Evaluation

Run from your project root:
    python evaluate_intent_model.py
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.intent_detection import normalize_text, EXPECTED_INTENTS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medical" / "medical_intents.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    print("=" * 65)
    print("MEDICAL AI CHATBOT - MODEL EVALUATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    if not {"text", "intent"}.issubset(df.columns):
        print("ERROR: Dataset must contain 'text' and 'intent' columns.")
        sys.exit(1)

    df = df[["text", "intent"]].dropna().copy()
    df["text"] = df["text"].map(normalize_text)
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]
    df = df[df["intent"].isin(EXPECTED_INTENTS)]
    df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)

    if df.empty:
        print("ERROR: No valid training examples found.")
        sys.exit(1)

    counts = df["intent"].value_counts()
    missing = set(EXPECTED_INTENTS) - set(counts.index)

    if missing:
        print("ERROR: Missing intents:", ", ".join(sorted(missing)))
        sys.exit(1)

    if counts.min() < 2:
        print("ERROR: Each intent needs at least two examples.")
        sys.exit(1)

    # Split before fitting TF-IDF to prevent data leakage.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["intent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTotal examples: {len(df)}")
    print(f"Training examples: {len(X_train_text)}")
    print(f"Testing examples: {len(X_test_text)}")
    print(f"Number of intents: {df['intent'].nunique()}")

    print("\n" + "=" * 65)
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 65)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=EXPECTED_INTENTS,
        target_names=EXPECTED_INTENTS,
        digits=4,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=EXPECTED_INTENTS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {i}" for i in EXPECTED_INTENTS],
        columns=[f"Predicted: {i}" for i in EXPECTED_INTENTS],
    )

    print("\nCONFUSION MATRIX")
    print("Rows = actual intent; columns = predicted intent.")
    print(matrix_df.to_string())

    # Save results for documentation and further analysis.
    output_dir = BASE_DIR / "reports"
    output_dir.mkdir(exist_ok=True)

    report = classification_report(
        y_test,
        predictions,
        labels=EXPECTED_INTENTS,
        target_names=EXPECTED_INTENTS,
        output_dict=True,
        zero_division=0,
    )

    report_path = output_dir / "intent_classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    matrix_path = output_dir / "intent_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    predictions_path = output_dir / "intent_test_predictions.csv"
    pd.DataFrame({
        "text": X_test_text.to_numpy(),
        "actual_intent": y_test.to_numpy(),
        "predicted_intent": predictions,
        "correct": y_test.to_numpy() == predictions,
    }).to_csv(predictions_path, index=False)

    print("\nEvaluation reports saved in:")
    print(report_path)
    print(matrix_path)
    print(predictions_path)
    print("\nEvaluation completed. Production files were not changed.")


if __name__ == "__main__":
    main()
