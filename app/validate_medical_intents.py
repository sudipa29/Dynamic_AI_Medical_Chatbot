
import pandas as pd
from pathlib import Path

# ============================================================
# MEDICAL INTENT DATASET VALIDATION
# ============================================================

DATA_PATH = Path("data/medical/medical_intents.csv")

EXPECTED_COLUMNS = ["text", "intent"]

EXPECTED_INTENTS = [
    "definition",
    "symptoms",
    "causes_risk_factors",
    "prevention",
    "diagnosis_tests",
    "general_management",
    "when_to_seek_care",
]

EXPECTED_QUESTIONS_PER_INTENT = 500
EXPECTED_TOTAL_RECORDS = 3500


def validate_dataset():
    print("=" * 65)
    print("MEDICAL INTENT DATASET VALIDATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as error:
        print(f"ERROR: Could not read dataset: {error}")
        return

    validation_passed = True

    # 1. Check columns
    print("\n1. COLUMN CHECK")
    print("Columns:", list(df.columns))

    if list(df.columns) == EXPECTED_COLUMNS:
        print("PASS: Column names and order are correct.")
    else:
        print("FAIL: Expected columns:", EXPECTED_COLUMNS)
        validation_passed = False

    # Stop if required columns are missing
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Cannot continue: required columns are missing.")
        return

    # 2. Check total records
    print("\n2. RECORD COUNT")
    print("Total records:", len(df))

    if len(df) == EXPECTED_TOTAL_RECORDS:
        print("PASS: Exactly 3,500 records found.")
    else:
        print(
            f"FAIL: Expected {EXPECTED_TOTAL_RECORDS} records, "
            f"found {len(df)}."
        )
        validation_passed = False

    # 3. Check missing values
    print("\n3. MISSING VALUES")
    missing_values = df[EXPECTED_COLUMNS].isnull().sum()
    print(missing_values)

    if missing_values.sum() == 0:
        print("PASS: No missing values.")
    else:
        print("FAIL: Missing values found.")
        validation_passed = False

    # 4. Check duplicate questions
    print("\n4. DUPLICATE QUESTIONS")
    normalized_text = (
        df["text"].fillna("").astype(str).str.strip().str.casefold()
    )
    duplicate_count = normalized_text.duplicated().sum()
    print("Duplicate questions:", duplicate_count)

    if duplicate_count == 0:
        print("PASS: No duplicate questions.")
    else:
        print("FAIL: Duplicate questions found.")
        validation_passed = False

    # 5. Check unique intents
    print("\n5. UNIQUE INTENTS")
    actual_intents = set(df["intent"].dropna().unique())
    expected_intents = set(EXPECTED_INTENTS)

    print("Intents found:", sorted(actual_intents))

    if actual_intents == expected_intents:
        print("PASS: All expected intents are present.")
    else:
        print("FAIL: Intent labels do not match expectations.")
        print("Missing:", sorted(expected_intents - actual_intents))
        print("Unexpected:", sorted(actual_intents - expected_intents))
        validation_passed = False

    # 6. Check questions per intent
    print("\n6. QUESTIONS PER INTENT")
    intent_counts = df["intent"].value_counts().reindex(
        EXPECTED_INTENTS, fill_value=0
    )
    print(intent_counts)

    if (intent_counts == EXPECTED_QUESTIONS_PER_INTENT).all():
        print("PASS: Every intent has exactly 500 questions.")
    else:
        print("FAIL: Some intents do not have 500 questions.")
        validation_passed = False

    # 7. Check empty or whitespace-only questions
    print("\n7. EMPTY TEXT CHECK")
    empty_text_count = (
        df["text"].fillna("").astype(str).str.strip().eq("").sum()
    )
    print("Empty questions:", empty_text_count)

    if empty_text_count == 0:
        print("PASS: No empty questions.")
    else:
        print("FAIL: Empty questions found.")
        validation_passed = False

    # 8. Check text and intent values are strings
    print("\n8. TEXT AND LABEL FORMAT CHECK")
    non_string_text_count = sum(
        not isinstance(value, str) for value in df["text"].dropna()
    )
    non_string_intent_count = sum(
        not isinstance(value, str) for value in df["intent"].dropna()
    )

    print("Non-string questions:", non_string_text_count)
    print("Non-string intent labels:", non_string_intent_count)

    if non_string_text_count == 0 and non_string_intent_count == 0:
        print("PASS: Questions and intent labels use text values.")
    else:
        print("FAIL: Unexpected value types found.")
        validation_passed = False

    # 9. Display sample records
    print("\n9. SAMPLE RECORDS")
    print(df.head(10).to_string(index=False))

    # 10. Final result
    print("\n" + "=" * 65)

    if validation_passed:
        print("FINAL RESULT: ALL VALIDATION CHECKS PASSED")
    else:
        print("FINAL RESULT: SOME VALIDATION CHECKS FAILED")

    print("=" * 65)


if __name__ == "__main__":
    validate_dataset()