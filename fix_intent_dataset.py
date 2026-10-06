import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/intents.csv")


# ============================================================
# CHECK DATASET EXISTS
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("INTENT DATASET CORRECTION")
print("=" * 60)

print(f"\nOriginal records: {len(df)}")

print("\nOriginal intent distribution:")
print(df["intent"].value_counts().sort_index())


# ============================================================
# CREATE BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = DATA_PATH.parent / (
    f"intents_backup_{timestamp}.csv"
)

df.to_csv(
    backup_path,
    index=False
)

print(
    f"\nBackup created successfully:\n"
    f"{backup_path}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {"text", "intent"}

missing_columns = required_columns - set(df.columns)

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# KEEP EXACTLY 25 GREETING EXAMPLES
# ============================================================

greeting_df = df[
    df["intent"] == "greeting"
].copy()


if len(greeting_df) < 25:

    raise ValueError(
        "There are fewer than 25 greeting examples."
    )


# Keep the FIRST 25 existing greeting examples.
greeting_df = greeting_df.head(25)


# ============================================================
# KEEP ALL OTHER INTENTS
# ============================================================

other_df = df[
    df["intent"] != "greeting"
].copy()


# ============================================================
# COMBINE DATASET
# ============================================================

corrected_df = pd.concat(
    [
        greeting_df,
        other_df
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

corrected_df = corrected_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE CORRECTED DATASET
# ============================================================

corrected_df.to_csv(
    DATA_PATH,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CORRECTED DATASET")
print("=" * 60)

print(
    f"\nTotal records: {len(corrected_df)}"
)

print("\nIntent distribution:")

distribution = (
    corrected_df["intent"]
    .value_counts()
    .sort_index()
)

print(distribution)


# ============================================================
# FINAL VALIDATION
# ============================================================

expected_intents = 15

expected_records_per_intent = 25

expected_total_records = (
    expected_intents *
    expected_records_per_intent
)


print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)


if len(corrected_df) != expected_total_records:

    raise ValueError(
        f"Expected {expected_total_records} records "
        f"but found {len(corrected_df)}."
    )


if corrected_df["intent"].nunique() != expected_intents:

    raise ValueError(
        f"Expected {expected_intents} intents "
        f"but found {corrected_df['intent'].nunique()}."
    )


if not all(
    distribution == expected_records_per_intent
):

    raise ValueError(
        "Dataset is not balanced correctly."
    )


if corrected_df["text"].isna().any():

    raise ValueError(
        "Dataset contains missing text values."
    )


if corrected_df["intent"].isna().any():

    raise ValueError(
        "Dataset contains missing intent values."
    )


print("\n✓ Total records: 375")
print("✓ Unique intents: 15")
print("✓ 25 examples per intent")
print("✓ No missing text values")
print("✓ No missing intent values")
print("✓ Dataset is balanced")


print("\nCorrected dataset saved to:")

print(DATA_PATH)

print("\nBackup saved to:")

print(backup_path)

print("\n" + "=" * 60)
print("DATASET CORRECTION COMPLETE")
print("=" * 60)