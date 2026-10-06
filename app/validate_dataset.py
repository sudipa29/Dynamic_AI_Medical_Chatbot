import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define dataset path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "intents.csv"

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("=" * 60)
print("AI CHATBOT - DATASET VALIDATION")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("\nERROR: intents.csv was not found.")
    print(f"Expected location: {DATA_PATH}")
    exit()

# --------------------------------------------------
# 3. Basic information
# --------------------------------------------------

print("\n--- Dataset Information ---")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn names:")
print(list(df.columns))

# --------------------------------------------------
# 4. Check required columns
# --------------------------------------------------

required_columns = ["text", "intent"]

print("\n--- Required Column Check ---")

for column in required_columns:
    if column in df.columns:
        print(f"✓ {column} column found")
    else:
        print(f"✗ {column} column is missing")

# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\n--- Missing Values ---")

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("✓ No missing values found.")
else:
    print("⚠ Missing values detected.")

# --------------------------------------------------
# 6. Check duplicate questions
# --------------------------------------------------

print("\n--- Duplicate Text Check ---")

duplicate_count = df["text"].duplicated().sum()

print(f"Duplicate questions: {duplicate_count}")

if duplicate_count == 0:
    print("✓ No duplicate questions found.")
else:
    print("⚠ Duplicate questions found.")

# --------------------------------------------------
# 7. Check number of intents
# --------------------------------------------------

print("\n--- Intent Information ---")

intent_counts = df["intent"].value_counts()

print(f"Total unique intents: {df['intent'].nunique()}")

print("\nExamples per intent:")
print(intent_counts)

# --------------------------------------------------
# 8. Check intents with very few examples
# --------------------------------------------------

print("\n--- Small Intent Check ---")

small_intents = intent_counts[intent_counts < 5]

if len(small_intents) == 0:
    print("✓ Every intent has at least 5 examples.")
else:
    print("⚠ These intents have fewer than 5 examples:")
    print(small_intents)

# --------------------------------------------------
# 9. Check empty text values
# --------------------------------------------------

print("\n--- Empty Text Check ---")

empty_text = df["text"].astype(str).str.strip().eq("").sum()

print(f"Empty text rows: {empty_text}")

if empty_text == 0:
    print("✓ No empty questions found.")
else:
    print("⚠ Empty questions found.")

# --------------------------------------------------
# 10. Display sample records
# --------------------------------------------------

print("\n--- Sample Records ---")

print(df.head(10).to_string(index=False))

# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET VALIDATION COMPLETED")
print("=" * 60)