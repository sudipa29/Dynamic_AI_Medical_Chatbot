
import pandas as pd
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "medical" / "medical_knowledge.csv"
BACKUP_PATH = BASE_DIR / "data" / "medical" / "medical_knowledge_before_spacing_fix.csv"


# ============================================================
# 2. LOAD AND BACK UP THE KNOWLEDGE BASE
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Knowledge base not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

if "content" not in df.columns:
    raise ValueError("The CSV does not contain a 'content' column.")

if not BACKUP_PATH.exists():
    df.to_csv(BACKUP_PATH, index=False, encoding="utf-8")
    print(f"Backup created: {BACKUP_PATH}")


# ============================================================
# 3. FIX KNOWN SPACING ERRORS
# ============================================================

replacements = {
    "severityand should": "severity and should",
    "bedetermined": "be determined",
    "atwork": "at work",
    "andshould": "and should",
}

for old_text, new_text in replacements.items():
    df["content"] = df["content"].astype(str).str.replace(
        old_text,
        new_text,
        regex=False,
    )


# ============================================================
# 4. SAVE AND VERIFY
# ============================================================

df.to_csv(CSV_PATH, index=False, encoding="utf-8")

verified_df = pd.read_csv(CSV_PATH)

print("\nCSV spacing repair completed.")
print("Total records:", len(verified_df))

for information_type in ["General Management", "Causes / Risk Factors"]:
    rows = verified_df[
        (verified_df["topic"].str.lower() == "asthma")
        & (
            verified_df["information_type"].str.lower()
            == information_type.lower()
        )
    ]

    if not rows.empty:
        print(f"\n{information_type}:")
        print(rows["content"].iloc[0])

remaining_errors = []

for old_text in replacements:
    if verified_df["content"].astype(str).str.contains(
        old_text,
        regex=False,
    ).any():
        remaining_errors.append(old_text)

if remaining_errors:
    print("\nPlease investigate these remaining patterns:", remaining_errors)
else:
    print("\nAll targeted spacing errors have been removed.")
