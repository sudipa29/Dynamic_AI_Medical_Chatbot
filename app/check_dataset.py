import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/raw/intents.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Display column names
# --------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Display first 10 rows
# --------------------------------------------------

print("\nFirst 10 rows:")
print(df.head(10))


# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. Check duplicate rows
# --------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 6. Check number of intents
# --------------------------------------------------

print("\nNumber of unique intents:", df["intent"].nunique())


# --------------------------------------------------
# 7. Display all intents
# --------------------------------------------------

print("\nIntent list:")
print(df["intent"].unique())


# --------------------------------------------------
# 8. Count examples for each intent
# --------------------------------------------------

print("\nExamples per intent:")
print(df["intent"].value_counts())


# --------------------------------------------------
# 9. Check empty text values
# --------------------------------------------------

empty_text = df["text"].astype(str).str.strip().eq("").sum()

print("\nEmpty text values:", empty_text)


# --------------------------------------------------
# 10. Check empty intent values
# --------------------------------------------------

empty_intent = df["intent"].astype(str).str.strip().eq("").sum()

print("Empty intent values:", empty_intent)