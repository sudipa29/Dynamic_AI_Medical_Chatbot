import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import clean_text


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/raw/intents.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Original shape:", df.shape)


# --------------------------------------------------
# 2. Clean text
# --------------------------------------------------

df["clean_text"] = df["text"].apply(clean_text)

print("\nText cleaning completed.")

# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

processed_file_path = "data/processed/intents_clean.csv"

df.to_csv(processed_file_path, index=False)

print("\nProcessed dataset saved to:")
print(processed_file_path)

# --------------------------------------------------
# 3. Define X and y
# --------------------------------------------------

X = df["clean_text"]

y = df["intent"]


print("\nNumber of input examples:", len(X))
print("Number of labels:", len(y))


# --------------------------------------------------
# 4. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\nTrain/Test Split completed.")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())