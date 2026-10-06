import re
import pandas as pd


def clean_text(text):
    """
    Clean user text for machine learning.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def load_and_clean_dataset(file_path):
    """
    Load intents dataset and clean the text column.
    """

    df = pd.read_csv(file_path)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    return df


if __name__ == "__main__":

    file_path = "data/raw/intents.csv"

    df = load_and_clean_dataset(file_path)

    print("\nDataset loaded successfully.")

    print("\nOriginal text:")
    print(df["text"].head(10))

    print("\nCleaned text:")
    print(df["clean_text"].head(10))