
"""
Text preprocessing utilities for the Medical AI Chatbot.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Normalize text for intent detection and medical knowledge retrieval.
    Preserves meaningful characters in terms such as COVID-19.
    """
    if text is None:
        return ""

    text = str(text).lower().strip()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Keep letters, digits, spaces, and hyphens.
    # This preserves medical terms such as COVID-19.
    text = re.sub(r"[^\w\s-]", " ", text)

    # Remove extra spaces created during cleaning.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Convert cleaned text into a list of tokens.
    """
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    return cleaned_text.split()
