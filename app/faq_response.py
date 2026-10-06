import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import clean_text
from app.intent_detection import predict_intent

# ============================================================
# FAQ DATASET
# ============================================================

FAQ_PATH = "data/raw/faqs.csv"

faq_df = pd.read_csv(FAQ_PATH)

faq_df["clean_question"] = faq_df["question"].apply(clean_text)

# ============================================================
# FAQ RESPONSE FUNCTION
# ============================================================

def get_faq_response(user_message, threshold=0.25):
    
    predicted_intent, intent_confidence = predict_intent(user_message)

    # --------------------------------------------------------
    # FILTER FAQS BY INTENT
    # --------------------------------------------------------
    matching_faqs = faq_df[faq_df["intent"] == predicted_intent].copy()

    if matching_faqs.empty:
        return None

    # --------------------------------------------------------
    # CREATE TF-IDF VECTORS
    # --------------------------------------------------------
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    faq_vectors = vectorizer.fit_transform(matching_faqs["clean_question"])

    user_vector = vectorizer.transform([clean_text(user_message)])

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------
    similarities = cosine_similarity(user_vector, faq_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    # --------------------------------------------------------
    # THRESHOLD CHECK
    # --------------------------------------------------------
    if best_score < threshold:
        return None

    matched_row = matching_faqs.iloc[best_index]

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------
    return {
        "answer": matched_row["answer"],
        "intent": predicted_intent,
        "intent_confidence": intent_confidence,
        "similarity": best_score,
        "matched_question": matched_row["question"]
    }