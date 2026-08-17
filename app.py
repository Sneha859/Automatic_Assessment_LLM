import streamlit as st
import numpy as np
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer

from modules.preprocessing.preprocess import preprocess_text
from modules.syntactic_analysis.syntax_analyzer import extract_syntactic_features
from modules.explanation.explanation_module import generate_explanation


# ----------------------------
# NLTK setup
# ----------------------------

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ----------------------------
# Load models
# ----------------------------

@st.cache_resource
def load_models():

    ridge_model = joblib.load("models/ridge_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    return ridge_model, scaler, semantic_model


ridge_model, scaler, semantic_model = load_models()


# ----------------------------
# Page config
# ----------------------------

st.set_page_config(
    page_title="Essay Assessment",
    layout="wide"
)

st.title("🧠 Automatic Answer Assessment")


# ----------------------------
# Inputs
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    question = st.text_area("Essay Question", height=120)

with col2:
    answer = st.text_area("Student Answer", height=120)

evaluate = st.button("Evaluate Answer")


# ----------------------------
# Evaluation
# ----------------------------

if evaluate:

    if question == "" or answer == "":
        st.warning("Please enter both question and answer.")
        st.stop()

    st.info("Processing answer...")


    # ----------------------------
    # Preprocess (for ML model)
    # ----------------------------

    processed_answer = preprocess_text(
        answer,
        stop_words,
        lemmatizer
    )


    # ----------------------------
    # Syntactic features
    # ----------------------------

    avg_sentence_length, noun_ratio, verb_ratio, adj_ratio, readability = extract_syntactic_features(
        answer
    )


    # ----------------------------
    # Semantic similarity
    # ----------------------------

    q_embedding = semantic_model.encode(question)
    a_embedding = semantic_model.encode(answer)

    similarity = np.dot(q_embedding, a_embedding) / (
        np.linalg.norm(q_embedding) * np.linalg.norm(a_embedding)
    )

    similarity = float(similarity)

    semantic_component = max(0, min((similarity - 0.3) / 0.6, 1))


    # ====================================================
    # ✅ NEW: Detect copied answer
    # ====================================================

    is_copied_answer = False

    question_clean = question.strip().lower()
    answer_clean = answer.strip().lower()

    # Exact match
    if question_clean == answer_clean:
        is_copied_answer = True

    # High similarity + similar length
    elif similarity > 0.9:
        q_len = len(question.split())
        a_len = len(answer.split())

        if abs(q_len - a_len) <= 3:
            is_copied_answer = True


    # ----------------------------
    # Embedding features
    # ----------------------------

    embedding_features = semantic_model.encode(answer)


    # ----------------------------
    # Feature vector
    # ----------------------------

    features = np.concatenate([
        [avg_sentence_length],
        [noun_ratio],
        [verb_ratio],
        [adj_ratio],
        [readability],
        [similarity],
        embedding_features
    ])

    features = features.reshape(1, -1)

    features_scaled = scaler.transform(features)


    # ----------------------------
    # Ridge prediction
    # ----------------------------

    predicted_score = ridge_model.predict(features_scaled)[0]
    predicted_score = np.clip(predicted_score, 1, 6)

    ridge_component = predicted_score / 6


    # ----------------------------
    # Syntactic component
    # ----------------------------

    length_score = min(avg_sentence_length / 18, 1)

    grammar_score = min((noun_ratio + verb_ratio + adj_ratio) / 3, 1)

    syntactic_component = (length_score + grammar_score) / 2


    # ----------------------------
    # Final score formula
    # ----------------------------

    final_quality = (
        0.65 * semantic_component +
        0.20 * syntactic_component +
        0.15 * ridge_component
    )

    final_score = final_quality * 10


    # ----------------------------
    # High quality boost
    # ----------------------------

    if similarity > 0.75 and syntactic_component > 0.6:
        final_score += 1.2

    elif similarity > 0.65:
        final_score += 0.6


    # ----------------------------
    # Irrelevant answer penalty
    # ----------------------------

    if similarity < 0.35:
        final_score *= 0.4


    # ====================================================
    # ✅ NEW: Penalize copied answer
    # ====================================================

    if is_copied_answer:
        final_score = 0


    final_score = round(min(final_score, 10), 2)


    # ====================================================
    # ✅ NEW: Zero-score condition
    # ====================================================

    is_totally_wrong = False

    if final_score <= 0.5 or is_copied_answer:
        final_score = 0
        content_score = 0
        language_score = 0
        organisation_score = 0
        is_totally_wrong = True

    else:
        content_score = round(final_score * 0.5, 2)
        language_score = round(final_score * 0.2, 2)
        organisation_score = round(final_score * 0.3, 2)


    # ----------------------------
    # Explanation
    # ----------------------------

    explanation, feedback = generate_explanation(
        question,
        answer,
        similarity,
        content_score,
        language_score,
        organisation_score
    )


    st.success("Evaluation Complete")


    # ----------------------------
    # Score cards
    # ----------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Final Score", f"{final_score}/10")
    c2.metric("Content", f"{content_score}/5")
    c3.metric("Language", f"{language_score}/2")
    c4.metric("Organisation", f"{organisation_score}/3")


    # ====================================================
    # ✅ NEW: Custom error messages
    # ====================================================

    if is_totally_wrong:
        if is_copied_answer:
            st.error("❌ Answer is copied from the question and does not provide any explanation.")
        else:
            st.error("❌ Answer is totally wrong and does not match the question.")


    # ----------------------------
    # Explanation
    # ----------------------------

    with st.expander("Explanation"):
        st.write(explanation)

    with st.expander("Feedback"):
        st.write(feedback)


    if similarity < 0.35:
        st.warning("⚠ Answer may not be relevant to the question.")