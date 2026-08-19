import streamlit as st
import numpy as np
import joblib
import nltk
import os
import html

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer

from modules.preprocessing.preprocess import preprocess_text
from modules.syntactic_analysis.syntax_analyzer import extract_syntactic_features
from modules.explanation.explanation_module import generate_explanation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoGrade AI | Automatic Answer Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.join(BASE_DIR, "nltk_data")

if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)


# ============================================================
# NLTK SETUP
# ============================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(99, 102, 241, 0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 95% 8%,
                rgba(168, 85, 247, 0.08),
                transparent 25%
            ),
            #f8fafc;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 48px;
        margin-bottom: 42px;
        border-radius: 28px;

        background:
            radial-gradient(
                circle at 95% 10%,
                rgba(129, 140, 248, 0.35),
                transparent 22%
            ),
            linear-gradient(
                135deg,
                #111827 0%,
                #172033 45%,
                #312e81 100%
            );

        box-shadow:
            0 24px 60px rgba(15, 23, 42, 0.18);

        color: white;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 210px;
        height: 210px;
        right: -65px;
        top: -75px;
        border-radius: 50%;
        background: rgba(129, 140, 248, 0.16);
    }

    .hero-title {
        position: relative;
        z-index: 2;
        font-size: 2.65rem;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        position: relative;
        z-index: 2;
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 24px;
    }

    .hero-description {
        position: relative;
        z-index: 2;
        max-width: 760px;
        color: #a5b4fc;
        font-size: 0.92rem;
        line-height: 1.7;
        margin-bottom: 25px;
    }

    .status-badge {
        position: relative;
        z-index: 2;
        display: inline-flex;
        align-items: center;
        gap: 9px;

        padding: 9px 16px;
        border-radius: 999px;

        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.35);

        color: #86efac;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 12px rgba(74, 222, 128, 0.8);
    }


    /* ========================================================
       SECTION HEADERS
    ======================================================== */

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 24px;
    }


    /* ========================================================
       INPUT LABELS
    ======================================================== */

    .input-label {
        font-size: 0.86rem;
        font-weight: 750;
        color: #334155;
        margin-bottom: 8px;
    }

    /* ========================================================
       TEXT AREAS
    ======================================================== */

    textarea {
        border-radius: 16px !important;
        border: 1px solid #dbe3ef !important;
        background: #ffffff !important;
        color: #1e293b !important;

        box-shadow:
            0 5px 18px rgba(15, 23, 42, 0.035) !important;
    }

    textarea:focus {
        border-color: #818cf8 !important;
        box-shadow:
            0 0 0 2px rgba(99, 102, 241, 0.12) !important;
    }


    /* ========================================================
       BUTTON
    ======================================================== */

    div.stButton > button {
        border: none !important;
        border-radius: 13px !important;

        padding: 12px 25px !important;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #7c3aed
            ) !important;

        color: white !important;

        font-size: 0.98rem !important;
        font-weight: 750 !important;

        box-shadow:
            0 12px 25px rgba(99, 102, 241, 0.25) !important;

        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 16px 32px rgba(99, 102, 241, 0.32) !important;
    }


    /* ========================================================
       INFO CARDS
    ======================================================== */

    .info-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 24px;
        box-shadow:
            0 12px 30px rgba(15, 23, 42, 0.05);
    }


    /* ========================================================
       SCORE DASHBOARD
    ======================================================== */

    .score-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 24px;
        min-height: 150px;

        box-shadow:
            0 12px 30px rgba(15, 23, 42, 0.055);
    }

    .score-label {
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.7px;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .score-value {
        font-size: 2rem;
        font-weight: 850;
        color: #0f172a;
        letter-spacing: -0.7px;
    }

    .score-description {
        margin-top: 8px;
        color: #64748b;
        font-size: 0.82rem;
    }


    /* ========================================================
       OVERALL SCORE
    ======================================================== */

    .overall-score-wrapper {
        display: flex;
        align-items: center;
        gap: 34px;

        background:
            linear-gradient(
                135deg,
                #ffffff,
                #f8faff
            );

        border: 1px solid #e2e8f0;
        border-radius: 24px;

        padding: 30px;

        margin-top: 20px;

        box-shadow:
            0 16px 40px rgba(15, 23, 42, 0.06);
    }

    .score-ring {
        width: 135px;
        height: 135px;
        border-radius: 50%;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            conic-gradient(
                #6366f1 var(--score),
                #e2e8f0 var(--score)
            );

        flex-shrink: 0;
    }

    .score-ring-inner {
        width: 103px;
        height: 103px;
        border-radius: 50%;

        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;

        background: white;
    }

    .ring-number {
        font-size: 1.45rem;
        font-weight: 850;
        color: #0f172a;
    }

    .ring-label {
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 2px;
    }

    .overall-title {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.7px;
        color: #64748b;
        text-transform: uppercase;
    }

    .overall-score {
        font-size: 2.7rem;
        font-weight: 850;
        color: #111827;
        letter-spacing: -1px;
        margin: 3px 0;
    }

    .overall-message {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.5;
    }


    /* ========================================================
       PROGRESS
    ======================================================== */

    .progress-bg {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: #e9eef6;
        overflow: hidden;
        margin-top: 14px;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                #6366f1,
                #8b5cf6
            );
    }


    /* ========================================================
       INSIGHT CARDS
    ======================================================== */

    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 21px;
        min-height: 145px;

        box-shadow:
            0 10px 26px rgba(15, 23, 42, 0.045);
    }

    .insight-title {
        font-size: 1rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 8px;
    }

    .insight-score {
        font-size: 0.88rem;
        color: #4f46e5;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .insight-text {
        font-size: 0.84rem;
        color: #64748b;
        line-height: 1.55;
    }


    /* ========================================================
       SUMMARY
    ======================================================== */

    .summary-box {
        background:
            linear-gradient(
                135deg,
                #eef2ff,
                #faf5ff
            );

        border: 1px solid #ddd6fe;
        border-radius: 20px;
        padding: 23px 26px;
        margin-top: 22px;
    }

    .summary-title {
        font-size: 1rem;
        font-weight: 800;
        color: #312e81;
        margin-bottom: 13px;
    }

    .summary-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        margin: 5px 7px 5px 0;
        padding: 8px 13px;

        border-radius: 999px;

        background: white;
        border: 1px solid #ddd6fe;

        color: #4338ca;
        font-size: 0.78rem;
        font-weight: 700;
    }


    /* ========================================================
       EXPLANATION / FEEDBACK
    ======================================================== */

    .result-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 23px 26px;
        margin-top: 14px;

        color: #334155;
        line-height: 1.75;

        box-shadow:
            0 8px 24px rgba(15, 23, 42, 0.04);
    }


    /* ========================================================
       PIPELINE
    ======================================================== */

    .pipeline-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 32px 24px;
        margin-top: 18px;

        box-shadow:
            0 14px 34px rgba(15, 23, 42, 0.05);
    }

    .pipeline-title {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 28px;
    }

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .pipeline-step {
        text-align: center;
        min-width: 105px;
    }

    .pipeline-icon {
        width: 48px;
        height: 48px;
        margin: auto;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 15px;

        background: #eef2ff;
        border: 1px solid #c7d2fe;

        font-size: 1.35rem;
    }

    .pipeline-name {
        margin-top: 9px;
        font-size: 0.76rem;
        font-weight: 800;
        color: #334155;
    }

    .pipeline-tech {
        margin-top: 4px;
        font-size: 0.68rem;
        color: #94a3b8;
    }

    .pipeline-arrow {
        color: #818cf8;
        font-size: 1.25rem;
        font-weight: 800;
    }


    /* ========================================================
       TECHNOLOGY BADGES
    ======================================================== */

    .tech-wrapper {
        text-align: center;
        margin-top: 18px;
    }

    .tech-badge {
        display: inline-block;

        padding: 8px 14px;
        margin: 5px;

        border-radius: 999px;

        background: white;
        border: 1px solid #c7d2fe;

        color: #4338ca;

        font-size: 0.76rem;
        font-weight: 750;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.76rem;
        padding-top: 42px;
        line-height: 1.7;
    }


    /* ========================================================
       DIVIDER
    ======================================================== */

    .soft-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 35px 0;
    }

    </style>
    """
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    ridge_model = joblib.load(
        os.path.join(BASE_DIR, "models", "ridge_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(BASE_DIR, "models", "scaler.pkl")
    )

    semantic_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return ridge_model, scaler, semantic_model


ridge_model, scaler, semantic_model = load_models()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 AutoGrade AI
        </div>

        <div class="hero-subtitle">
            LLM-Driven Automatic Answer Assessment & Intelligent Feedback
        </div>

        <div class="hero-description">
            An intelligent assessment system that combines
            NLP preprocessing, semantic understanding,
            syntactic analysis and machine-learning based scoring
            to evaluate student answers automatically.
        </div>

        <div class="status-badge">
            <span class="status-dot"></span>
            AI Assessment Engine Online
        </div>

    </div>
    """
)


# ============================================================
# ASSESSMENT WORKSPACE
# ============================================================

st.html(
    """
    <div class="section-title">
        📝 Assessment Workspace
    </div>

    <div class="section-subtitle">
        Enter the question and student's answer to generate
        an automated rubric-based assessment.
    </div>
    """
)


col1, col2 = st.columns(2, gap="large")


with col1:

    st.html(
        """
        <div class="input-label">
            Essay Question
        </div>
        """
    )

    question = st.text_area(
        "Essay Question",
        height=175,
        placeholder="Example: Explain the importance of machine learning in healthcare.",
        label_visibility="collapsed",
        key="question_input"
    )

    if question:
        question_words = len(question.split())

        st.caption(
            f"Question length: {question_words} words"
        )


with col2:

    st.html(
        """
        <div class="input-label">
            Student Answer
        </div>
        """
    )

    answer = st.text_area(
        "Student Answer",
        height=175,
        placeholder="Enter the student's answer here...",
        label_visibility="collapsed",
        key="answer_input"
    )

    if answer:
        answer_words = len(answer.split())

        st.caption(
            f"Answer length: {answer_words} words"
        )


st.write("")


# ============================================================
# EVALUATE BUTTON
# ============================================================

evaluate = st.button(
    "✨  Evaluate Answer",
    use_container_width=False
)


# ============================================================
# EVALUATION
# ============================================================

if evaluate:

    if question.strip() == "" or answer.strip() == "":
        st.warning(
            "Please enter both the essay question and student's answer."
        )
        st.stop()

    with st.spinner(
        "AI assessment engine is analysing the response..."
    ):

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_answer = preprocess_text(
            answer,
            stop_words,
            lemmatizer
        )


        # ----------------------------------------------------
        # SYNTACTIC FEATURES
        # ----------------------------------------------------

        (
            avg_sentence_length,
            noun_ratio,
            verb_ratio,
            adj_ratio,
            readability
        ) = extract_syntactic_features(answer)


        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

        q_embedding = semantic_model.encode(question)
        a_embedding = semantic_model.encode(answer)

        similarity = np.dot(
            q_embedding,
            a_embedding
        ) / (
            np.linalg.norm(q_embedding) *
            np.linalg.norm(a_embedding)
        )

        similarity = float(similarity)

        semantic_component = max(
            0,
            min(
                (similarity - 0.3) / 0.6,
                1
            )
        )


        # ====================================================
        # COPIED ANSWER DETECTION
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


        # ----------------------------------------------------
        # EMBEDDING FEATURES
        # ----------------------------------------------------

        embedding_features = semantic_model.encode(answer)


        # ----------------------------------------------------
        # FEATURE VECTOR
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RIDGE PREDICTION
        # ----------------------------------------------------

        predicted_score = ridge_model.predict(
            features_scaled
        )[0]

        predicted_score = np.clip(
            predicted_score,
            1,
            6
        )

        ridge_component = predicted_score / 6


        # ----------------------------------------------------
        # SYNTACTIC COMPONENT
        # ----------------------------------------------------

        length_score = min(
            avg_sentence_length / 18,
            1
        )

        grammar_score = min(
            (
                noun_ratio +
                verb_ratio +
                adj_ratio
            ) / 3,
            1
        )

        syntactic_component = (
            length_score +
            grammar_score
        ) / 2


        # ----------------------------------------------------
        # FINAL SCORE FORMULA
        # ----------------------------------------------------

        final_quality = (
            0.65 * semantic_component +
            0.20 * syntactic_component +
            0.15 * ridge_component
        )

        final_score = final_quality * 10


        # ----------------------------------------------------
        # HIGH QUALITY BOOST
        # ----------------------------------------------------

        if similarity > 0.75 and syntactic_component > 0.6:

            final_score += 1.2

        elif similarity > 0.65:

            final_score += 0.6


        # ----------------------------------------------------
        # IRRELEVANT ANSWER PENALTY
        # ----------------------------------------------------

        if similarity < 0.35:

            final_score *= 0.4


        # ====================================================
        # COPIED ANSWER PENALTY
        # ====================================================

        if is_copied_answer:

            final_score = 0


        final_score = round(
            min(final_score, 10),
            2
        )


        # ====================================================
        # ZERO-SCORE CONDITION
        # ====================================================

        is_totally_wrong = False

        if final_score <= 0.5 or is_copied_answer:

            final_score = 0

            content_score = 0
            language_score = 0
            organisation_score = 0

            is_totally_wrong = True

        else:

            content_score = round(
                final_score * 0.5,
                2
            )

            language_score = round(
                final_score * 0.2,
                2
            )

            organisation_score = round(
                final_score * 0.3,
                2
            )


        # ----------------------------------------------------
        # EXPLANATION + FEEDBACK
        # ----------------------------------------------------

        explanation, feedback = generate_explanation(
            question,
            answer,
            similarity,
            content_score,
            language_score,
            organisation_score
        )


        # ====================================================
        # STORE RESULTS
        # ====================================================

        st.session_state["assessment_result"] = {
            "final_score": final_score,
            "content_score": content_score,
            "language_score": language_score,
            "organisation_score": organisation_score,
            "similarity": similarity,
            "explanation": explanation,
            "feedback": feedback,
            "is_copied_answer": is_copied_answer,
            "is_totally_wrong": is_totally_wrong
        }


# ============================================================
# DISPLAY STORED RESULTS
# ============================================================

if "assessment_result" in st.session_state:

    result = st.session_state["assessment_result"]

    final_score = result["final_score"]
    content_score = result["content_score"]
    language_score = result["language_score"]
    organisation_score = result["organisation_score"]
    similarity = result["similarity"]
    explanation = result["explanation"]
    feedback = result["feedback"]

    is_copied_answer = result["is_copied_answer"]
    is_totally_wrong = result["is_totally_wrong"]


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        "Evaluation Complete"
    )


    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.html(
        """
        <div class="section-title">
            📊 Assessment Results
        </div>

        <div class="section-subtitle">
            Automated scoring based on semantic, syntactic
            and machine-learning features.
        </div>
        """
    )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    score_percentage = max(
        0,
        min(
            final_score * 10,
            100
        )
    )

    if final_score >= 8:

        performance_message = (
            "Excellent response with strong relevance and clarity."
        )

    elif final_score >= 6:

        performance_message = (
            "Good response with room for deeper explanation."
        )

    elif final_score >= 4:

        performance_message = (
            "Moderate response that could benefit from stronger detail."
        )

    else:

        performance_message = (
            "The response requires significant improvement."
        )


    st.html(
        f"""
        <div class="overall-score-wrapper">

            <div
                class="score-ring"
                style="--score:{score_percentage}%"
            >

                <div class="score-ring-inner">

                    <div class="ring-number">
                        {score_percentage:.1f}%
                    </div>

                    <div class="ring-label">
                        Performance
                    </div>

                </div>

            </div>


            <div>

                <div class="overall-title">
                    Overall Assessment Score
                </div>

                <div class="overall-score">
                    {final_score}/10
                </div>

                <div class="overall-message">
                    {html.escape(performance_message)}
                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # SCORE CARDS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4,
        gap="medium"
    )


    with c1:

        st.html(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Overall Score
                </div>

                <div class="score-value">
                    {final_score}/10
                </div>

                <div class="score-description">
                    Final assessment score
                </div>

            </div>
            """
        )


    with c2:

        content_percent = min(
            (content_score / 5) * 100,
            100
        )

        st.html(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Content
                </div>

                <div class="score-value">
                    {content_score}/5
                </div>

                <div class="score-description">
                    Relevance & key concepts
                </div>

                <div class="progress-bg">
                    <div
                        class="progress-fill"
                        style="width:{content_percent}%"
                    ></div>
                </div>

            </div>
            """
        )


    with c3:

        language_percent = min(
            (language_score / 2) * 100,
            100
        )

        st.html(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Language
                </div>

                <div class="score-value">
                    {language_score}/2
                </div>

                <div class="score-description">
                    Clarity & language quality
                </div>

                <div class="progress-bg">
                    <div
                        class="progress-fill"
                        style="width:{language_percent}%"
                    ></div>
                </div>

            </div>
            """
        )


    with c4:

        organisation_percent = min(
            (organisation_score / 3) * 100,
            100
        )

        st.html(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Organisation
                </div>

                <div class="score-value">
                    {organisation_score}/3
                </div>

                <div class="score-description">
                    Structure & coherence
                </div>

                <div class="progress-bg">
                    <div
                        class="progress-fill"
                        style="width:{organisation_percent}%"
                    ></div>
                </div>

            </div>
            """
        )


    # ========================================================
    # ERROR / WARNING STATES
    # ========================================================

    if is_totally_wrong:

        if is_copied_answer:

            st.error(
                "❌ Answer appears to be copied from the question "
                "and does not provide an independent explanation."
            )

        else:

            st.error(
                "❌ Answer does not sufficiently match the question."
            )


    elif similarity < 0.35:

        st.warning(
            "⚠ The answer may not be sufficiently relevant to the question."
        )


    # ========================================================
    # AI ASSESSMENT SUMMARY
    # ========================================================

    st.html(
        """
        <div class="section-title">
            🤖 AI Assessment Insights
        </div>

        <div class="section-subtitle">
            Rubric-based interpretation of the student's response.
        </div>
        """
    )


    summary_items = []


    if similarity >= 0.70:

        summary_items.append(
            "✓ Strong semantic relevance"
        )

    elif similarity >= 0.50:

        summary_items.append(
            "✓ Moderate semantic relevance"
        )

    else:

        summary_items.append(
            "⚠ Limited semantic relevance"
        )


    if language_score >= 1.5:

        summary_items.append(
            "✓ Clear language quality"
        )

    elif language_score >= 1:

        summary_items.append(
            "✓ Adequate language quality"
        )

    else:

        summary_items.append(
            "⚠ Language can be improved"
        )


    if organisation_score >= 2.25:

        summary_items.append(
            "✓ Well-organised response"
        )

    elif organisation_score >= 1.5:

        summary_items.append(
            "✓ Reasonably structured response"
        )

    else:

        summary_items.append(
            "⚠ Structure needs improvement"
        )


    summary_html = ""

    for item in summary_items:

        summary_html += (
            f'<span class="summary-item">'
            f'{html.escape(item)}'
            f'</span>'
        )


    st.html(
        f"""
        <div class="summary-box">

            <div class="summary-title">
                Assessment Summary
            </div>

            {summary_html}

        </div>
        """
    )


    # ========================================================
    # RUBRIC INSIGHTS
    # ========================================================

    r1, r2, r3 = st.columns(
        3,
        gap="medium"
    )


    with r1:

        st.html(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    📚 Content
                </div>

                <div class="insight-score">
                    Score: {content_score}/5
                </div>

                <div class="insight-text">
                    Evaluates relevance, concepts and
                    coverage of the question.
                </div>

            </div>
            """
        )


    with r2:

        st.html(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    ✍️ Language
                </div>

                <div class="insight-score">
                    Score: {language_score}/2
                </div>

                <div class="insight-text">
                    Evaluates clarity, readability and
                    language quality.
                </div>

            </div>
            """
        )


    with r3:

        st.html(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    🧩 Organisation
                </div>

                <div class="insight-score">
                    Score: {organisation_score}/3
                </div>

                <div class="insight-text">
                    Evaluates structure, flow and
                    coherence of the response.
                </div>

            </div>
            """
        )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.html(
        """
        <div class="section-title">
            💬 AI-Generated Explanation
        </div>

        <div class="section-subtitle">
            Detailed reasoning behind the rubric-based assessment.
        </div>
        """
    )


    with st.expander(
        "🔍  View detailed evaluation explanation",
        expanded=True
    ):

        st.write(explanation)


    # ========================================================
    # PERSONALIZED FEEDBACK
    # ========================================================

    st.html(
        """
        <div class="section-title">
            💡 Personalized Feedback
        </div>

        <div class="section-subtitle">
            Suggestions to help improve the student's response.
        </div>
        """
    )


    with st.expander(
        "🚀  View improvement suggestions",
        expanded=True
    ):

        st.write(feedback)


# ============================================================
# AI ASSESSMENT WORKFLOW
# ============================================================

st.html(
    """
    <div class="soft-divider"></div>

    <div class="section-title">
        ⚙️ How the AI Assessment Works
    </div>

    <div class="section-subtitle">
        The system combines NLP preprocessing, semantic analysis,
        syntactic features and machine-learning based scoring.
    </div>
    """
)


st.html(
    """
    <div class="pipeline-container">

        <div class="pipeline-title">
            Assessment Pipeline
        </div>

        <div class="pipeline">

            <div class="pipeline-step">

                <div class="pipeline-icon">
                    📝
                </div>

                <div class="pipeline-name">
                    Student Answer
                </div>

                <div class="pipeline-tech">
                    Input
                </div>

            </div>


            <div class="pipeline-arrow">
                →
            </div>


            <div class="pipeline-step">

                <div class="pipeline-icon">
                    🧹
                </div>

                <div class="pipeline-name">
                    Preprocessing
                </div>

                <div class="pipeline-tech">
                    NLP
                </div>

            </div>


            <div class="pipeline-arrow">
                →
            </div>


            <div class="pipeline-step">

                <div class="pipeline-icon">
                    📐
                </div>

                <div class="pipeline-name">
                    Syntactic Analysis
                </div>

                <div class="pipeline-tech">
                    NLP Features
                </div>

            </div>


            <div class="pipeline-arrow">
                →
            </div>


            <div class="pipeline-step">

                <div class="pipeline-icon">
                    🧠
                </div>

                <div class="pipeline-name">
                    Semantic Analysis
                </div>

                <div class="pipeline-tech">
                    MiniLM
                </div>
                

            </div>


            <div class="pipeline-arrow">
                →
            </div>


            <div class="pipeline-step">

                <div class="pipeline-icon">
                    📊
                </div>

                <div class="pipeline-name">
                    Score Prediction
                </div>

                <div class="pipeline-tech">
                    Ridge Regression
                </div>

            </div>


            <div class="pipeline-arrow">
                →
            </div>


            <div class="pipeline-step">

                <div class="pipeline-icon">
                    🤖
                </div>

                <div class="pipeline-name">
                    AI Feedback
                </div>

                <div class="pipeline-tech">
                    Explanation Module
                </div>

            </div>

        </div>

    </div>
    """
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.html(
    """
    <div class="section-title">
        🛠️ Technology Stack
    </div>

    <div class="section-subtitle">
        Technologies and machine-learning components used in AutoGrade AI.
    </div>
    """
)


st.html(
    """
    <div class="tech-wrapper">

        <span class="tech-badge">
            Python
        </span>

        <span class="tech-badge">
            NLP
        </span>

        <span class="tech-badge">
            MiniLM
        </span>

        <span class="tech-badge">
            Sentence Transformers
        </span>

        <span class="tech-badge">
            Ridge Regression
        </span>

        <span class="tech-badge">
            Machine Learning
        </span>

        <span class="tech-badge">
            Streamlit
        </span>

        <span class="tech-badge">
            AI Evaluation
        </span>

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <strong>
            AutoGrade AI
        </strong>
        · Automatic Answer Assessment System

        <br>

        Semantic Analysis · Syntactic Analysis · ML Scoring · AI Feedback

    </div>
    """
)