import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Words that should NOT be treated as concepts
instruction_words = {
    "explain","describe","define","discuss",
    "what","why","how","importance",
    "features","advantages","types"
}


# ---------------------------------------------------
# Extract meaningful concepts from question
# ---------------------------------------------------

def extract_keywords(text):

    tokens = word_tokenize(text.lower())

    keywords = [
        word for word in tokens
        if word.isalpha()
        and word not in stop_words
        and word not in instruction_words
    ]

    return list(set(keywords))


# ---------------------------------------------------
# Measure concept coverage
# ---------------------------------------------------

def keyword_coverage(question_keywords, answer):

    answer_tokens = set(word_tokenize(answer.lower()))

    covered = [k for k in question_keywords if k in answer_tokens]

    missing = [k for k in question_keywords if k not in answer_tokens]

    ratio = len(covered) / max(len(question_keywords), 1)

    return ratio, covered, missing


# ---------------------------------------------------
# Generate explanation + feedback
# ---------------------------------------------------

def generate_explanation(
        question,
        answer,
        similarity,
        content_score,
        language_score,
        organisation_score
):

    explanation_parts = []
    feedback_parts = []

    words = answer.split()
    sentences = sent_tokenize(answer)

    answer_length = len(words)
    sentence_count = len(sentences)

    # ---------------------------------------
    # Concept extraction
    # ---------------------------------------

    question_keywords = extract_keywords(question)

    coverage_ratio, covered, missing = keyword_coverage(
        question_keywords,
        answer
    )

    # ---------------------------------------
    # CONTENT EXPLANATION
    # ---------------------------------------

    if similarity < 0.35:

        explanation_parts.append(
            "Content Evaluation: The answer is not relevant to the topic asked in the question."
        )

        feedback_parts.append(
            "Focus on answering the concept mentioned in the question instead of discussing unrelated ideas."
        )

    elif similarity < 0.60:

        explanation_parts.append(
            "Content Evaluation: The answer partially addresses the question but does not fully explain all important aspects."
        )

        if missing:
            feedback_parts.append(
                f"Include discussion about concepts such as {', '.join(missing[:2])} to better address the question."
            )

    else:

        if covered:
            explanation_parts.append(
                f"Content Evaluation: The answer correctly discusses key concepts such as {', '.join(covered[:3])}, showing good relevance to the question."
            )
        else:
            explanation_parts.append(
                "Content Evaluation: The answer is relevant but could include more direct discussion of the main concepts in the question."
            )

        if missing:
            feedback_parts.append(
                f"To improve the answer further, consider explaining additional aspects such as {', '.join(missing[:2])}."
            )

    # ---------------------------------------
    # LANGUAGE EXPLANATION
    # ---------------------------------------

    if language_score < 0.8:

        explanation_parts.append(
            "Language Evaluation: Some sentences may contain grammatical issues or unclear phrasing."
        )

        feedback_parts.append(
            "Improve sentence clarity and grammar to make the explanation easier to read."
        )

    else:

        explanation_parts.append(
            "Language Evaluation: The answer uses clear and understandable language."
        )

    # ---------------------------------------
    # ORGANISATION EXPLANATION
    # ---------------------------------------

    if organisation_score < 1.2:

        explanation_parts.append(
            "Organisation Evaluation: The ideas are present but the structure of the explanation could be clearer."
        )

        feedback_parts.append(
            "Organize the explanation using clearer paragraphs or logical steps."
        )

    else:

        explanation_parts.append(
            "Organisation Evaluation: The answer presents ideas in a reasonably organized way."
        )

    # ---------------------------------------
    # LENGTH ANALYSIS
    # ---------------------------------------

    if answer_length < 40:

        feedback_parts.append(
            "The answer is quite short. Adding more explanation or examples would improve the score."
        )

    elif answer_length > 220:

        feedback_parts.append(
            "The answer is lengthy. Try summarizing the key points more concisely."
        )

    # ---------------------------------------
    # STRUCTURE ANALYSIS
    # ---------------------------------------

    if sentence_count <= 2:

        feedback_parts.append(
            "Use multiple sentences or points to explain the concept more clearly."
        )

    # ---------------------------------------
# FINAL OUTPUT
# ---------------------------------------

    explanation = " | ".join(explanation_parts)

    if not feedback_parts:
        feedback_parts.append(
            "The answer is generally good. Adding more examples or deeper explanation could further strengthen the response."
        )
    
    

    feedback = " ".join(feedback_parts)

    return explanation, feedback