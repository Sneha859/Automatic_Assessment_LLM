import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def prepare_features(df):

    syntactic_features = [
        "avg_sentence_length",
        "noun_ratio",
        "verb_ratio",
        "adj_ratio",
        "readability",
        "qa_similarity"
    ]

    embedding_features = [c for c in df.columns if c.startswith("embed_")]

    feature_columns = syntactic_features + embedding_features

    return df[feature_columns]



def generate_rubric_scores(final_score):

    content = final_score * 0.5
    language = final_score * 0.2
    organisation = final_score * 0.3

    return round(content,2), round(language,2), round(organisation,2)


# -------------------------------------------------
# Prediction Formatter
# -------------------------------------------------

def build_output_dataframe(original_df, predictions):

    # Convert to /10 scale
    final_scores = (predictions / 6) * 10

    content_scores = []
    language_scores = []
    organisation_scores = []

    for score in final_scores:

        c, l, o = generate_rubric_scores(score)

        content_scores.append(c)
        language_scores.append(l)
        organisation_scores.append(o)

    # Human score also scaled to /10
    human_scaled = (original_df["human_score"] / 6) * 10

    output_df = pd.DataFrame({

        "essay_question": original_df["essay_question"],
        "essay_answer": original_df["essay_answer"],

        "human_score": human_scaled.round(2),
        "predicted_score": final_scores.round(2),

        "content_score": content_scores,
        "language_score": language_scores,
        "organisation_score": organisation_scores
    })

    return output_df


# -------------------------------------------------
# Main Training
# -------------------------------------------------

def train_model():

    print("\n[Scoring Module] Loading datasets...")

    train_df = pd.read_csv("data/processed/train_semantic.csv")
    test_df = pd.read_csv("data/processed/test_semantic.csv")

    print("[Scoring Module] Preparing features...")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    y_train = train_df["human_score"]
    y_test = test_df["human_score"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[Scoring Module] Training Ridge Regression...")

    model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)

    print("[Scoring Module] Training completed")

    print("[Scoring Module] Predicting train scores...")

    train_predictions = model.predict(X_train)
    train_predictions = np.clip(train_predictions, 1, 6)

    print("[Scoring Module] Predicting test scores...")

    test_predictions = model.predict(X_test)
    test_predictions = np.clip(test_predictions, 1, 6)

    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    print(f"[Scoring Module] Test RMSE: {rmse:.4f}")

    # -------------------------------------------------
    # Build Output DataFrames
    # -------------------------------------------------

    train_output = build_output_dataframe(train_df, train_predictions)
    test_output = build_output_dataframe(test_df, test_predictions)

    # -------------------------------------------------
    # Save Outputs
    # -------------------------------------------------

    os.makedirs("data/outputs", exist_ok=True)

    train_output_path = "data/outputs/train_predictions.csv"
    test_output_path = "data/outputs/test_predictions.csv"

    train_output.to_csv(train_output_path, index=False)
    test_output.to_csv(test_output_path, index=False)

    print(f"[Scoring Module] Train predictions saved → {train_output_path}")
    print(f"[Scoring Module] Test predictions saved → {test_output_path}")

    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/ridge_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("[Scoring Module] Model saved → models/ridge_model.pkl")

    print("\n[Scoring Module] Essay scoring completed successfully\n")


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":

    train_model()