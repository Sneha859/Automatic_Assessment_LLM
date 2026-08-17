import pandas as pd
import os
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


print("\n[Semantic Analyzer] Loading MiniLM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"[Semantic Analyzer] Model loaded on {device}")


def compute_embeddings(texts):

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    return embeddings

def compute_similarity(question_embeddings, answer_embeddings):

    similarities = []

    for q, a in zip(question_embeddings, answer_embeddings):

        sim = cosine_similarity([q], [a])[0][0]
        similarities.append(sim)

    return similarities


def run_semantic_analysis(input_path, output_path):

    print(f"\n[Semantic Analyzer] Loading dataset: {input_path}")

    df = pd.read_csv(input_path)

    print("[Semantic Analyzer] Generating question embeddings...")

    question_embeddings = compute_embeddings(df["essay_question"].tolist())

    print("[Semantic Analyzer] Generating answer embeddings...")

    answer_embeddings = compute_embeddings(df["processed_answer"].tolist())

    print("[Semantic Analyzer] Computing similarity scores...")

    similarity_scores = compute_similarity(question_embeddings, answer_embeddings)

    df["qa_similarity"] = similarity_scores

    print("[Semantic Analyzer] Adding embedding features...")

    # Convert embeddings to dataframe (FAST + NO WARNINGS)
    embedding_df = pd.DataFrame(
        answer_embeddings,
        columns=[f"embed_{i}" for i in range(len(answer_embeddings[0]))]
    )

    # Concatenate with original dataframe
    df = pd.concat([df, embedding_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"[Semantic Analyzer] Semantic features saved to: {output_path}")


def main():

    train_input = "data/processed/train_syntax.csv"
    test_input = "data/processed/test_syntax.csv"

    train_output = "data/processed/train_semantic.csv"
    test_output = "data/processed/test_semantic.csv"

    run_semantic_analysis(train_input, train_output)
    run_semantic_analysis(test_input, test_output)

    print("\n[Semantic Analyzer] Semantic analysis completed\n")


if __name__ == "__main__":
    main()