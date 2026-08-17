import pandas as pd
import spacy
import textstat
import os
from tqdm import tqdm

print("\n[Syntactic Analyzer] Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("[Syntactic Analyzer] Model loaded successfully")


def extract_syntactic_features(text):

    doc = nlp(text)

    num_sentences = len(list(doc.sents))
    num_tokens = len(doc)

    if num_sentences == 0:
        avg_sentence_length = 0
    else:
        avg_sentence_length = num_tokens / num_sentences

    noun_count = 0
    verb_count = 0
    adj_count = 0

    for token in doc:

        if token.pos_ == "NOUN":
            noun_count += 1

        elif token.pos_ == "VERB":
            verb_count += 1

        elif token.pos_ == "ADJ":
            adj_count += 1

    if num_tokens == 0:
        noun_ratio = 0
        verb_ratio = 0
        adj_ratio = 0
    else:
        noun_ratio = noun_count / num_tokens
        verb_ratio = verb_count / num_tokens
        adj_ratio = adj_count / num_tokens

    readability = textstat.flesch_reading_ease(text)

    return avg_sentence_length, noun_ratio, verb_ratio, adj_ratio, readability


def run_syntactic_analysis(input_path, output_path):

    print(f"\n[Syntactic Analyzer] Loading dataset: {input_path}")

    df = pd.read_csv(input_path)

    print("[Syntactic Analyzer] Extracting syntactic features...")

    tqdm.pandas()

    features = df["processed_answer"].progress_apply(extract_syntactic_features)

    df["avg_sentence_length"] = features.apply(lambda x: x[0])
    df["noun_ratio"] = features.apply(lambda x: x[1])
    df["verb_ratio"] = features.apply(lambda x: x[2])
    df["adj_ratio"] = features.apply(lambda x: x[3])
    df["readability"] = features.apply(lambda x: x[4])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"[Syntactic Analyzer] Features saved to: {output_path}")


def main():

    train_input = "data/processed/train_preprocessed.csv"
    test_input = "data/processed/test_preprocessed.csv"

    train_output = "data/processed/train_syntax.csv"
    test_output = "data/processed/test_syntax.csv"

    run_syntactic_analysis(train_input, train_output)
    run_syntactic_analysis(test_input, test_output)

    print("\n[Syntactic Analyzer] Syntactic analysis completed\n")


if __name__ == "__main__":
    main()