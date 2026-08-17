import pandas as pd
import nltk
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


def preprocess_text(text, stop_words, lemmatizer):
    """
    Perform basic NLP preprocessing
    """

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(str(text))

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join back to sentence
    cleaned_text = " ".join(tokens)

    return cleaned_text


def preprocess_dataset(input_path, output_path):

    print("\n[Preprocessing Module] Loading dataset...")

    df = pd.read_csv(input_path)

    print(f"[Preprocessing Module] Dataset shape: {df.shape}")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    print("[Preprocessing Module] Starting text preprocessing...")

    tqdm.pandas()

    df["processed_answer"] = df["essay_answer"].progress_apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )

    print("[Preprocessing Module] Text preprocessing completed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"[Preprocessing Module] Processed file saved at: {output_path}")


def run_preprocessing():

    train_input = "data/processed/train.csv"
    test_input = "data/processed/test.csv"

    train_output = "data/processed/train_preprocessed.csv"
    test_output = "data/processed/test_preprocessed.csv"

    preprocess_dataset(train_input, train_output)
    preprocess_dataset(test_input, test_output)

    print("\n[Preprocessing Module] Preprocessing completed successfully\n")


if __name__ == "__main__":
    run_preprocessing()