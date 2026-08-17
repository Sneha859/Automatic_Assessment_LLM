import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset():

    print("\n[Dataset Module] Loading dataset...")

    dataset_path = "data/raw/aes_dataset.csv"
    df = pd.read_csv(dataset_path)

    print(f"Dataset loaded successfully with shape: {df.shape}")

    # Keep only required columns
    print("[Dataset Module] Removing unnecessary metadata columns...")

    df = df[['assignment', 'full_text', 'score']]

    # Rename columns
    df.rename(columns={
        'assignment': 'essay_question',
        'full_text': 'essay_answer',
        'score': 'human_score'
    }, inplace=True)

    print("[Dataset Module] Columns renamed successfully")

    # Remove empty rows
    df.dropna(inplace=True)

    print(f"[Dataset Module] Dataset after cleaning: {df.shape}")

    # Train Test Split
    print("[Dataset Module] Splitting dataset into train and test (80/20)...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    print(f"Train size: {train_df.shape}")
    print(f"Test size: {test_df.shape}")

    # Save files
    os.makedirs("data/processed", exist_ok=True)

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n[Dataset Module] Files saved successfully:")
    print(train_path)
    print(test_path)

    print("\n[Dataset Module] Dataset preparation completed\n")


if __name__ == "__main__":
    split_dataset()