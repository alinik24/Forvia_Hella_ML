# ---- Preprocessing of measurements dataset ----
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

from src.data_preprocessing.utils.preprocessing_utils import preprocess_data

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "data_hella_single_line", "measurements_single_line.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "measurements_encoded_data.parquet")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
TARGET_COLUMN = "book_state"  # Name of your target variable
DATETIME_COLS = ["created_at", "updated_at"]  # List your datetime columns
BOOLEAN_COLS = []  # List your boolean columns


def main():
    # Configuration
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)

    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Data loaded. Shape: {df.shape}")


    print("Preprocessing data...")
    y, feature_names, X_processed = preprocess_data(
        df=df,
        target_column=TARGET_COLUMN,
        datetime_cols=DATETIME_COLS,
        boolean_cols=BOOLEAN_COLS,
        high_card_threshold=100,
        vectorizer_path=VECTORIZER_PATH,
        max_sample_size=500
    )

    # Combine features and target
    processed_df = pd.concat([y, X_processed], axis=1)

    print(f"Saving preprocessed data to: {output_path}")
    processed_df.to_parquet(output_path)
    print(f"Saved processed data. Shape: {processed_df.shape}")
    print(f"Preprocessing pipeline complete. Final data saved to: {output_path}")


if __name__ == '__main__':
    main()#