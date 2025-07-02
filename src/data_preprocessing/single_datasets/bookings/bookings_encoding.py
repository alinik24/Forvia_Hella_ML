# ---- Preprocessing of bookings dataset ----
import os

from src.data_preprocessing.single_datasets.utils.preprocessing_utils import load_parquet_dataset, preprocess_data

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "data_hella_single_line", "bookings_single_line.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "bookings_encoded_data.parquet")


def main():
    print("Loading data...")
    df = load_parquet_dataset(INPUT_FILE)

    print("Preprocessing data...")
    y, feature_names, preprocessed_df = preprocess_data(
        df,
        target_column='book_state',
        datetime_cols=['created_at', 'updated_at', 'book_stamp'],
        boolean_cols=['has_failures'],
        high_card_threshold=50,
        vectorizer_path='bookings_vectorizer.pkl'
    )

    print("Saving preprocessed data...")
    preprocessed_df['target'] = y.values  # Add target back in
    preprocessed_df.to_parquet(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()
