# ---- Preprocessing of bookmeas dataset ----
import os

from src.data_preprocessing.utils.preprocessing_utils import load_parquet_dataset, preprocess_data

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "bookmeas_1_week_encoded.parquet")


def main():
    print("Loading data...")
    try:
        df = load_parquet_dataset(INPUT_FILE)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        print("Preprocessing data...")
        y, feature_names, preprocessed_df = preprocess_data(
            df,
            target_column='book_state',
            datetime_cols=['created_at', 'updated_at', 'book_stamp'],
            boolean_cols=['has_failures'],
            vectorizer_path='bookmeas_vectorizer.pkl',
            max_unique_snids=3000
        )
        print(f"Preprocessed data shape: {preprocessed_df.shape}")
    except MemoryError as e:
        print(f"Memory error during preprocessing: {e}")
        print("Try reducing max_unique_snids or using a machine with more RAM")
        return
    except Exception as e:
        print(f"Critical preprocessing error: {e}")
        return

    try:
        print("Saving preprocessed data...")
        # Add target back as a separate column
        preprocessed_df['target'] = y.values

        # Handle large DataFrames with PyArrow
        preprocessed_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
        print(f"Successfully saved preprocessed data to {OUTPUT_FILE}")

        # Additional validation
        print(f"Saved data info: {preprocessed_df.shape[0]} rows, {preprocessed_df.shape[1]} columns")
        print(f"Target distribution: \n{preprocessed_df['target'].value_counts()}")
    except Exception as e:
        print(f"Error saving data: {e}")
        # Fallback to CSV if Parquet fails
        try:
            print("Attempting CSV fallback...")
            preprocessed_df.to_csv(OUTPUT_FILE.replace('.parquet', '.csv'), index=False)
            print(f"Saved as CSV: {OUTPUT_FILE.replace('.parquet', '.csv')}")
        except Exception as fallback_e:
            print(f"CSV fallback failed: {fallback_e}")


if __name__ == '__main__':
    main()
