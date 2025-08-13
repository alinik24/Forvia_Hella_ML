# ---- Preprocessing of materials dataset ----
from pathlib import Path

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.preprocessing_utils import load_parquet_dataset, preprocess_data


def main():
    print("Loading config...")
    reuse_last = input("Reuse last input/output paths? (y/n): ").strip().lower() == "y"

    if reuse_last:
        paths = load_last_run()
        input_main_path = paths["input_path"]
        output_main_path = paths["output_path"]
    else:
        base_input_dir = input("Enter base input directory: ").strip()
        specific_input_filename = input(
            "Enter specific input filename (e.g. bookmeas_1_week_2025-03-12_to_2025-03-27.parquet): ").strip()
        input_main_path = str(Path(base_input_dir) / specific_input_filename)

        # REVISION: Automatically create the output path with a suffix
        input_path_obj = Path(input_main_path)
        # Extract filename without extension, add suffix, then re-add extension
        output_filename = f"{input_path_obj.stem}_encoded{input_path_obj.suffix}"
        output_main_path = str(input_path_obj.parent / output_filename)
        print(f"Output path automatically set to: {output_main_path}")

    print("Loading data...")
    try:
        df = load_parquet_dataset(str(input_main_path))
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        print("Preprocessing data...")
        y, feature_names, preprocessed_df = preprocess_data(
            df,
            target_column='book_state',
            datetime_cols=['setup_started_at', 'created_at', 'lot_packed_at'],
            boolean_cols=[],
            vectorizer_path='materials_vectorizer.pkl',
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
        preprocessed_df.to_parquet(output_main_path, index=False, engine='pyarrow')
        print(f"Successfully saved preprocessed data to {output_main_path}")

        # Additional validation
        print(f"Saved data info: {preprocessed_df.shape[0]} rows, {preprocessed_df.shape[1]} columns")
        print(f"Target distribution: \n{preprocessed_df['target'].value_counts()}")

        # Save last run info
        save_last_run(input_path=input_main_path, output_path=output_main_path, version="v1")

    except Exception as e:
        print(f"Error saving data: {e}")
        # Fallback to CSV if Parquet fails
        try:
            print("Attempting CSV fallback...")
            preprocessed_df.to_csv(output_main_path.replace('.parquet', '.csv'), index=False)
            print(f"Saved as CSV: {output_main_path.replace('.parquet', '.csv')}")
        except Exception as fallback_e:
            print(f"CSV fallback failed: {fallback_e}")


if __name__ == '__main__':
    main()
