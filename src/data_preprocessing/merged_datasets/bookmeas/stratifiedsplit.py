# main.py

import os
from pathlib import Path

import dask.dataframe as dd

from src.data_preprocessing.utils.stratifiedsplit_utility import stratified_split_and_save


def main():
    """
    Main function to handle data loading, stratified splitting, and saving.
    """
    # Prompt user for input and output paths
    input_path = input("Enter path to input Parquet file: ").strip()
    output_dir = input("Enter output directory path: ").strip()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("Loading data...")
    try:
        ddf = dd.read_parquet(input_path, engine='pyarrow')
        print(f"Loaded data with shape: {len(ddf):,} rows, {len(ddf.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract original file name without extension for output file naming
    original_filename = Path(input_path).stem

    # Perform the stratified split and save the files
    stratified_split_and_save(
        ddf=ddf,
        output_dir=output_dir,
        target_column='book_state',
        time_column='created_at',
        test_ratio=0.1,
        original_filename=original_filename
    )


if __name__ == "__main__":
    main()
