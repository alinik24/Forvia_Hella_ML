from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.utils import drop_highly_correlated_columns_pandas, \
    drop_highly_correlated_columns_dask


def main(method: str = 'dask'):
    """
    method: 'dask' for Dask + Pearson
            'pandas' for Pandas + Spearman
    """
    print("=== Remove Highly Correlated Columns ===")
    reuse_last = input("Reuse last input/output paths? (y/n): ").strip().lower() == "y"

    if reuse_last:
        paths = load_last_run()
        input_main_path = paths.get("input_path")
        output_main_path = paths.get("output_path")
        if not input_main_path or not output_main_path:
            print("No valid paths found in config. Please enter manually.")
            reuse_last = False


    if not reuse_last:
        input_main_path = input("Enter full path to the dataset (parquet): ").strip()

        # Automatically create the output path with method suffix
        input_path_obj = Path(input_main_path)
        suffix = f"_cleaned_{method}"
        output_filename = f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"
        output_main_path = str(input_path_obj.parent / output_filename)

        print(f"Output path automatically set to: {output_main_path}")

        # Perform correlation cleaning
    try:
        if method == 'pearson':
            print("Using Dask + Pearson correlation...")
            ddf = dd.read_parquet(input_main_path)
            cleaned_ddf = drop_highly_correlated_columns_dask(ddf)
            cleaned_ddf.to_parquet(output_main_path, write_index=False)

        elif method == 'spearman':
            print("Using Pandas + Spearman correlation...")
            df = pd.read_parquet(input_main_path)
            cleaned_df = drop_highly_correlated_columns_pandas(df)
            cleaned_df.to_parquet(output_main_path, index=False)

        else:
            raise ValueError("Invalid method. Use 'pearson' or 'spearman'!")

        print(f"Cleaned dataset saved to: {output_main_path}")

        # Save last run
        save_last_run(input_path=input_main_path, output_path=output_main_path, version="v1")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main(method='spearman')
