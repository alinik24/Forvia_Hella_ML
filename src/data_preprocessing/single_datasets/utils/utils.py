# single_datasets/utils.py

import os
import pyarrow.parquet as pq


def filter_parquet_columns(input_path, output_path, keep_columns):
    """
    Filter a Parquet file by keeping only the specified columns.

    Parameters:
        input_path (str): Path to the source Parquet file.
        output_path (str): Path to save the filtered file.
        keep_columns (list): Columns to retain in the output file.
    """
    table = pq.read_table(input_path)
    filtered_table = table.select(keep_columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(filtered_table, output_path)
    print(f"Filtered dataset saved to: {output_path}")
