# single_datasets/utils.py

import os
import pyarrow.parquet as pq
import pandas as pd


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


def print_nan_columns_info(df: pd.DataFrame):
    print("\nColumns with actual NaN values:")
    nan_cols = df.columns[df.isna().any()].tolist()
    for col in nan_cols:
        num_missing = df[col].isna().sum()
        print(f" - {col} (missing: {num_missing})")

    print("\nColumns with 'nan' in their name (case-insensitive):")
    name_nan_cols = [col for col in df.columns if 'nan' in col.lower()]
    for col in name_nan_cols:
        print(f" - {col}")


def drop_column_if_exists(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Securely drop a column from a DataFrame if it exists.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): Name of the column to remove.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the column removed if it existed.
    """
    if column_name in df.columns:
        print(f"Dropping column: '{column_name}'")
        return df.drop(columns=[column_name])
    else:
        print(f"Column '{column_name}' does not exist in DataFrame. Skipping.")
        return df
