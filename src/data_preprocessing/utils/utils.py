import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def filter_parquet_columns_contains(input_path, output_path, base_names):
    """
    Filter a Parquet file by keeping columns that contain any of the base names.

    Parameters:
        input_path (str): Path to the source Parquet file.
        output_path (str): Path to save the filtered file.
        base_names (list): Base column name substrings to match.
    """
    table = pq.read_table(input_path)
    all_columns = table.column_names

    # Keep columns that contain any of the base names
    keep_columns = [
        col for col in all_columns
        if any(base in col for base in base_names)
    ]

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


def analyze_missingness(df):
    print("\n=== Missingness Indicators Analysis ===")
    nan_cols = [col for col in df.columns if col.endswith('_nan')]
    print(f"Found {len(nan_cols)} missingness indicator columns.")

    total_rows = df.shape[0]

    print("\nMissingness count and percentage per '_nan' column:")
    nan_summary = {}
    for col in nan_cols:
        missing_count = df[col].sum()  # sum of 1s = missing occurrences
        missing_pct = 100 * missing_count / total_rows
        nan_summary[col] = missing_pct
        print(f"{col}: {missing_count} ({missing_pct:.2f}%)")

    # Row-level missingness: how many missing flags per row
    df['missing_indicator_count'] = df[nan_cols].sum(axis=1)
    print("\nRow-level missingness indicator counts:")
    print(df['missing_indicator_count'].describe())

    no_missing = (df['missing_indicator_count'] == 0).sum()
    some_missing = (df['missing_indicator_count'] > 0).sum()
    print(f"\nRows with no missing indicators: {no_missing} ({100 * no_missing / total_rows:.2f}%)")
    print(f"Rows with at least one missing indicator: {some_missing} ({100 * some_missing / total_rows:.2f}%)")

    threshold = 5
    many_missing = df[df['missing_indicator_count'] > threshold]
    print(f"\nRows with more than {threshold} missing indicators: {many_missing.shape[0]}")

    # Uncomment to drop rows with too many missing indicators:
    # df_cleaned = df[df['missing_indicator_count'] <= threshold].copy()
    # print(f"Dropped {many_missing.shape[0]} rows with >{threshold} missing indicators.")
    # return df_cleaned

    return df


def drop_highly_correlated_columns_pandas(df: pd.DataFrame, threshold: float = 0.95,
                                          sample_size: int = 10000) -> pd.DataFrame:
    print(f"\n[INFO] Sampling up to {sample_size} rows for correlation analysis...")
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    corr_matrix = sample_df.corr(method='spearman').abs()

    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = corr_matrix.where(upper)

    to_drop = set()
    print(f"\n[INFO] Columns with correlation > {threshold}:")
    for col in upper_matrix.columns:
        high_corr_pairs = upper_matrix[col][upper_matrix[col] > threshold]
        for idx, corr_value in high_corr_pairs.items():
            print(f"  - {col} and {idx} | correlation = {corr_value:.4f}")
            to_drop.add(idx)

    print(f"\n[INFO] Total columns to drop: {len(to_drop)}")
    if to_drop:
        print("Dropped columns:", ", ".join(to_drop))
    else:
        print("No columns exceed the correlation threshold.")

    return df.drop(columns=list(to_drop))


def drop_highly_correlated_columns_dask(dask_df: dd.DataFrame, threshold: float = 0.95,
                                        sample_size: int = 10000) -> dd.DataFrame:
    print(f"\n[INFO] Sampling up to {sample_size} rows for correlation analysis...")
    sampled_df = dask_df.sample(frac=min(1.0, sample_size / len(dask_df))).compute()
    corr_matrix = sampled_df.corr(method='pearson').abs()

    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = corr_matrix.where(upper)

    to_drop = set()
    print(f"\n[INFO] Columns with correlation > {threshold}:")
    for col in upper_matrix.columns:
        high_corr_pairs = upper_matrix[col][upper_matrix[col] > threshold]
        for idx, corr_value in high_corr_pairs.items():
            print(f"  - {col} and {idx} | correlation = {corr_value:.4f}")
            to_drop.add(idx)

    print(f"\n[INFO] Total columns to drop: {len(to_drop)}")
    if to_drop:
        print("Dropped columns:", ", ".join(to_drop))
    else:
        print("No columns exceed the correlation threshold.")

    return dask_df.drop(columns=list(to_drop))
