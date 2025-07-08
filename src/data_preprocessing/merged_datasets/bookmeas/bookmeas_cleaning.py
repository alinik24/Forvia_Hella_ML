import os

import numpy as np
import pandas as pd
import dask.dataframe as dd

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "bookmeas_1_week_encoded.parquet")
OUTPUT_FILE_PEARSON = os.path.join(DATA_DIR, "bookmeas_cleaned_data_pearson.parquet")
OUTPUT_FILE_SPEARMAN = os.path.join(DATA_DIR, "bookmeas_cleaned_data_spearman.parquet")


def drop_highly_correlated_columns_pandas(df: pd.DataFrame, threshold: float = 0.99, sample_size: int = 10000) -> pd.DataFrame:
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    corr_matrix = sample_df.corr(method='spearman').abs()

    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = corr_matrix.where(upper)

    to_drop = set()
    for col in upper_matrix.columns:
        high_corr = upper_matrix[col][upper_matrix[col] > threshold].index
        to_drop.update(high_corr)

    return df.drop(columns=list(to_drop))


def drop_highly_correlated_columns_dask(dask_df: dd.DataFrame, threshold: float = 0.99, sample_size: int = 10000) -> dd.DataFrame:
    # Sample and bring to memory for correlation computation
    sampled_df = dask_df.sample(frac=min(1.0, sample_size / len(dask_df))).compute()
    corr_matrix = sampled_df.corr(method='pearson').abs()

    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = corr_matrix.where(upper)

    to_drop = set()
    for col in upper_matrix.columns:
        high_corr = upper_matrix[col][upper_matrix[col] > threshold].index
        to_drop.update(high_corr)

    return dask_df.drop(columns=list(to_drop))


def main(method: str = 'dask'):
    """
    method: 'dask' for Dask + Pearson
            'pandas' for Pandas + Spearman
    """
    if method == 'pearson':
        print("Using Dask + Pearson correlation")
        ddf = dd.read_parquet(INPUT_FILE)
        cleaned_ddf = drop_highly_correlated_columns_dask(ddf)
        cleaned_ddf.to_parquet(OUTPUT_FILE_PEARSON, write_index=False)
        print(f"Cleaned data saved to {OUTPUT_FILE_PEARSON}")
    elif method == 'spearman':
        print("Using Pandas + Spearman correlation")
        df = pd.read_parquet(INPUT_FILE)
        cleaned_df = drop_highly_correlated_columns_pandas(df)
        cleaned_df.to_parquet(OUTPUT_FILE_SPEARMAN, index=False)
        print(f"Cleaned data saved to {OUTPUT_FILE_SPEARMAN}")
    else:
        raise ValueError("Invalid method. Use 'pearson' or 'spearman'!")


if __name__ == "__main__":
    main(method='spearman')