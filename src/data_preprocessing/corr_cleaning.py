import dask.dataframe as dd
import pandas as pd

from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_1w, OUTPUT_BOOKMEAS_FILE_PEARSON, \
    OUTPUT_BOOKMEAS_FILE_SPEARMAN
from src.data_preprocessing.utils.utils import drop_highly_correlated_columns_pandas, \
    drop_highly_correlated_columns_dask


def main(method: str = 'dask'):
    """
    method: 'dask' for Dask + Pearson
            'pandas' for Pandas + Spearman
    """
    if method == 'pearson':
        print("Using Dask + Pearson correlation")
        ddf = dd.read_parquet(INPUT_FILE_BOOKMEAS_1w)
        cleaned_ddf = drop_highly_correlated_columns_dask(ddf)
        cleaned_ddf.to_parquet(OUTPUT_BOOKMEAS_FILE_PEARSON, write_index=False)
        print(f"Cleaned data saved to {OUTPUT_BOOKMEAS_FILE_PEARSON}")
    elif method == 'spearman':
        print("Using Pandas + Spearman correlation")
        df = pd.read_parquet(INPUT_FILE_BOOKMEAS_1w)
        cleaned_df = drop_highly_correlated_columns_pandas(df)
        cleaned_df.to_parquet(OUTPUT_BOOKMEAS_FILE_SPEARMAN, index=False)
        print(f"Cleaned data saved to {OUTPUT_BOOKMEAS_FILE_SPEARMAN}")
    else:
        raise ValueError("Invalid method. Use 'pearson' or 'spearman'!")


if __name__ == "__main__":
    main(method='spearman')
