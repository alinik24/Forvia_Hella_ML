# ---- Preprocessing of materials dataset ----
from dask import dataframe as dd


def create_n_day_dataset(n_days: int, input_path: str, output_path: str):
    """
    Create a dataset with exactly `n_days` of calendar days with data.

    Args:
        n_days (int): Number of distinct calendar days to include (with data).
        input_path (str): Path to input Parquet dataset.
        output_path (str): Path to save the filtered Parquet dataset.
    """
    ddf = dd.read_parquet(input_path, engine='pyarrow')
    ddf = ddf.assign(created_at=dd.to_datetime(ddf['created_at'], format='ISO8601', utc=True))

    # Check for null datetimes
    null_count = ddf['created_at'].isnull().sum().compute()
    total_rows = len(ddf)
    if null_count > 0:
        print(
            f"Warning: {null_count}/{total_rows:,} rows ({null_count / total_rows:.2%}) failed datetime conversion")
        bad_rows = ddf[ddf['created_at'].isnull()].head(5).compute()
        print("Sample problematic rows:", bad_rows)

    # Extract distinct dates with data
    ddf['date'] = ddf['created_at'].dt.date
    unique_dates = ddf['date'].dropna().drop_duplicates().compute()
    unique_dates = sorted(unique_dates)

    if len(unique_dates) < n_days:
        raise ValueError(f"Only {len(unique_dates)} days of data available. Need at least {n_days} full days.")

    # Pick the last N days that actually have data
    selected_days = set(unique_dates[-n_days:])
    print(f"Using dates: {sorted(selected_days)}")

    # Filter down to selected calendar days
    ddf_filtered = ddf[ddf['date'].isin(selected_days)]

    # Convert 'has_failures' to string if it exists and is boolean
    if 'has_failures' in ddf_filtered.columns:
        ddf_filtered['has_failures'] = ddf_filtered['has_failures'].astype(str)
        print(f"Converted 'has_failures' dtype to: {ddf_filtered['has_failures'].dtype}")

    # Get timing info
    min_time = ddf_filtered['created_at'].min().compute()
    max_time = ddf_filtered['created_at'].max().compute()
    unique_day_count = ddf_filtered['date'].nunique().compute()

    print(f"--- Final {n_days}-day dataset ---")
    print(f"Rows             : {len(ddf_filtered)}")
    print(f"Min Timestamp    : {min_time}")
    print(f"Max Timestamp    : {max_time}")
    print(f"Calendar Span    : {(max_time - min_time).days} days")
    print(f"Days With Data   : {unique_day_count} (should be {n_days})")

    # Drop helper column
    ddf_filtered = ddf_filtered.drop(columns=["date"])

    # Save with inferred schema
    ddf_filtered.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        write_index=False,
        schema='infer'
    )


def analyze_and_transform_data(input_path: str):
    """Load, transform, and analyze the timeframe of a dataset"""
    # Load data with memory optimization
    ddf = dd.read_parquet(
        input_path,
        engine='pyarrow',
    )

    # Convert to proper datetime with ISO format
    ddf = ddf.assign(
        created_at=dd.to_datetime(
            ddf['created_at'],
            format='ISO8601',
            utc=True,
            errors='coerce'
        )
    )

    # Compute actual timestamp span
    min_time = ddf['created_at'].min().compute()
    max_time = ddf['created_at'].max().compute()

    # Compute number of unique days
    ddf['date'] = ddf['created_at'].dt.date
    num_unique_days = ddf['date'].nunique().compute()

    print(f"--- Dataset Time Span ---")
    print(f"Rows             : {len(ddf)}")
    print(f"Min Timestamp    : {min_time}")
    print(f"Max Timestamp    : {max_time}")
    print(f"Calendar Span    : {(max_time - min_time).days} days")
    print(f"Days With Data   : {num_unique_days}")
