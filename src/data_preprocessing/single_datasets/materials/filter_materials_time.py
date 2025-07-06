# ---- Preprocessing of materials dataset ----
import os
from datetime import datetime, timedelta

from dask import dataframe as dd

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "materials", "materials_4_weeks.parquet")
OUTPUT_FILE_1_WEEK = os.path.join(DATA_DIR, "materials", "materials_1_week.parquet")
OUTPUT_FILE_2_WEEKS = os.path.join(DATA_DIR, "materials", "materials_2_weeks.parquet")


def create_2_week_dataset():
    """Create dataset with exactly 14 calendar days that actually have data."""
    ddf = dd.read_parquet(INPUT_FILE, engine='pyarrow')
    ddf = ddf.assign(created_at=dd.to_datetime(ddf['created_at'], format='ISO8601', utc=True))

    # Check for null datetimes
    null_count = ddf['created_at'].isnull().sum().compute()
    total_rows = len(ddf)
    if null_count > 0:
        print(
            f"⚠️ Warning: {null_count}/{total_rows:,} rows ({null_count / total_rows:.2%}) failed datetime conversion")
        bad_rows = ddf[ddf['created_at'].isnull()].head(5).compute()
        print("Sample problematic rows:", bad_rows)

    # Extract distinct dates with data
    ddf['date'] = ddf['created_at'].dt.date
    unique_dates = ddf['date'].dropna().drop_duplicates().compute()
    unique_dates = sorted(unique_dates)

    if len(unique_dates) < 14:
        raise ValueError(f"Only {len(unique_dates)} days of data available. Need at least 7 full days.")

    # Get the last 14 dates with data
    last_14_days = set(unique_dates[-14:])
    print("Using dates:", sorted(last_14_days))

    # Filter to only those days
    ddf = ddf[ddf['date'].isin(last_14_days)].drop(columns=['date'])

    # Recompute actual timestamp span
    min_time = ddf['created_at'].min().compute()
    max_time = ddf['created_at'].max().compute()

    # Compute number of unique days
    ddf['date'] = ddf['created_at'].dt.date
    num_unique_days = ddf['date'].nunique().compute()

    print(f"--- Final 1-week dataset ---")
    print(f"Rows             : {len(ddf)}")
    print(f"Min Timestamp    : {min_time}")
    print(f"Max Timestamp    : {max_time}")
    print(f"Calendar Span    : {(max_time - min_time).days} days")
    print(f"Days With Data   : {num_unique_days} (should be 14)")

    # Save
    ddf.drop(columns=["date"]).to_parquet(
        OUTPUT_FILE_1_WEEK,
        engine='pyarrow',
        compression='snappy',
        write_index=False
    )


def create_1_week_dataset():
    """Create dataset with exactly 7 calendar days that actually have data."""
    ddf = dd.read_parquet(INPUT_FILE, engine='pyarrow')
    ddf = ddf.assign(created_at=dd.to_datetime(ddf['created_at'], format='ISO8601', utc=True))

    # Check for null datetimes
    null_count = ddf['created_at'].isnull().sum().compute()
    total_rows = len(ddf)
    if null_count > 0:
        print(
            f"⚠️ Warning: {null_count}/{total_rows:,} rows ({null_count / total_rows:.2%}) failed datetime conversion")
        bad_rows = ddf[ddf['created_at'].isnull()].head(5).compute()
        print("Sample problematic rows:", bad_rows)

    # Extract distinct dates with data
    ddf['date'] = ddf['created_at'].dt.date
    unique_dates = ddf['date'].dropna().drop_duplicates().compute()
    unique_dates = sorted(unique_dates)

    if len(unique_dates) < 7:
        raise ValueError(f"Only {len(unique_dates)} days of data available. Need at least 7 full days.")

    # Get the last 7 dates with data
    last_7_days = set(unique_dates[-7:])
    print("Using dates:", sorted(last_7_days))

    # Filter to only those days
    ddf = ddf[ddf['date'].isin(last_7_days)].drop(columns=['date'])

    # Recompute actual timestamp span
    min_time = ddf['created_at'].min().compute()
    max_time = ddf['created_at'].max().compute()

    # Compute number of unique days
    ddf['date'] = ddf['created_at'].dt.date
    num_unique_days = ddf['date'].nunique().compute()

    print(f"--- Final 1-week dataset ---")
    print(f"Rows             : {len(ddf)}")
    print(f"Min Timestamp    : {min_time}")
    print(f"Max Timestamp    : {max_time}")
    print(f"Calendar Span    : {(max_time - min_time).days} days")
    print(f"Days With Data   : {num_unique_days} (should be 7)")

    # Save
    ddf.drop(columns=["date"]).to_parquet(
        OUTPUT_FILE_1_WEEK,
        engine='pyarrow',
        compression='snappy',
        write_index=False
    )


def analyze_and_transform_data():
    """Load, transform, and analyze the original dataset"""
    # Load data with memory optimization
    ddf = dd.read_parquet(
        INPUT_FILE,
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

    print(f"--- Final 1-week dataset ---")
    print(f"Rows             : {len(ddf)}")
    print(f"Min Timestamp    : {min_time}")
    print(f"Max Timestamp    : {max_time}")
    print(f"Calendar Span    : {(max_time - min_time).days} days")
    print(f"Days With Data   : {num_unique_days} (should be 7)")


if __name__ == "__main__":
    #analyze_and_transform_data()
    #create_2_week_dataset()
    create_1_week_dataset()
