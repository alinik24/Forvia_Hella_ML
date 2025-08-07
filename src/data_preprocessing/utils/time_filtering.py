# ---- Preprocessing of materials dataset ----
from datetime import datetime
from pathlib import Path
from typing import List

from dask import dataframe as dd


def create_n_day_dataset(n_days: int, input_path: str, output_path: str):
    """
    Create a dataset with a user-selected `n_days` of calendar days with data.

    Args:
        n_days (int): Number of distinct calendar days to include per group.
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

    # Extract unique calendar days with data
    ddf['date'] = ddf['created_at'].dt.date
    unique_dates = sorted(ddf['date'].dropna().drop_duplicates().compute())

    if len(unique_dates) < n_days:
        raise ValueError(f"Only {len(unique_dates)} days of data available. Need at least {n_days} full days.")

    # Break into week-like chunks
    week_groups: List[List[datetime.date]] = [
        unique_dates[i:i + n_days] for i in range(0, len(unique_dates), n_days)
        if len(unique_dates[i:i + n_days]) == n_days
    ]

    # Show available week groups
    print("\nAvailable day groups:")
    for idx, group in enumerate(week_groups):
        print(f"{idx + 1}: {group[0]} to {group[-1]}")

    # Ask user to select one
    while True:
        try:
            selection = int(input(f"\nSelect a group to extract (1 - {len(week_groups)}): "))
            if 1 <= selection <= len(week_groups):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

    selected_days = set(week_groups[selection - 1])
    print(f"\nUsing dates: {sorted(selected_days)}")

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

    # Format date suffix for file
    start_date_str = min(selected_days).isoformat()
    end_date_str = max(selected_days).isoformat()
    date_suffix = f"{start_date_str}_to_{end_date_str}"

    # Modify output path to include date range
    output_path = Path(output_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    output_ext = output_path.suffix

    dated_output_path = output_dir / f"{base_name}_{date_suffix}{output_ext}"

    # Save with inferred schema
    ddf_filtered.to_parquet(
        dated_output_path,
        engine='pyarrow',
        compression='snappy',
        write_index=False,
        schema='infer'
    )

    print(f"\nSaved filtered data to: {dated_output_path}")


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
