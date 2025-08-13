"""
Script: filter_and_sample_non_label_stations.py

Purpose:
--------
This script scans a large `measurements.parquet` dataset to identify and extract rows from *non-label stations*
within a specified date range. It processes the data in batches for efficiency and writes two main outputs:
a filtered Parquet sample and CSV summaries.

Main Steps:
-----------
1. **Scan All Rows**:
   - Extract all unique `station_id`s.
   - Determine the earliest and latest `created_at` timestamps in the dataset.

2. **Validate Requested Time Period**:
   - Ensure the hardcoded date range (`start_date` to `end_date`) is within the bounds of available data.

3. **Filter for Non-Label Stations**:
   - Filter rows belonging to stations not listed in the `label_stations`.
   - Further filter those rows to only include timestamps within the specified date range.

4. **Save Outputs**:
   - `non_label_sample_measurements.parquet`: Filtered Parquet file containing relevant rows.
   - `non_label_stations_summary.csv`: CSV summarizing how many rows were extracted for each non-label station.
   - `final_processing_summary.csv`: Metadata about processing time, row counts, and summary stats.

Parameters:
-----------
- `label_stations`: List of known label-producing stations (excluded from output).
- `start_date`, `end_date`: Hardcoded time filter window (must be within dataset’s bounds).
- `batch_size`: Controls how many rows are read per batch from the Parquet file (adjustable based on memory).
- `created_at`: Assumed timestamp column used for temporal single_datasets.

Usage Notes:
------------
- Reuses cached filtered Parquet file if it already exists.
- Includes ETA estimates while scanning and single_datasets.
- Designed for SMT/production telemetry datasets with millions of records.

Output Files:
-------------
- `non_label_sample_measurements.parquet`
- `non_label_stations_summary.csv`
- `final_processing_summary.csv`
"""
import gc
import os
import time
from datetime import timedelta

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# Define paths
base_path = 'C:/Users/alina/Downloads/data_hella/'
input_file = os.path.join(base_path, 'measurements.parquet')
output_dir = os.path.join(base_path, 'station_outputs_full')
sample_parquet = os.path.join(output_dir, 'non_label_sample_measurements.parquet')
non_label_summary_csv = os.path.join(output_dir, 'non_label_stations_summary.csv')
final_summary_csv = os.path.join(output_dir, 'final_processing_summary.csv')
os.makedirs(output_dir, exist_ok=True)

# Define label stations (used only for single_datasets)
label_stations = [
    'b17ad0f9', '5a3f0928', '031c4441', '14473147', 'eef8a574',
    '9a991014', '0bde46ac', '4b1b68dd', '507926e9', 'c06ba294',
    'e40d07f7', '0d84218d', '94ff9bdd', 'a24958aa', '0883123a',
    'b94b00ce', '8ce235cf', '63afba48', 'bc01f8be', 'd6806593', 'de579af6'
]

# Parameters
batch_size = 100000  # Adjust based on RAM
timestamp_column = 'created_at'

# Hardcoded date range
start_date = pd.to_datetime('2024-06-01', utc=False)
end_date = pd.to_datetime('2024-07-01', utc=False)
requested_period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

# Step 1: Extract unique station_ids and date range
print("Extracting unique station_ids and date range from measurements.parquet...")
parquet_file = pq.ParquetFile(input_file)
station_ids = set()
min_date = None
max_date = None

# Progress bar for batch processing
total_rows = parquet_file.metadata.num_rows
num_batches = (total_rows + batch_size - 1) // batch_size

start_time = time.time()
batch_times = []

with tqdm(total=total_rows, desc="Scanning batches", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_start = time.time()
        df = batch.to_pandas()
        station_ids.update(df['station_id'].unique())

        # Process timestamps: Convert UTC timezone-aware to timezone-naive
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce', utc=True).dt.tz_localize(None)
        batch_min = df[timestamp_column].min()
        batch_max = df[timestamp_column].max()

        if min_date is None or (batch_min is not pd.NaT and batch_min < min_date):
            min_date = batch_min
        if max_date is None or (batch_max is not pd.NaT and batch_max > max_date):
            max_date = batch_max

        # Update progress and estimate time
        pbar.update(len(df))
        batch_times.append(time.time() - batch_start)

        # Estimate remaining time every 10 batches
        if len(batch_times) % 10 == 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = num_batches - len(batch_times)
            eta_seconds = avg_batch_time * remaining_batches
            eta = timedelta(seconds=int(eta_seconds))
            pbar.set_postfix({"ETA": str(eta)})

        del df, batch
        gc.collect()

if min_date is None or max_date is None:
    print("Error: Could not determine date range. Check timestamp column format.")
    exit()

max_period = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
total_days = (max_date - min_date).days
print(f"\nMaximum available data period: {max_period}")
print(f"Total duration: {total_days} days")

# Validate hardcoded dates
min_date_naive = min_date
max_date_naive = max_date
start_date_date = start_date.normalize()
end_date_date = end_date.normalize()
min_date_date = min_date_naive.normalize()
max_date_date = max_date_naive.normalize()

if not (min_date_date <= start_date_date <= max_date_date):
    print(
        f"Error: Start date {start_date.strftime('%Y-%m-%d')} is outside the range {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
    exit()
if not (min_date_date <= end_date_date <= max_date_date):
    print(
        f"Error: End date {end_date.strftime('%Y-%m-%d')} is outside the range {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
    exit()
if start_date_date > end_date_date:
    print(f"Error: Start date {start_date.strftime('%Y-%m-%d')} is after end date {end_date.strftime('%Y-%m-%d')}.")
    exit()

# Step 2: Identify non-label stations
non_label_stations = [sid for sid in station_ids if sid not in label_stations]
if not non_label_stations:
    print("Error: No non-label stations found in measurements.parquet.")
    print("Please verify station_id values or contact project team for metadata.")
    exit()

print(f"Found {len(non_label_stations)} non-label stations: {non_label_stations}")

# Step 3: Filter non-label stations and create temporal sample
if not os.path.exists(sample_parquet):
    print(f"\nCreating sample dataset for non-label stations in period: {requested_period}")
    sample_dfs = []
    non_label_row_counts = {sid: 0 for sid in non_label_stations}

    batch_times = []
    with tqdm(total=total_rows, desc="Filtering rows", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_start = time.time()
            df = batch.to_pandas()

            # Filter non-label stations
            filtered_df = df[df['station_id'].isin(non_label_stations)]

            if not filtered_df.empty:
                # Apply temporal filter: Convert UTC timezone-aware to timezone-naive
                filtered_df[timestamp_column] = pd.to_datetime(filtered_df[timestamp_column], errors='coerce',
                                                               utc=True).dt.tz_localize(None)
                sample_df = filtered_df[
                    (filtered_df[timestamp_column] >= start_date) &
                    (filtered_df[timestamp_column] <= end_date)
                    ]

                if not sample_df.empty:
                    sample_dfs.append(sample_df)
                    # Update row counts for non-label stations
                    for sid in non_label_stations:
                        non_label_row_counts[sid] += len(sample_df[sample_df['station_id'] == sid])
                    pbar.update(len(sample_df))

            # Update ETA
            batch_times.append(time.time() - batch_start)
            if len(batch_times) % 10 == 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = num_batches - len(batch_times)
                eta_seconds = avg_batch_time * remaining_batches
                eta = timedelta(seconds=int(eta_seconds))
                pbar.set_postfix({"ETA": str(eta)})

            del df, filtered_df, sample_df, batch
            gc.collect()

    if sample_dfs:
        sample_data = pd.concat(sample_dfs, ignore_index=True)
        # Preserve original timezone-aware timestamps in output
        sample_data.to_parquet(sample_parquet, index=False)
        print(f"\nSample dataset created: {sample_parquet} ({len(sample_data)} rows)")
        print(f"Unique serial_numbers in sample: {sample_data['serial_number'].nunique()}")
    else:
        print("No data in the specified date range or for non-label stations. Adjust the date range or check data.")
        exit()
else:
    sample_data = pd.read_parquet(sample_parquet)
    non_label_row_counts = {sid: len(sample_data[sample_data['station_id'] == sid]) for sid in non_label_stations}
    print(f"\nUsing existing sample dataset: {sample_parquet} ({len(sample_data)} rows)")
    print(f"Unique serial_numbers in sample: {sample_data['serial_number'].nunique()}")

# Step 4: Save non-label stations summary
non_label_summary_data = [{'station_id': sid, 'num_rows': count} for sid, count in non_label_row_counts.items()]
non_label_summary = pd.DataFrame(non_label_summary_data)
non_label_summary.to_csv(non_label_summary_csv, index=False)

# Step 5: Prepare and save final summary
final_summary_data = [
    {'category': 'Metadata', 'key': 'Maximum Data Period', 'value': max_period},
    {'category': 'Metadata', 'key': 'Requested Period', 'value': requested_period},
    {'category': 'Metadata', 'key': 'Total Processing Time',
     'value': str(timedelta(seconds=int(time.time() - start_time)))},
    {'category': 'Metadata', 'key': 'Sample Rows', 'value': str(len(sample_data))},
    {'category': 'Metadata', 'key': 'Unique Serial Numbers', 'value': str(sample_data['serial_number'].nunique())}
]

# Add non-label stations summary
for _, row in non_label_summary.iterrows():
    final_summary_data.append({
        'category': 'Non-Label Stations',
        'key': row['station_id'],
        'value': str(row['num_rows'])
    })

final_summary = pd.DataFrame(final_summary_data)
final_summary.to_csv(final_summary_csv, index=False)

# Step 6: Print final summary
print(f"\n=== Processing Summary (saved as {final_summary_csv}) ===")
print(f"Maximum available data period: {max_period}")
print(f"Requested sampling period: {requested_period}")
print(f"\nNon-Label Stations Summary (saved as {non_label_summary_csv}):")
print(non_label_summary)
print(f"\nTotal processing time: {timedelta(seconds=int(time.time() - start_time))}")
print(f"Sample dataset rows: {len(sample_data)}")
print(f"Unique serial_numbers in sample: {sample_data['serial_number'].nunique()}")
print(f"\nAll summary information saved to {final_summary_csv}")
print("Processing complete. Review the summary CSV and output Parquet file.")
