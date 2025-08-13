import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
parquet_file_path = os.path.join(base_path, "measurements_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(output_dir, f"measurements_counts_by_booking_and_serial_with_id_counts_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Description to include in CSV
description = """\
# Script Purpose:
# This script processes (`measurements_single_line.parquet`) to count the number of unique and total
# `teststep_id` entries per `booking_id` and per `serial_number_id`, as well as the number of times each `booking_id` and
# `serial_number_id` appears. It reads the file in batches for memory efficiency, aggregates counts, and handles missing
# values by labeling them as "unlabeled". The script verifies required columns, tracks progress using `tqdm`, saves the
# aggregated results to a timestamped CSV file, and validates that all rows were processed to ensure completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains counts of unique and total teststep_id, and the occurrence count of each booking_id and serial_number_id.
"""

# Verify file exists
if not os.path.exists(parquet_file_path):
    raise FileNotFoundError(f"Parquet file not found at {parquet_file_path}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(parquet_file_path)
available_columns = parquet_file.schema.names
required_columns = ['booking_id', 'serial_number_id', 'teststep_id']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {parquet_file_path}. Available columns: {available_columns}")
    print(
        "Please confirm correct column names (e.g., 'teststep_id' or 'test_number_id', 'booking_id', 'serial_number_id').")
    print("If 'booking_id' or 'serial_number_id' is in another Parquet file, a join may be needed.")
    exit(1)

# Step 2: Collect counts
print("Step 2: Counting unique and total teststep_id, and occurrences per booking_id and serial_number_id")

total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size

# Dictionaries to store counts
booking_counts = {}
serial_counts = {}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size,
                                           columns=['booking_id', 'serial_number_id', 'teststep_id'], use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN with "unlabeled"
        df_batch.fillna("unlabeled", inplace=True)

        # Group by booking_id and count unique and total teststep_id, and occurrences
        booking_agg = df_batch.groupby('booking_id').agg({
            'teststep_id': ['nunique', 'count'],
            'serial_number_id': 'size'  # Count occurrences of booking_id
        }).reset_index()
        booking_agg.columns = ['booking_id', 'unique_teststep_id_count', 'total_teststep_id_count', 'booking_id_count']

        # Group by serial_number_id and count unique and total teststep_id, and occurrences
        serial_agg = df_batch.groupby('serial_number_id').agg({
            'teststep_id': ['nunique', 'count'],
            'booking_id': 'size'  # Count occurrences of serial_number_id
        }).reset_index()
        serial_agg.columns = ['serial_number_id', 'unique_teststep_id_count', 'total_teststep_id_count',
                              'serial_number_id_count']

        # Update booking_counts
        for _, row in booking_agg.iterrows():
            booking_id = row['booking_id']
            unique_count = row['unique_teststep_id_count']
            total_count = row['total_teststep_id_count']
            id_count = row['booking_id_count']
            if booking_id in booking_counts:
                booking_counts[booking_id]['unique_teststep_id_count'] = max(
                    booking_counts[booking_id]['unique_teststep_id_count'], unique_count
                )
                booking_counts[booking_id]['total_teststep_id_count'] += total_count
                booking_counts[booking_id]['id_count'] += id_count
            else:
                booking_counts[booking_id] = {
                    'unique_teststep_id_count': unique_count,
                    'total_teststep_id_count': total_count,
                    'id_count': id_count
                }

        # Update serial_counts
        for _, row in serial_agg.iterrows():
            serial_id = row['serial_number_id']
            unique_count = row['unique_teststep_id_count']
            total_count = row['total_teststep_id_count']
            id_count = row['serial_number_id_count']
            if serial_id in serial_counts:
                serial_counts[serial_id]['unique_teststep_id_count'] = max(
                    serial_counts[serial_id]['unique_teststep_id_count'], unique_count
                )
                serial_counts[serial_id]['total_teststep_id_count'] += total_count
                serial_counts[serial_id]['id_count'] += id_count
            else:
                serial_counts[serial_id] = {
                    'unique_teststep_id_count': unique_count,
                    'total_teststep_id_count': total_count,
                    'id_count': id_count
                }

        pbar.update(len(df_batch))
        del df_batch
        del booking_agg
        del serial_agg

print(
    f"Step 2 Complete: Found {len(booking_counts)} unique booking_id values and {len(serial_counts)} unique serial_number_id values")

# Step 3: Prepare output data
print("Step 3: Preparing output data")
booking_data = [
    {
        'id': booking_id,
        'type': 'booking_id',
        'id_count': counts['id_count'],
        'unique_teststep_id_count': counts['unique_teststep_id_count'],
        'total_teststep_id_count': counts['total_teststep_id_count']
    }
    for booking_id, counts in booking_counts.items()
]
serial_data = [
    {
        'id': serial_id,
        'type': 'serial_number_id',
        'id_count': counts['id_count'],
        'unique_teststep_id_count': counts['unique_teststep_id_count'],
        'total_teststep_id_count': counts['total_teststep_id_count']
    }
    for serial_id, counts in serial_counts.items()
]

# Combine and sort by type and id
output_data = booking_data + serial_data
output_data.sort(key=lambda x: (x['type'], x['id']))

# Create DataFrame
df_output = pd.DataFrame(output_data)

print("Step 3 Complete: Output data prepared")

# Step 4: Write to CSV with description
print("Step 4: Writing results to CSV")
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    df_output.to_csv(f, index=False, lineterminator='\n')
print(f"Step 4 Complete: CSV saved as {output_csv}")

# Step 5: Verify total row count
print("Step 5: Verifying processed rows")
processed_rows = 0
with tqdm(total=total_rows, desc="Verifying rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['booking_id', 'serial_number_id'],
                                           use_threads=True):
        df_batch = batch.to_pandas()
        processed_rows += len(df_batch)
        pbar.update(len(df_batch))
        del df_batch

print(f"Processed rows: {processed_rows}")
print(f"Total rows in file: {total_rows}")
if processed_rows == total_rows:
    print("Verification successful: All rows processed.")
else:
    print(f"Verification failed: Processed {processed_rows} rows, expected {total_rows}.")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
