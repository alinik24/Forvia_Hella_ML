# This script processes a Parquet file (`bookings_single_line.parquet`) to compute, for each `serial_number_id`, the count of unique and
# total values across several booking-related columns (e.g., `booking_id`, `station_id`, `workstep_id`). It reads the data in large batches
# to optimize performance, replaces missing values with a placeholder, and aggregates the statistics using group-by operations. The results
# are saved to a timestamped CSV file with a multi-level column header and verified to ensure that the row count matches the total number
# of rows in the original file.
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
parquet_file_path = os.path.join(base_path, "bookings_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(base_path, f"serial_booking_counts_{timestamp}.csv")

# Verify file exists
if not os.path.exists(parquet_file_path):
    raise FileNotFoundError(f"Parquet file not found at {parquet_file_path}. Please check the file path or name.")

# Columns to process
columns = [
    'booking_id', 'book_state', 'workstep_number_mes', 'sequence_number', 'lot_id',
    'station_id', 'station_diag_id', 'workorder_id', 'panel_position_number',
    'station_number', 'object_id', 'workplan_id', 'erp_group_id', 'workstep_number_erp',
    'workstep_id', 'erp_group_desc', 'line_id', 'serial_number'
]

# Step 1: Read the Parquet file and collect counts
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Step 1: Collecting counts for {len(columns)} columns per serial_number_id")

parquet_file = pq.ParquetFile(parquet_file_path)
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size

# Dictionary to store counts
serial_counts = {}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['serial_number_id'] + columns,
                                           use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values
        df_batch.fillna("unlabeled", inplace=True)

        # Group by serial_number_id
        agg = df_batch.groupby('serial_number_id').agg({
            col: ['nunique', 'count'] for col in columns
        })

        # Flatten multi-level column names
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

        # Update serial_counts
        for serial, row in agg.iterrows():
            if serial not in serial_counts:
                serial_counts[serial] = {}
            for col in columns:
                serial_counts[serial][f"unique_{col}_count"] = row[f"{col}_nunique"]
                serial_counts[serial][f"total_{col}_count"] = row[f"{col}_count"]

        pbar.update(len(df_batch))
        del df_batch
        del agg

print(f"Step 1 Complete: Found {len(serial_counts)} unique serial_number_id values")

# Step 2: Prepare output data for CSV
print("Step 2: Preparing output data for CSV")
output_data = []
for serial in serial_counts:
    row = {'serial_number_id': serial}
    for col in columns:
        row[f"unique_{col}_count"] = serial_counts[serial].get(f"unique_{col}_count", 0)
        row[f"total_{col}_count"] = serial_counts[serial].get(f"total_{col}_count", 0)
    output_data.append(row)

# Sort by serial_number_id
output_data.sort(key=lambda x: x['serial_number_id'])

# Create DataFrame for CSV
df_output = pd.DataFrame(output_data)

# Define multi-level header
header1 = ["serial_number_id"] + [col for col in columns for _ in range(2)]
header2 = [""] + ["unique", "total"] * len(columns)
multi_index = pd.MultiIndex.from_arrays([header1, header2])

# Assign multi-level header to DataFrame
df_output.columns = multi_index

print("Step 2 Complete: Output data prepared")

# Step 3: Write to CSV
print("Step 3: Writing results to CSV")
df_output.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Step 3 Complete: CSV saved as {output_csv}")

# Step 4: Verify total row count
print("Step 4: Verifying total row count")
total_counted_rows = sum(row[f"total_booking_id_count"] for row in output_data)
print(f"Sum of total booking_id counts: {total_counted_rows}")
print(f"Total rows in file: {total_rows}")
if total_counted_rows == total_rows:
    print("Verification successful: All rows accounted for.")
else:
    print(
        f"Verification failed: Sum of total booking_id counts ({total_counted_rows}) does not match total rows ({total_rows}).")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
