# This script scans the `measurements_single_line.parquet` file to extract all rows where the `measure_value` column is missing (NaN).
# It processes the file in memory-efficient batches, appends any matching rows to a list, and combines them into a single DataFrame.
# The resulting set of rows with missing `measure_value` is saved to a timestamped CSV file for further inspection or cleaning.
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
output_csv = os.path.join(base_path, f"rows_with_missing_measure_value_{timestamp}.csv")

# Read the Parquet file and collect rows with missing measure_value
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Collecting rows with missing measure_value (all columns)")

parquet_file = pq.ParquetFile(measurements_file)
batch_size = 100000  # Smaller batch size to reduce memory usage
missing_rows = []

# Read all columns (no specific column selection)
for batch in parquet_file.iter_batches(batch_size=batch_size):
    df_batch = batch.to_pandas()
    # Collect rows where measure_value is missing
    missing_mask = df_batch['measure_value'].isna()
    if missing_mask.any():
        missing_rows.append(df_batch[missing_mask])
    del df_batch  # Free memory

# Combine all missing rows into a single DataFrame
missing_rows_df = pd.concat(missing_rows, ignore_index=True) if missing_rows else pd.DataFrame(
    columns=parquet_file.schema.names)

print(f"Found {len(missing_rows_df)} rows with missing measure_value")

# Save to CSV
print("Writing rows with missing measure_value to CSV")
missing_rows_df.to_csv(output_csv, index=False)
print(f"CSV saved as {output_csv}")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
