# This script extracts all unique `teststep_id` values from the `measurements_single_line.parquet` file. It reads the data in
# memory-efficient batches, filters out nulls, and collects distinct IDs into a set. The final list of unique `teststep_id`s
# is saved to a timestamped CSV file, providing a compact reference for downstream tasks such as single_datasets, mapping, or analysis.
import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
output_csv = os.path.join(base_path, f"teststep_ids_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

# Step 1: Read the Parquet file and collect unique teststep_id
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Step 1: Collecting unique teststep_id values")

parquet_file = pq.ParquetFile(measurements_file)
total_rows = parquet_file.metadata.num_rows
batch_size = 100000  # Smaller batch size to reduce memory usage (100k rows per batch)

teststep_ids = set()
with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['teststep_id']):
        df_batch = batch.to_pandas()
        # Add unique teststep_id values to the set, excluding null values
        teststep_ids.update(df_batch['teststep_id'].dropna().unique())
        pbar.update(len(df_batch))
        # Free memory by deleting the batch DataFrame
        del df_batch

print(f"Step 1 Complete: Found {len(teststep_ids)} unique teststep_id values")

# Step 2: Prepare output
print("Step 2: Preparing output")
output_data = []
for teststep_id in tqdm(sorted(teststep_ids), desc="Preparing output data", unit="teststep_id"):
    output_data.append({'teststep_id': teststep_id})

print("Step 2 Complete: Output prepared")

# Step 3: Write to CSV
print("Step 3: Writing results to CSV")
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write("teststep_id\n")
    for row in tqdm(output_data, desc="Writing to CSV", unit="row"):
        f.write(f"{row['teststep_id']}\n")

print(f"Step 3 Complete: CSV saved as {output_csv}")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")