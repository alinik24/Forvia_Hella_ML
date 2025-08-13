import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(output_dir, f"measurements_unique_product_combinations_and_values_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify file exists
if not os.path.exists(measurements_file):
    raise FileNotFoundError(f"Parquet file not found at {measurements_file}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(measurements_file)
available_columns = parquet_file.schema.names
required_columns = ['product_id', 'product_variant_id', 'part_number']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {measurements_file}. Available columns: {available_columns}")
    print("Please confirm correct column names (e.g., 'product_id', 'product_variant_id', 'part_number').")
    exit(1)

# Step 2: Collect unique combinations and standalone unique values
print(
    "Step 2: Collecting unique combinations and standalone values for product_id, product_variant_id, and part_number")
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size for performance

# Sets to store unique values
unique_combinations = set()
unique_product_ids = set()
unique_product_variant_ids = set()
unique_part_numbers = set()

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_columns, use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values with placeholders
        df_batch.fillna("unlabeled", inplace=True)

        # Extract unique combinations
        combinations = df_batch[['product_id', 'product_variant_id', 'part_number']].drop_duplicates()
        for row in combinations.itertuples(index=False):
            unique_combinations.add((row.product_id, row.product_variant_id, row.part_number))

        # Extract unique standalone values
        unique_product_ids.update(df_batch['product_id'].unique())
        unique_product_variant_ids.update(df_batch['product_variant_id'].unique())
        unique_part_numbers.update(df_batch['part_number'].unique())

        pbar.update(len(df_batch))
        # Free memory
        del df_batch
        del combinations

print(f"Step 2 Complete: Found {len(unique_combinations)} unique combinations, "
      f"{len(unique_product_ids)} unique product_id values, "
      f"{len(unique_product_variant_ids)} unique product_variant_id values, "
      f"{len(unique_part_numbers)} unique part_number values")

# Step 3: Prepare output data
print("Step 3: Preparing output data")

# Prepare combinations section
combinations_data = [
    {
        'type': 'combination',
        'product_id': combo[0],
        'product_variant_id': combo[1],
        'part_number': combo[2]
    }
    for combo in sorted(unique_combinations)  # Sort for consistent output
]

# Prepare standalone unique values
product_id_data = [
    {
        'type': 'product_id',
        'product_id': pid,
        'product_variant_id': '',
        'part_number': ''
    }
    for pid in sorted(unique_product_ids)
]
product_variant_id_data = [
    {
        'type': 'product_variant_id',
        'product_id': '',
        'product_variant_id': pvid,
        'part_number': ''
    }
    for pvid in sorted(unique_product_variant_ids)
]
part_number_data = [
    {
        'type': 'part_number',
        'product_id': '',
        'product_variant_id': '',
        'part_number': pn
    }
    for pn in sorted(unique_part_numbers)
]

# Combine all data
output_data = combinations_data + product_id_data + product_variant_id_data + part_number_data

# Create DataFrame
df_output = pd.DataFrame(output_data)

# Reorder columns for clarity
df_output = df_output[['type', 'product_id', 'product_variant_id', 'part_number']]

print("Step 3 Complete: Output data prepared")

# Step 4: Write to CSV with description
print("Step 4: Writing results to CSV")
description = f"""\
# Script Purpose:
# This script processes the `measurements_single_line.parquet` file to extract all unique combinations of `product_id`,
# `product_variant_id`, and `part_number`, as well as unique standalone values for each of these columns. It reads the file
# in large, memory-efficient batches, replaces missing values with a placeholder ("unlabeled"), and stores results in sets.
# The final results are sorted and saved to a timestamped CSV file for further analysis or reference.
# Source File: measurements_single_line.parquet
# Output: This CSV contains unique combinations of product_id, product_variant_id, and part_number (type='combination'),
# and unique standalone values for product_id (type='product_id'), product_variant_id (type='product_variant_id'),
# and part_number (type='part_number'). Empty fields are represented as '' for standalone values.
# Unique Combinations: {len(unique_combinations)}
# Unique product_id Values: {len(unique_product_ids)}
# Unique product_variant_id Values: {len(unique_product_variant_ids)}
# Unique part_number Values: {len(unique_part_numbers)}
"""
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    df_output.to_csv(f, index=False, lineterminator='\n')
print(f"Step 4 Complete: CSV saved as {output_csv}")

# Step 5: Verify processed rows
print("Step 5: Verifying processed rows")
processed_rows = 0
with tqdm(total=total_rows, desc="Verifying rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['product_id'], use_threads=True):
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
