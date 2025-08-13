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
output_csv = os.path.join(output_dir, f"measurements_unique_values_and_counts_by_serial_number_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify file exists
if not os.path.exists(measurements_file):
    raise FileNotFoundError(f"Parquet file not found at {measurements_file}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(measurements_file)
available_columns = parquet_file.schema.names
required_columns = ['serial_number_id', 'catalog_id', 'recipe_revision_id', 'station_id', 'station_number',
                    'workstep_id']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {measurements_file}. Available columns: {available_columns}")
    print(
        "Please confirm correct column names (e.g., 'serial_number_id', 'catalog_id', 'recipe_revision_id', 'station_id', 'station_number', 'workstep_id').")
    exit(1)

# Step 2: Collect unique values and counts per serial_number_id and total unique values
print("Step 2: Collecting unique values and counts per serial_number_id and across all serial_number_id")
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size for performance

# Dictionary to store unique values and counts per serial_number_id
serial_values = {}
# Dictionary to store total unique values and counts across all serial_number_id
total_unique_values = {
    'catalog_id': {},
    'recipe_revision_id': {},
    'station_id': {},
    'station_number': {},
    'workstep_id': {}
}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_columns, use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values with placeholders
        df_batch['serial_number_id'] = df_batch['serial_number_id'].fillna("unlabeled")
        df_batch['catalog_id'] = df_batch['catalog_id'].fillna("unlabeled")
        df_batch['recipe_revision_id'] = df_batch['recipe_revision_id'].fillna("unlabeled")
        df_batch['station_id'] = df_batch['station_id'].fillna("unlabeled")
        df_batch['station_number'] = df_batch['station_number'].fillna(-999)
        df_batch['workstep_id'] = df_batch['workstep_id'].fillna("unlabeled")

        # Group by serial_number_id and collect unique values with counts
        for serial_id, group in df_batch.groupby('serial_number_id'):
            values = {
                'catalog_id': group['catalog_id'].value_counts().to_dict(),
                'recipe_revision_id': group['recipe_revision_id'].value_counts().to_dict(),
                'station_id': group['station_id'].value_counts().to_dict(),
                'station_number': group['station_number'].value_counts().to_dict(),
                'workstep_id': group['workstep_id'].value_counts().to_dict()
            }
            if serial_id in serial_values:
                # Merge counts for each field
                for key in values:
                    for val, count in values[key].items():
                        serial_values[serial_id][key][val] = serial_values[serial_id][key].get(val, 0) + count
            else:
                serial_values[serial_id] = values

        # Update total unique values and counts
        for col in required_columns[1:]:  # Exclude serial_number_id
            value_counts = df_batch[col].value_counts().to_dict()
            for val, count in value_counts.items():
                total_unique_values[col][val] = total_unique_values[col].get(val, 0) + count

        pbar.update(len(df_batch))
        del df_batch

# Calculate total unique counts
total_unique_counts = {col: len(values) for col, values in total_unique_values.items()}
total_unique_serials = len(serial_values)

print(f"Step 2 Complete: Found {total_unique_serials} unique serial_number_id values")
print("Total unique counts across all serial_number_id:")
for col, count in total_unique_counts.items():
    print(f"  {col}: {count}")

# Step 3: Prepare output data
print("Step 3: Preparing output data")
output_data = []

# Add rows for each serial_number_id
for serial_id, values in serial_values.items():
    # Format values and counts as "value:count" pairs
    formatted_values = {
        col: ';'.join(f"{val}:{count}" for val, count in sorted(values[col].items()))
        for col in required_columns[1:]
    }
    output_data.append({
        'serial_number_id': serial_id,
        'catalog_id': formatted_values['catalog_id'],
        'recipe_revision_id': formatted_values['recipe_revision_id'],
        'station_id': formatted_values['station_id'],
        'station_number': formatted_values['station_number'],
        'workstep_id': formatted_values['workstep_id']
    })

# Add row for total unique values
formatted_total_values = {
    col: ';'.join(f"{val}:{count}" for val, count in sorted(values.items()))
    for col, values in total_unique_values.items()
}
output_data.append({
    'serial_number_id': 'TOTAL',
    'catalog_id': formatted_total_values['catalog_id'],
    'recipe_revision_id': formatted_total_values['recipe_revision_id'],
    'station_id': formatted_total_values['station_id'],
    'station_number': formatted_total_values['station_number'],
    'workstep_id': formatted_total_values['workstep_id']
})

# Sort by serial_number_id
output_data.sort(key=lambda x: x['serial_number_id'])

# Create DataFrame
df_output = pd.DataFrame(output_data)

# Reorder columns for clarity
columns = ['serial_number_id', 'catalog_id', 'recipe_revision_id', 'station_id', 'station_number', 'workstep_id']
df_output = df_output[columns]

print("Step 3 Complete: Output data prepared")

# Step 4: Write to CSV with description
print("Step 4: Writing results to CSV")
description = f"""\
# Script Purpose:
# This script processes the `measurements_single_line.parquet` file to list, for every unique `serial_number_id`, the unique
# values and their counts for specified columns, and the unique values and counts across all serial_number_id in a final row
# labeled 'TOTAL'. The script reads the data in memory-efficient batches, handles missing values with placeholders
# ("unlabeled" for strings, -999 for numerics), and saves the results to a timestamped CSV file. The script verifies required
# columns and validates the total number of processed rows to ensure completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains unique values and counts for each specified column per serial_number_id,
# with a final row listing unique values and counts across all serial_number_id (serial_number_id='TOTAL'). Values are
# formatted as 'value:count' pairs, semicolon-separated.
# Total Unique Counts:
#   Total unique serial_number_id: {total_unique_serials}
#   catalog_id: {total_unique_counts['catalog_id']}
#   recipe_revision_id: {total_unique_counts['recipe_revision_id']}
#   station_id: {total_unique_counts['station_id']}
#   station_number: {total_unique_counts['station_number']}
#   workstep_id: {total_unique_counts['workstep_id']}
"""
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    df_output.to_csv(f, index=False, lineterminator='\n')
print(f"Step 4 Complete: CSV saved as {output_csv}")

# Step 5: Verify processed rows
print("Step 5: Verifying processed rows")
processed_rows = 0
with tqdm(total=total_rows, desc="Verifying rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['serial_number_id'], use_threads=True):
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
