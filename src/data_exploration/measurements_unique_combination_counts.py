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
output_csv = os.path.join(output_dir, f"measurements_unique_combination_counts_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Description to include in CSV
description = """\
# Script Purpose:
# This script processes a large Parquet file (`measurements_single_line.parquet`) to identify and count all unique combinations
# of `measurement_unit`, `measurement_type`, `lower_limit`, and `upper_limit`. It reads the data in memory-efficient batches,
# groups rows by these four fields (handling missing values with placeholders), and calculates the frequency and percentage of
# each unique combination. The results are saved to a timestamped CSV file and verified against the total row count to ensure
# accuracy and completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains unique combinations of measurement_unit, measurement_type, lower_limit, and upper_limit, along with their row counts and percentages.
"""

# Verify file exists
if not os.path.exists(measurements_file):
    raise FileNotFoundError(f"Parquet file not found at {measurements_file}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(measurements_file)
available_columns = parquet_file.schema.names
required_columns = ['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {measurements_file}. Available columns: {available_columns}")
    print(
        "Please confirm correct column names (e.g., 'measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit').")
    exit(1)

# Step 2: Collect unique combinations
print("Step 2: Collecting unique combinations of measurement_unit, measurement_type, lower_limit, and upper_limit")
print("Rows with identical values for all four parameters are considered duplicates and counted together")
total_rows = parquet_file.metadata.num_rows
batch_size = 100000  # Smaller batch size to reduce memory usage (100k rows per batch)

combinations = {}
with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size,
                                           columns=['measurement_unit', 'measurement_type', 'lower_limit',
                                                    'upper_limit'], use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values with placeholders to ensure they are included in grouping
        df_batch['measurement_unit'] = df_batch['measurement_unit'].fillna("unlabeled")
        df_batch['measurement_type'] = df_batch['measurement_type'].fillna("unlabeled")
        df_batch['lower_limit'] = df_batch['lower_limit'].fillna(-999)
        df_batch['upper_limit'] = df_batch['upper_limit'].fillna(-999)
        # Group by the four columns to identify duplicates (identical combinations)
        group = df_batch.groupby(['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit']).size()
        for (unit, mtype, lower, upper), count in group.items():
            combination = (unit, mtype, lower, upper)
            combinations[combination] = combinations.get(combination, 0) + count
        pbar.update(len(df_batch))
        # Free memory
        del df_batch
        del group

print(f"Step 2 Complete: Found {len(combinations)} unique combinations")

# Step 3: Prepare output
print("Step 3: Preparing output")
output_data = []
for combination, count in tqdm(combinations.items(), desc="Preparing output data", unit="combination"):
    unit, mtype, lower, upper = combination
    percentage = (count / total_rows) * 100  # Calculate percentage
    output_data.append({
        'measurement_unit': unit,
        'measurement_type': mtype,
        'lower_limit': lower,
        'upper_limit': upper,
        'row_count': count,
        'percentage': percentage
    })

# Sort by row_count in descending order for better readability
output_data.sort(key=lambda x: x['row_count'], reverse=True)

# Create DataFrame
df_output = pd.DataFrame(output_data)

print("Step 3 Complete: Output prepared")

# Step 4: Write to CSV with description
print("Step 4: Writing results to CSV")
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    df_output.to_csv(f, index=False, lineterminator='\n')
print(f"Step 4 Complete: CSV saved as {output_csv}")

# Step 5: Verify total row count and percentage
print("Step 5: Verifying total row count and percentage")
total_counted_rows = sum(row['row_count'] for row in output_data)
total_percentage = sum(row['percentage'] for row in output_data)
print(f"Sum of row counts: {total_counted_rows}")
print(f"Total rows in file: {total_rows}")
print(f"Sum of percentages: {total_percentage:.2f}%")
if total_counted_rows == total_rows and abs(total_percentage - 100.0) < 0.01:  # Allow for floating-point precision
    print("Verification successful: All rows and percentages are accounted for.")
else:
    print(
        f"Verification failed: Sum of row counts ({total_counted_rows}) does not match total rows ({total_rows}), or sum of percentages ({total_percentage:.2f}%) deviates from 100%.")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
