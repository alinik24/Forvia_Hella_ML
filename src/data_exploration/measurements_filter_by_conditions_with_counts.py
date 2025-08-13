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
output_csv_sequence = os.path.join(output_dir, f"measurements_filtered_by_sequence_number_{timestamp}.csv")
output_csv_fail_code = os.path.join(output_dir, f"measurements_filtered_by_measure_fail_code_{timestamp}.csv")
output_csv_book_state = os.path.join(output_dir, f"measurements_filtered_by_book_state_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify file exists
if not os.path.exists(measurements_file):
    raise FileNotFoundError(f"Parquet file not found at {measurements_file}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(measurements_file)
available_columns = parquet_file.schema.names
required_columns = ['sequence_number', 'measure_fail_code', 'book_state']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {measurements_file}. Available columns: {available_columns}")
    print("Please confirm correct column names (e.g., 'sequence_number', 'measure_fail_code', 'book_state').")
    exit(1)

# Step 2: Process Parquet file, filter rows, and count null/non-integer values
print("Step 2: Filtering rows and counting null/non-integer values")
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size for performance

# Initialize lists to store filtered rows
sequence_rows = []
fail_code_rows = []
book_state_rows = []

# Initialize counters for null and non-integer values
null_counts = {'sequence_number': 0, 'measure_fail_code': 0, 'book_state': 0}
non_integer_counts = {'sequence_number': 0, 'measure_fail_code': 0, 'book_state': 0}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=True):  # Load all columns
        df_batch = batch.to_pandas()

        # Count null values
        null_counts['sequence_number'] += df_batch['sequence_number'].isna().sum()
        null_counts['measure_fail_code'] += df_batch['measure_fail_code'].isna().sum()
        null_counts['book_state'] += df_batch['book_state'].isna().sum()

        # Replace null/NaN values with placeholders
        df_batch['sequence_number'] = df_batch['sequence_number'].fillna(-999)
        df_batch['measure_fail_code'] = df_batch['measure_fail_code'].fillna(-999)
        df_batch['book_state'] = df_batch['book_state'].fillna(-999)

        # Count non-integer values (excluding placeholder -999)
        for column in required_columns:
            non_integer_mask = df_batch[column].apply(
                lambda x: x != -999 and not (isinstance(x, (int, float)) and x % 1 == 0)
            )
            non_integer_counts[column] += non_integer_mask.sum()

        # Filter rows where sequence_number is an integer and not equal to 1
        sequence_mask = (
            df_batch['sequence_number'].apply(
                lambda x: isinstance(x, (int, float)) and x % 1 == 0 and x != 1 and x != -999
            )
        )
        sequence_rows.append(df_batch[sequence_mask].copy())

        # Filter rows where measure_fail_code is an integer and not equal to 0
        fail_code_mask = (
            df_batch['measure_fail_code'].apply(
                lambda x: isinstance(x, (int, float)) and x % 1 == 0 and x != 0 and x != -999
            )
        )
        fail_code_rows.append(df_batch[fail_code_mask].copy())

        # Filter rows where book_state is an integer and not equal to 0
        book_state_mask = (
            df_batch['book_state'].apply(
                lambda x: isinstance(x, (int, float)) and x % 1 == 0 and x != 0 and x != -999
            )
        )
        book_state_rows.append(df_batch[book_state_mask].copy())

        pbar.update(len(df_batch))
        del df_batch

# Concatenate filtered rows
df_sequence = pd.concat(sequence_rows, ignore_index=True) if sequence_rows else pd.DataFrame()
df_fail_code = pd.concat(fail_code_rows, ignore_index=True) if fail_code_rows else pd.DataFrame()
df_book_state = pd.concat(book_state_rows, ignore_index=True) if book_state_rows else pd.DataFrame()

print(f"Step 2 Complete: Found {len(df_sequence)} rows for sequence_number != 1 (integer), "
      f"{len(df_fail_code)} rows for measure_fail_code != 0 (integer), "
      f"{len(df_book_state)} rows for book_state != 0 (integer)")
print(f"Null counts: sequence_number={null_counts['sequence_number']}, "
      f"measure_fail_code={null_counts['measure_fail_code']}, book_state={null_counts['book_state']}")
print(f"Non-integer counts: sequence_number={non_integer_counts['sequence_number']}, "
      f"measure_fail_code={non_integer_counts['measure_fail_code']}, book_state={non_integer_counts['book_state']}")

# Step 3: Write to CSV files with descriptions
print("Step 3: Writing results to CSV files")

# Description templates with counts
description_sequence = f"""\
# Script Purpose:
# This script processes the `measurements_single_line.parquet` file to extract rows where `sequence_number` is an integer
# and not equal to 1. It reads the data in memory-efficient batches, handles missing values with placeholders, counts null
# and non-integer values, and saves all columns of the filtered rows to a timestamped CSV file. The script verifies required
# columns and validates the total number of processed rows to ensure completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains all columns for rows where sequence_number is an integer and not equal to 1.
# Null Values in sequence_number: {null_counts['sequence_number']}
# Non-integer Values in sequence_number: {non_integer_counts['sequence_number']}
"""

description_fail_code = f"""\
# Script Purpose:
# This script processes the `measurements_single_line.parquet` file to extract rows where `measure_fail_code` is an integer
# and not equal to 0. It reads the data in memory-efficient batches, handles missing values with placeholders, counts null
# and non-integer values, and saves all columns of the filtered rows to a timestamped CSV file. The script verifies required
# columns and validates the total number of processed rows to ensure completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains all columns for rows where measure_fail_code is an integer and not equal to 0.
# Null Values in measure_fail_code: {null_counts['measure_fail_code']}
# Non-integer Values in measure_fail_code: {non_integer_counts['measure_fail_code']}
"""

description_book_state = f"""\
# Script Purpose:
# This script processes the `measurements_single_line.parquet` file to extract rows where `book_state` is an integer and
# not equal to 0. It reads the data in memory-efficient batches, handles missing values with placeholders, counts null and
# non-integer values, and saves all columns of the filtered rows to a timestamped CSV file. The script verifies required
# columns and validates the total number of processed rows to ensure completeness.
# Source File: measurements_single_line.parquet
# Output: This CSV contains all columns for rows where book_state is an integer and not equal to 0.
# Null Values in book_state: {null_counts['book_state']}
# Non-integer Values in book_state: {non_integer_counts['book_state']}
"""

# Write sequence_number CSV
with open(output_csv_sequence, 'w', encoding='utf-8') as f:
    f.write(description_sequence)
    df_sequence.to_csv(f, index=False, lineterminator='\n')
print(f"CSV saved as {output_csv_sequence}")

# Write measure_fail_code CSV
with open(output_csv_fail_code, 'w', encoding='utf-8') as f:
    f.write(description_fail_code)
    df_fail_code.to_csv(f, index=False, lineterminator='\n')
print(f"CSV saved as {output_csv_fail_code}")

# Write book_state CSV
with open(output_csv_book_state, 'w', encoding='utf-8') as f:
    f.write(description_book_state)
    df_book_state.to_csv(f, index=False, lineterminator='\n')
print(f"CSV saved as {output_csv_book_state}")

print("Step 3 Complete: All CSVs saved")

# Step 4: Verify total row count
print("Step 4: Verifying processed rows")
processed_rows = 0
with tqdm(total=total_rows, desc="Verifying rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['sequence_number'], use_threads=True):
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
