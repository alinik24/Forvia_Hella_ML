import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
bookings_file = os.path.join(base_path, "bookings_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(output_dir, f"bookings_unique_values_and_counts_by_serial_number_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify file exists
if not os.path.exists(bookings_file):
    raise FileNotFoundError(f"Parquet file not found at {bookings_file}. Please check the file path or name.")

# Step 1: Check column existence
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(bookings_file)
available_columns = parquet_file.schema.names
required_columns = [
    'serial_number_id', 'workstep_number_mes', 'sequence_number', 'lot_id', 'station_id',
    'station_diag_id', 'workorder_id', 'panel_position_number', 'workorder_type',
    'workorder_number', 'object_id', 'workplan_id', 'part_group', 'erp_group_id',
    'workstep_number_erp', 'workstep_id', 'erp_group_desc', 'line_id'
]

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {bookings_file}. Available columns: {available_columns}")
    print("Please confirm correct column names.")
    exit(1)

# Step 2: Collect unique values and counts per serial_number_id and total unique values
print("Step 2: Collecting unique values and counts per serial_number_id and across all serial_number_id")
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Optimized batch size for performance

# Dictionary to store unique values and counts per serial_number_id
serial_values = {}
# Dictionary to store total unique values and counts across all serial_number_id
total_unique_values = {
    'workstep_number_mes': {},
    'sequence_number': {},
    'lot_id': {},
    'station_id': {},
    'station_diag_id': {},
    'workorder_id': {},
    'panel_position_number': {},
    'workorder_type': {},
    'workorder_number': {},
    'object_id': {},
    'workplan_id': {},
    'part_group': {},
    'erp_group_id': {},
    'workstep_number_erp': {},
    'workstep_id': {},
    'erp_group_desc': {},
    'line_id': {}
}
# Dictionary to store total occurrences per serial_number_id for each column
serial_total_counts = {}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_columns, use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values with placeholders
        for col in required_columns:
            if col in ['workstep_number_mes', 'sequence_number', 'panel_position_number', 'workstep_number_erp']:
                df_batch[col] = df_batch[col].fillna(-999)
            else:
                df_batch[col] = df_batch[col].fillna("unlabeled")

        # Group by serial_number_id and collect unique values with counts
        for serial_id, group in df_batch.groupby('serial_number_id'):
            values = {
                'workstep_number_mes': group['workstep_number_mes'].value_counts().to_dict(),
                'sequence_number': group['sequence_number'].value_counts().to_dict(),
                'lot_id': group['lot_id'].value_counts().to_dict(),
                'station_id': group['station_id'].value_counts().to_dict(),
                'station_diag_id': group['station_diag_id'].value_counts().to_dict(),
                'workorder_id': group['workorder_id'].value_counts().to_dict(),
                'panel_position_number': group['panel_position_number'].value_counts().to_dict(),
                'workorder_type': group['workorder_type'].value_counts().to_dict(),
                'workorder_number': group['workorder_number'].value_counts().to_dict(),
                'object_id': group['object_id'].value_counts().to_dict(),
                'workplan_id': group['workplan_id'].value_counts().to_dict(),
                'part_group': group['part_group'].value_counts().to_dict(),
                'erp_group_id': group['erp_group_id'].value_counts().to_dict(),
                'workstep_number_erp': group['workstep_number_erp'].value_counts().to_dict(),
                'workstep_id': group['workstep_id'].value_counts().to_dict(),
                'erp_group_desc': group['erp_group_desc'].value_counts().to_dict(),
                'line_id': group['line_id'].value_counts().to_dict()
            }
            # Count total occurrences for each column
            total_counts = {
                col: len(group[col].dropna()) for col in required_columns[1:]
            }
            if serial_id in serial_values:
                # Merge counts for each field
                for key in values:
                    for val, count in values[key].items():
                        serial_values[serial_id][key][val] = serial_values[serial_id][key].get(val, 0) + count
                for key in total_counts:
                    serial_total_counts[serial_id][key] = serial_total_counts[serial_id].get(key, 0) + total_counts[key]
            else:
                serial_values[serial_id] = values
                serial_total_counts[serial_id] = total_counts

        # Update total unique values and counts
        for col in required_columns[1:]:  # Exclude serial_number_id
            value_counts = df_batch[col].value_counts().to_dict()
            for val, count in value_counts.items():
                total_unique_values[col][val] = total_unique_values[col].get(val, 0) + count

        pbar.update(len(df_batch))
        del df_batch

# Calculate total unique counts and total unique serial_number_id
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
    # Add total counts for each column
    formatted_total_counts = {
        col: serial_total_counts[serial_id][col] for col in required_columns[1:]
    }
    output_data.append({
        'serial_number_id': serial_id,
        'workstep_number_mes': formatted_values['workstep_number_mes'],
        'sequence_number': formatted_values['sequence_number'],
        'lot_id': formatted_values['lot_id'],
        'station_id': formatted_values['station_id'],
        'station_diag_id': formatted_values['station_diag_id'],
        'workorder_id': formatted_values['workorder_id'],
        'panel_position_number': formatted_values['panel_position_number'],
        'workorder_type': formatted_values['workorder_type'],
        'workorder_number': formatted_values['workorder_number'],
        'object_id': formatted_values['object_id'],
        'workplan_id': formatted_values['workplan_id'],
        'part_group': formatted_values['part_group'],
        'erp_group_id': formatted_values['erp_group_id'],
        'workstep_number_erp': formatted_values['workstep_number_erp'],
        'workstep_id': formatted_values['workstep_id'],
        'erp_group_desc': formatted_values['erp_group_desc'],
        'line_id': formatted_values['line_id']
    })

# Add row for total unique values
formatted_total_values = {
    col: ';'.join(f"{val}:{count}" for val, count in sorted(values.items()))
    for col, values in total_unique_values.items()
}
output_data.append({
    'serial_number_id': 'TOTAL',
    'workstep_number_mes': formatted_total_values['workstep_number_mes'],
    'sequence_number': formatted_total_values['sequence_number'],
    'lot_id': formatted_total_values['lot_id'],
    'station_id': formatted_total_values['station_id'],
    'station_diag_id': formatted_total_values['station_diag_id'],
    'workorder_id': formatted_total_values['workorder_id'],
    'panel_position_number': formatted_total_values['panel_position_number'],
    'workorder_type': formatted_total_values['workorder_type'],
    'workorder_number': formatted_total_values['workorder_number'],
    'object_id': formatted_total_values['object_id'],
    'workplan_id': formatted_total_values['workplan_id'],
    'part_group': formatted_total_values['part_group'],
    'erp_group_id': formatted_total_values['erp_group_id'],
    'workstep_number_erp': formatted_total_values['workstep_number_erp'],
    'workstep_id': formatted_total_values['workstep_id'],
    'erp_group_desc': formatted_total_values['erp_group_desc'],
    'line_id': formatted_total_values['line_id']
})

# Sort by serial_number_id
output_data.sort(key=lambda x: x['serial_number_id'])

# Create DataFrame
df_output = pd.DataFrame(output_data)

# Reorder columns for clarity
columns = ['serial_number_id'] + [col for col in required_columns[1:]]
df_output = df_output[columns]

print("Step 3 Complete: Output data prepared")

# Step 4: Write to CSV with description
print("Step 4: Writing results to CSV")
description = f"""\
# Script Purpose:
# This script processes the `bookings_single_line.parquet` file to list, for every unique `serial_number_id`, the unique
# values and their counts for specified columns, and the unique values and counts across all serial_number_id in a final row
# labeled 'TOTAL'. The script reads the data in memory-efficient batches, handles missing values with placeholders
# ("unlabeled" for strings, -999 for numerics), and saves the results to a timestamped CSV file. The script verifies required
# columns and validates the total number of processed rows to ensure completeness.
# Source File: bookings_single_line.parquet
# Output: This CSV contains unique values and their counts for each specified column per serial_number_id,
# with a final row listing unique values and counts across all serial_number_id (serial_number_id='TOTAL'). Values are
# formatted as 'value:count' pairs, semicolon-separated.
# Total Unique Counts:
#   Total unique serial_number_id: {total_unique_serials}
#   workstep_number_mes: {total_unique_counts['workstep_number_mes']}
#   sequence_number: {total_unique_counts['sequence_number']}
#   lot_id: {total_unique_counts['lot_id']}
#   station_id: {total_unique_counts['station_id']}
#   station_diag_id: {total_unique_counts['station_diag_id']}
#   workorder_id: {total_unique_counts['workorder_id']}
#   panel_position_number: {total_unique_counts['panel_position_number']}
#   workorder_type: {total_unique_counts['workorder_type']}
#   workorder_number: {total_unique_counts['workorder_number']}
#   object_id: {total_unique_counts['object_id']}
#   workplan_id: {total_unique_counts['workplan_id']}
#   part_group: {total_unique_counts['part_group']}
#   erp_group_id: {total_unique_counts['erp_group_id']}
#   workstep_number_erp: {total_unique_counts['workstep_number_erp']}
#   workstep_id: {total_unique_counts['workstep_id']}
#   erp_group_desc: {total_unique_counts['erp_group_desc']}
#   line_id: {total_unique_counts['line_id']}
# Total Unique Values and Counts:
{';'.join(f"{col}: {';'.join(f'{val}:{count}' for val, count in sorted(total_unique_values[col].items()))}" for col in required_columns[1:])}
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
