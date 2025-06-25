# Key Features:
# 1. Input File:
#    - Processes `measurements_non_labeled.parquet` (filtered by non-labeled `station_desc`).
# 2. Grouping Logic:
#    - Combinations: Rows with valid (non-null, numeric) lower_limit and upper_limit AND at least one of measurement_unit or measurement_type non-null. Grouped by unique (measurement_unit, measurement_type, lower_limit, upper_limit).
#    - Missing Values: Rows lacking valid lower_limit or upper_limit OR missing both measurement_unit and measurement_type. Grouped by measurement_name.
#    - Every row belongs to exactly one group.
# 3. Summary CSVs:
#    - Primary (`measurements_group_summary_{timestamp}.csv`):
#      - Header: Total rows, unique measurement_name count and list, unique measurement_name counts under combinations and missing values, total unique groups, unique combinations and missing value groups, unique measurement_unit/measurement_type in missing_values, multi-unit/type measurement_names, verification status, earliest/latest created_at.
#      - Columns: group_name, category, measurement_unit, measurement_type, lower_limit, upper_limit, measurement_name, row_count, percentage.
#      - Sorted by row_count (descending).
#      - For missing_values, uses mode of measurement_unit/measurement_type/lower_limit/upper_limit.
#    - Secondary (`measurements_unique_measurement_names_{timestamp}.csv`):
#      - Header: Total count of unique measurement_name values.
#      - Columns: measurement_name, row_count, percentage.
#      - Lists all unique measurement_name values with counts and percentages.
# 4. Output Parquet:
#    - Preserves all original columns and data.
#    - Adds group_name column (combination name or measurement_name).
#    - Rows sorted by group_name.
#    - Matches input schema (lower_limit/upper_limit as strings).
# 5. Timestamps:
#    - Earliest/latest created_at with full row, row number, and total rows.
# 6. Additional Metrics:
#    - Count of all unique measurement_name values (overall, under combinations, under missing values).
#    - Total number of unique groups (combinations + missing value subgroups).
#    - Percentage of rows for each group.
#    - Sum of row counts for verification.
# 7. Verification:
#    - Confirms sum of row_count equals total rows.
#    - Checks all rows are processed.
# 8. Debugging:
#    - Logs sample rows for combinations and missing_values (first 100 rows).
#    - Tracks measurement_names in both categories to detect overlaps.
#    - Counts unique measurement_unit/measurement_type per measurement_name in missing_values.
#
# Output Structure:
# - Primary Summary CSV:
#   - Header: Total rows, unique measurement_name details, unique groups, verification, timestamps, unique measurement_unit/measurement_type in missing_values, multi-unit/type measurement_names, overlapping measurement_names.
#   - Columns: group_name, category, measurement_unit, measurement_type, lower_limit, upper_limit, measurement_name, row_count, percentage.
#   - Rows: One per combination (with measurement_name list) or measurement_name subgroup.
# - Secondary Summary CSV:
#   - Header: Total unique measurement_name count.
#   - Columns: measurement_name, row_count, percentage.
#   - Rows: One per unique measurement_name.
# - Output Parquet:
#   - Original columns + group_name.
#   - Rows grouped by group_name.
#
# Notes:
# - Assumptions:
#   - measurements_non_labeled.parquet exists and contains required columns.
#   - All rows can be grouped.
# - Batch Size: 50,000 rows to reduce memory usage.
# - Error Handling: Checks for missing columns and file existence.
# - Enhanced missing values detection to handle empty strings, invalid values in lower_limit/upper_limit.
# - Converts lower_limit/upper_limit to strings for parquet output to match input schema.
# - Tracks overlapping measurement_names to verify grouping logic.

import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa

# Define paths
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
input_parquet = os.path.join(output_dir, "measurements_non_labeled.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(output_dir, f"measurements_group_summary_{timestamp}.csv")
output_mnames_csv = os.path.join(output_dir, f"measurements_unique_measurement_names_{timestamp}.csv")
output_parquet = os.path.join(output_dir, f"measurements_non_labeled_with_groups.parquet")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify input file exists
if not os.path.exists(input_parquet):
    raise FileNotFoundError(f"Filtered parquet file not found at {input_parquet}. Please ensure the previous script has run.")

# Step 1: Check column existence and schema
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
parquet_file = pq.ParquetFile(input_parquet)
available_columns = parquet_file.schema.names
required_columns = ['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit', 'measurement_name', 'created_at']

missing_cols = [col for col in required_columns if col not in available_columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {input_parquet}. Available columns: {available_columns}")
    exit(1)

# Print input schema for debugging
print("Input parquet schema:")
for field in parquet_file.schema_arrow:
    print(f"  {field.name}: {field.type}")

# Step 2: Collect groups, measurement names, timestamps, and missing values details
print("Step 2: Collecting groups, measurement names, and timestamps")
total_rows = parquet_file.metadata.num_rows
batch_size = 50000  # Batch size to manage memory

combinations = {}  # (unit, mtype, lower, upper) -> count
combination_mnames = {}  # (unit, mtype, lower, upper) -> set of measurement_name
missing_groups = {}  # measurement_name -> count
missing_mnames = {}  # measurement_name -> set of measurement_name (single value)
missing_values_details = {}  # measurement_name -> (unit, mtype, lower, upper)
missing_multi_units_types = {}  # measurement_name -> (unit_count, type_count)
combination_names = {}  # (unit, mtype, lower, upper) -> name
all_mnames_counts = {}  # measurement_name -> count
all_mnames = set()  # All unique measurement_name values
missing_units = set()  # Unique measurement_unit in missing_values
missing_types = set()  # Unique measurement_type in missing_values
overlapping_mnames = set()  # measurement_names in both combinations and missing_values
earliest_time = None
latest_time = None
earliest_row = None
latest_row = None
earliest_row_number = None
latest_row_number = None
row_counter = 0
combination_samples = 0  # Track sampled combination rows
missing_samples = 0  # Track sampled missing rows

with tqdm(total=total_rows, desc="Processing parquet rows for groups", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=True):
        df_batch = batch.to_pandas()
        batch_row_count = len(df_batch)
        batch_indices = range(row_counter, row_counter + batch_row_count)

        # Collect all measurement_name values and counts
        df_batch['measurement_name'] = df_batch['measurement_name'].fillna("unlabeled_measurement_name")
        mname_counts = df_batch['measurement_name'].value_counts()
        for mname, count in mname_counts.items():
            all_mnames.add(mname)
            all_mnames_counts[mname] = all_mnames_counts.get(mname, 0) + count

        # Clean and validate data
        df_batch['lower_limit'] = pd.to_numeric(df_batch['lower_limit'], errors='coerce')
        df_batch['upper_limit'] = pd.to_numeric(df_batch['upper_limit'], errors='coerce')
        df_batch['measurement_unit'] = df_batch['measurement_unit'].replace(['', ' ', 'null'], pd.NA)
        df_batch['measurement_type'] = df_batch['measurement_type'].replace(['', ' ', 'null'], pd.NA)

        # Identify combination rows: valid lower_limit, upper_limit, and at least one of unit or type
        has_lower = df_batch['lower_limit'].notna()
        has_upper = df_batch['upper_limit'].notna()
        has_unit = df_batch['measurement_unit'].notna()
        has_type = df_batch['measurement_type'].notna()
        combination_mask = has_lower & has_upper & (has_unit | has_type)
        df_combination = df_batch[combination_mask].copy()
        if not df_combination.empty:
            # Fill missing unit/type with 'unlabeled' for grouping
            df_combination['measurement_unit'] = df_combination['measurement_unit'].fillna("unlabeled")
            df_combination['measurement_type'] = df_combination['measurement_type'].fillna("unlabeled")
            # Log sample combination rows (first 100)
            if combination_samples < 100:
                sample_count = min(100 - combination_samples, len(df_combination))
                print(f"Sample combination rows (batch {row_counter // batch_size + 1}):\n{df_combination[['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit', 'measurement_name']].head(sample_count)}")
                combination_samples += sample_count
            # Group by combination
            group = df_combination.groupby(['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit'])
            for (unit, mtype, lower, upper), group_df in group:
                combination = (unit, mtype, lower, upper)
                count = len(group_df)
                combinations[combination] = combinations.get(combination, 0) + count
                if combination not in combination_names:
                    combination_names[combination] = f"{unit},{mtype},{lower},{upper}"
                if combination not in combination_mnames:
                    combination_mnames[combination] = set()
                new_mnames = set(group_df['measurement_name'].values)
                combination_mnames[combination].update(new_mnames)
                # Check for overlaps
                overlapping_mnames.update(new_mnames.intersection(missing_mnames.get(unit, set())))

        # Identify missing_values rows: missing lower_limit or upper_limit, or missing both unit and type
        missing_mask = (~has_lower | ~has_upper) | (~has_unit & ~has_type)
        df_missing = df_batch[missing_mask].copy()
        if not df_missing.empty:
            # Log sample missing rows (first 100)
            if missing_samples < 100:
                sample_count = min(100 - missing_samples, len(df_missing))
                print(f"Sample missing rows (batch {row_counter // batch_size + 1}):\n{df_missing[['measurement_unit', 'measurement_type', 'lower_limit', 'upper_limit', 'measurement_name']].head(sample_count)}")
                missing_samples += sample_count
            # Group by measurement_name
            group = df_missing.groupby('measurement_name')
            for mname, group_df in group:
                count = len(group_df)
                missing_groups[mname] = missing_groups.get(mname, 0) + count
                if mname not in missing_mnames:
                    missing_mnames[mname] = {mname}
                # Store mode of measurement_unit, measurement_type, lower_limit, upper_limit
                unit_mode = group_df['measurement_unit'].mode().iloc[0] if not group_df['measurement_unit'].mode().empty else "unlabeled"
                type_mode = group_df['measurement_type'].mode().iloc[0] if not group_df['measurement_type'].mode().empty else "unlabeled"
                lower_mode = group_df['lower_limit'].mode().iloc[0] if not group_df['lower_limit'].mode().empty else -999.0
                upper_mode = group_df['upper_limit'].mode().iloc[0] if not group_df['upper_limit'].mode().empty else -999.0
                missing_values_details[mname] = (unit_mode, type_mode, lower_mode, upper_mode)
                # Count unique units and types per measurement_name
                unit_count = group_df['measurement_unit'].nunique()
                type_count = group_df['measurement_type'].nunique()
                missing_multi_units_types[mname] = (unit_count, type_count)
                # Collect unique units and types
                missing_units.update(group_df['measurement_unit'].dropna().unique())
                missing_types.update(group_df['measurement_type'].dropna().unique())
                # Check for overlaps
                for combo_mnames in combination_mnames.values():
                    if mname in combo_mnames:
                        overlapping_mnames.add(mname)

        # Update timestamps
        if 'created_at' in df_batch.columns:
            batch_min = df_batch['created_at'].min()
            batch_max = df_batch['created_at'].max()
            if batch_min and (earliest_time is None or batch_min < earliest_time):
                earliest_time = batch_min
                earliest_row = df_batch[df_batch['created_at'] == batch_min].iloc[0].to_dict()
                earliest_row_number = batch_indices[df_batch['created_at'].idxmin()]
            if batch_max and (latest_time is None or batch_max > latest_time):
                latest_time = batch_max
                latest_row = df_batch[df_batch['created_at'] == batch_max].iloc[0].to_dict()
                latest_row_number = batch_indices[df_batch['created_at'].idxmax()]

        row_counter += batch_row_count
        pbar.update(batch_row_count)
        del df_batch

# Calculate unique measurement_name counts
all_mnames_count = len(all_mnames)
combination_mnames_set = set()
for mnames in combination_mnames.values():
    combination_mnames_set.update(mnames)
combination_mnames_count = len(combination_mnames_set)
missing_mnames_set = set()
for mnames in missing_mnames.values():
    missing_mnames_set.update(mnames)
missing_mnames_count = len(missing_mnames_set)
total_groups = len(combinations) + len(missing_groups)

# Count measurement_names with multiple units or types
multi_unit_mnames = sum(1 for unit_count, _ in missing_multi_units_types.values() if unit_count > 1)
multi_type_mnames = sum(1 for _, type_count in missing_multi_units_types.values() if type_count > 1)

print(f"Step 2 Complete: Found {len(combinations)} combinations, {len(missing_groups)} missing value groups, {all_mnames_count} unique measurement names")
print(f"Missing values: {multi_unit_mnames} measurement_names with multiple units, {multi_type_mnames} with multiple types")
print(f"Overlapping measurement_names (in both combinations and missing_values): {', '.join(sorted(overlapping_mnames)) if overlapping_mnames else 'None'}")

# Step 3: Prepare primary summary CSV
print("Step 3: Preparing primary summary CSV")
output_data = []

# Add combination groups
for combination, count in combinations.items():
    unit, mtype, lower, upper = combination
    mnames = ", ".join(sorted(combination_mnames[combination]))
    percentage = (count / total_rows) * 100 if total_rows > 0 else 0
    output_data.append({
        'group_name': combination_names[combination],
        'category': 'combination',
        'measurement_unit': unit,
        'measurement_type': mtype,
        'lower_limit': lower,
        'upper_limit': upper,
        'measurement_name': mnames,
        'row_count': count,
        'percentage': percentage
    })

# Add missing value groups
for mname, count in missing_groups.items():
    percentage = (count / total_rows) * 100 if total_rows > 0 else 0
    unit, mtype, lower, upper = missing_values_details.get(mname, ("unlabeled", "unlabeled", -999.0, -999.0))
    output_data.append({
        'group_name': mname,
        'category': 'missing_values',
        'measurement_unit': unit,
        'measurement_type': mtype,
        'lower_limit': lower,
        'upper_limit': upper,
        'measurement_name': mname,
        'row_count': count,
        'percentage': percentage
    })

# Sort by row_count descending
output_data.sort(key=lambda x: x['row_count'], reverse=True)

# Create DataFrame for primary CSV
df_summary = pd.DataFrame(output_data)

# Verify all rows are accounted for
total_counted_rows = sum(row['row_count'] for row in output_data)
total_percentage = sum(row['percentage'] for row in output_data)
verification_status = "successful" if total_counted_rows == total_rows and abs(total_percentage - 100.0) < 0.01 else "failed"

# Description for primary CSV header
unique_groups = [row['group_name'] for row in output_data]
unique_combinations_list = ", ".join([row['group_name'] for row in output_data if row['category'] == 'combination'])
unique_measurement_names = ", ".join(sorted(all_mnames))
unique_missing_mnames = ", ".join([row['group_name'] for row in output_data if row['category'] == 'missing_values'])
missing_units_list = ", ".join(sorted(missing_units)) if missing_units else "None"
missing_types_list = ", ".join(sorted(missing_types)) if missing_types else "None"
overlapping_mnames_list = ", ".join(sorted(overlapping_mnames)) if overlapping_mnames else "None"
description = f"""\
# Script Purpose:
# Processes measurements_non_labeled.parquet to group rows by combinations (valid lower_limit and upper_limit, and at least one of measurement_unit or measurement_type non-null) or by measurement_name (missing lower_limit/upper_limit or both measurement_unit and measurement_type).
# Assigns group names (combinations: 'unit,type,lower,upper'; missing_values: measurement_name).
# Generates a new parquet file with a 'group_name' column and rows sorted by group.
# Source File: measurements_non_labeled.parquet
# Total Rows in Source File: {total_rows}
# Unique Measurement Names (Count: {all_mnames_count}): {unique_measurement_names}
# Unique Measurement Names in Combinations (Count: {combination_mnames_count}): {', '.join(sorted(combination_mnames_set))}
# Unique Measurement Names in Missing Values (Count: {missing_mnames_count}): {', '.join(sorted(missing_mnames_set))}
# Total Unique Groups (Combinations + Missing Values): {total_groups}
# Unique Combinations: {unique_combinations_list}
# Unique Measurement Names (Missing Values): {unique_missing_mnames}
# Unique Measurement Units in Missing Values: {missing_units_list}
# Unique Measurement Types in Missing Values: {missing_types_list}
# Measurement Names with Multiple Units: {multi_unit_mnames}
# Measurement Names with Multiple Types: {multi_type_mnames}
# Overlapping Measurement Names (in Combinations and Missing Values): {overlapping_mnames_list}
# Sum of Row Counts: {total_counted_rows}
# Sum of Percentages: {total_percentage:.2f}%
# Verification Status: {verification_status} (all rows assigned to exactly one group)
# Earliest Time: {earliest_time}
# Earliest Time Row Number: {earliest_row_number}/{total_rows}
# Earliest Time Row: {earliest_row}
# Latest Time: {latest_time}
# Latest Time Row Number: {latest_row_number}/{total_rows}
# Latest Time Row: {latest_row}
"""

# Write primary summary CSV
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    df_summary.to_csv(f, index=False, lineterminator='\n')
print(f"Step 3 Complete: Primary summary CSV saved as {output_csv}")

# Step 4: Prepare secondary summary CSV for unique measurement names
print("Step 4: Preparing secondary summary CSV for unique measurement names")
mnames_data = []
for mname, count in all_mnames_counts.items():
    percentage = (count / total_rows) * 100 if total_rows > 0 else 0
    mnames_data.append({
        'measurement_name': mname,
        'row_count': count,
        'percentage': percentage
    })

# Sort by row_count descending
mnames_data.sort(key=lambda x: x['row_count'], reverse=True)

# Create DataFrame for secondary CSV
df_mnames = pd.DataFrame(mnames_data)

# Description for secondary CSV header
mnames_description = f"""\
# Script Purpose:
# Lists all unique measurement_name values from measurements_non_labeled.parquet with their row counts and percentages.
# Source File: measurements_non_labeled.parquet
# Total Rows in Source File: {total_rows}
# Total Unique Measurement Names: {all_mnames_count}
# Sum of Row Counts: {sum(row['row_count'] for row in mnames_data)}
# Sum of Percentages: {sum(row['percentage'] for row in mnames_data):.2f}%
"""

# Write secondary summary CSV
with open(output_mnames_csv, 'w', encoding='utf-8') as f:
    f.write(mnames_description)
    df_mnames.to_csv(f, index=False, lineterminator='\n')
print(f"Step 4 Complete: Secondary summary CSV saved as {output_mnames_csv}")

# Step 5: Generate new parquet file with group_name column
print("Step 5: Generating new parquet file with group_name column")
# Create new schema with group_name column
original_schema = parquet_file.schema_arrow
new_schema = pa.schema(list(original_schema) + [pa.field('group_name', pa.string())])

# Initialize parquet writer
parquet_writer = pq.ParquetWriter(output_parquet, new_schema, compression='snappy')

# Collect all rows with group_name and sort by group_name
all_rows = []
with tqdm(total=total_rows, desc="Processing parquet rows for output", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=True):
        df_batch = batch.to_pandas()
        # Clean and validate data
        df_batch['lower_limit'] = pd.to_numeric(df_batch['lower_limit'], errors='coerce')
        df_batch['upper_limit'] = pd.to_numeric(df_batch['upper_limit'], errors='coerce')
        df_batch['measurement_unit'] = df_batch['measurement_unit'].replace(['', ' ', 'null'], pd.NA)
        df_batch['measurement_type'] = df_batch['measurement_type'].replace(['', ' ', 'null'], pd.NA)
        # Assign group_name
        df_batch['group_name'] = None
        # Combination group
        has_lower = df_batch['lower_limit'].notna()
        has_upper = df_batch['upper_limit'].notna()
        has_unit = df_batch['measurement_unit'].notna()
        has_type = df_batch['measurement_type'].notna()
        combination_mask = has_lower & has_upper & (has_unit | has_type)
        df_combination = df_batch[combination_mask].copy()
        if not df_combination.empty:
            df_combination['measurement_unit'] = df_combination['measurement_unit'].fillna("unlabeled")
            df_combination['measurement_type'] = df_combination['measurement_type'].fillna("unlabeled")
            df_combination['group_name'] = df_combination.apply(
                lambda row: combination_names.get(
                    (row['measurement_unit'], row['measurement_type'], row['lower_limit'], row['upper_limit']),
                    "unknown"
                ),
                axis=1
            )
            df_batch.loc[combination_mask, 'group_name'] = df_combination['group_name']
        # Missing values group
        missing_mask = (~has_lower | ~has_upper) | (~has_unit & ~has_type)
        df_batch.loc[missing_mask, 'group_name'] = df_batch.loc[missing_mask, 'measurement_name'].fillna("unlabeled_measurement_name")
        # Convert lower_limit and upper_limit to strings to match input schema
        df_batch['lower_limit'] = df_batch['lower_limit'].astype(str).replace('nan', '')
        df_batch['upper_limit'] = df_batch['upper_limit'].astype(str).replace('nan', '')
        all_rows.append(df_batch)
        pbar.update(len(df_batch))

# Concatenate and sort by group_name
df_all = pd.concat(all_rows, ignore_index=True)
df_all = df_all.sort_values(by='group_name')

# Write sorted rows to parquet
table = pa.Table.from_pandas(df_all, schema=new_schema)
parquet_writer.write_table(table)
parquet_writer.close()

print(f"Step 5 Complete: New parquet file saved as {output_parquet}")

# Step 6: Verify total row count
print("Step 6: Verifying total row count")
processed_rows = 0
with tqdm(total=total_rows, desc="Verifying rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['measurement_unit'], use_threads=True):
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