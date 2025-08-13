"""
Script: extract_serial_number_data.py

Purpose:
--------
This script processes large production datasets (`bookings`, `measurements`, `materials` in Parquet format),
selects a random sample of `serial_number_id` values, and extracts all related rows in chunks. It also
deduplicates records, counts duplicates, and generates structured outputs for analysis.

Key Features:
-------------
- Random sampling of `serial_number_id` values from `bookings` data.
- Efficient batch processing using `pyarrow` to minimize memory usage.
- Chunk-wise extraction of related rows across all three datasets.
- Deduplication and duplicate counting per serial number.
- Generation of three outputs:
  1. **Detailed Parquet**: All rows per serial ID across all files.
  2. **Summary CSV**: Counts of matched and duplicate rows per serial ID.
  3. **Duplicates CSV**: Grouped and readable view of duplicate records.

Usage:
------
Configure the following before running:
- `base_path`: Folder containing the three Parquet files.
- `timestamp`: Unique identifier used in output filenames.
- Optional: `num_serial_ids` and `chunk_size` to control processing scale.

Output:
-------
- `output_<timestamp>_detailed.parquet`
- `output_<timestamp>_summary.csv`
- `output_<timestamp>_duplicates.csv`

Designed for high-volume SMT or production line traceability datasets.
"""
import os
import random
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def process_data_with_timespan(base_path, timestamp_str, num_serial_ids=10000, chunk_size=10000):
    """
    Process Parquet files with a given timestamp for file naming, with progress tracking and chunked processing.
    
    Args:
        base_path (str): Directory path where Parquet files are located
        timestamp_str (str): Timestamp string in format 'YYYYMMDD_HHMM'
        num_serial_ids (int): Number of serial number IDs to sample (default: 100000)
        chunk_size (int): Number of serial number IDs to process per chunk (default: 20000)
    """
    print(f"\nStarting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with timestamp {timestamp_str}")

    # File paths for the three Parquet files
    bookings_file = os.path.join(base_path, "bookings_single_line.parquet")
    measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
    materials_file = os.path.join(base_path, "materials_single_line.parquet")

    # Output prefix with provided timestamp
    output_prefix = os.path.join(base_path, f"output_{timestamp_str}")

    # Define column lists for each Parquet file
    bookings_columns = [
        "booking_id", "book_state", "workstep_number_mes", "sequence_number", "created_at", "updated_at",
        "book_stamp", "serial_number_id", "lot_id", "station_id", "station_diag_id", "workorder_id",
        "panel_position_number", "workorder_type", "station_number", "station_desc", "workorder_number",
        "workorder_desc", "object_id", "product_variant_id", "workplan_id", "part_number", "part_desc",
        "part_group", "has_failures", "erp_group_id", "workstep_number_erp", "workstep_desc", "workstep_id",
        "erp_group_desc", "line_id", "line_desc"
    ]

    measurements_columns = [
        "recipe_revision_id", "measure_step_number", "measure_value", "measure_fail_code", "updated_at",
        "created_at", "booking_id", "sequence_number", "book_state", "product_id", "product_variant_id",
        "part_number", "serial_number_id", "serial_number", "station_id", "station_number", "station_desc",
        "workstep_id", "measurement_name", "measurement_unit", "measurement_type", "lower_limit", "upper_limit",
        "teststep_id"
    ]

    materials_columns = [
        "station_desc", "sequence_number", "book_state", "serial_number_id", "serial_number", "workorder_id",
        "workorder_number", "workorder_desc", "part_number", "part_desc", "lot_packed_at", "panel_position",
        "supplier_id", "supplier_code", "supplier_name", "container_number", "supplier_order_date_code",
        "supplier_order_number", "supplier_order_desc", "mounting_place"
    ]

    # Step 1: Initialize sets and dictionaries for chunked processing
    print("Step 1: Initializing data structures")
    all_serial_numbers = set()

    # Batch size for reading Parquet files
    batch_size = 10000

    # Step 2: Read bookings file to get random serial number IDs
    print(f"Step 2: Collecting unique serial_number_id values from bookings for {num_serial_ids} IDs")
    parquet_file = pq.ParquetFile(bookings_file)
    total_rows = parquet_file.metadata.num_rows
    with tqdm(total=total_rows, desc="Processing bookings batches", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            chunk = batch.to_pandas()
            serial_numbers = chunk['serial_number_id'].unique()
            all_serial_numbers.update(serial_numbers)
            pbar.update(len(chunk))

    # Ensure there are enough serial numbers to sample
    if len(all_serial_numbers) < num_serial_ids:
        raise ValueError(
            f"Not enough unique serial_number_id values in bookings file. Found {len(all_serial_numbers)}, required {num_serial_ids}.")
    random_serial_numbers = random.sample(list(all_serial_numbers), num_serial_ids)
    print(f"Step 2 Complete: Selected {len(random_serial_numbers)} serial_number_id values")

    # Step 3: Collect all rows for each serial number ID in chunks
    all_columns = ["Parquet_file", "serial_number_id"] + bookings_columns + measurements_columns + materials_columns + [
        "duplicate_count"]
    detailed_parquet = f"{output_prefix}_detailed.parquet"
    summary = {}
    duplicates = []

    # Split serial numbers into chunks
    serial_number_chunks = [random_serial_numbers[i:i + chunk_size] for i in
                            range(0, len(random_serial_numbers), chunk_size)]

    for chunk_idx, serial_chunk in enumerate(
            tqdm(serial_number_chunks, desc="Processing serial number chunks", unit="chunk")):
        print(
            f"Step 3: Processing chunk {chunk_idx + 1}/{len(serial_number_chunks)} ({len(serial_chunk)} serial numbers)")

        # Initialize data structures for this chunk
        bookings_data = []
        measurements_data = []
        materials_data = []

        # Helper function to collect rows for a chunk
        def collect_rows(batch, target_list, columns, file_name, pbar):
            chunk = batch.to_pandas()
            if 'serial_number_id' in chunk.columns:
                filtered_chunk = chunk[chunk['serial_number_id'].isin(serial_chunk)]
                if not filtered_chunk.empty:
                    target_list.extend(filtered_chunk.to_dict('records'))
            pbar.update(len(chunk))

        # Process bookings
        print(f"Step 3a: Collecting bookings data for chunk {chunk_idx + 1}")
        parquet_file = pq.ParquetFile(bookings_file)
        total_rows = parquet_file.metadata.num_rows
        with tqdm(total=total_rows, desc="Processing bookings", unit="rows") as pbar:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                collect_rows(batch, bookings_data, bookings_columns, "bookings", pbar)

        # Process measurements
        print(f"Step 3b: Collecting measurements data for chunk {chunk_idx + 1}")
        parquet_file = pq.ParquetFile(measurements_file)
        total_rows = parquet_file.metadata.num_rows
        with tqdm(total=total_rows, desc="Processing measurements", unit="rows") as pbar:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                collect_rows(batch, measurements_data, measurements_columns, "measurements", pbar)

        # Process materials
        print(f"Step 3c: Collecting materials data for chunk {chunk_idx + 1}")
        parquet_file = pq.ParquetFile(materials_file)
        total_rows = parquet_file.metadata.num_rows
        with tqdm(total=total_rows, desc="Processing materials", unit="rows") as pbar:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                collect_rows(batch, materials_data, materials_columns, "materials", pbar)

        # Convert lists of dictionaries to DataFrames
        print(f"Step 3d: Converting collected data to DataFrames for chunk {chunk_idx + 1}")
        bookings_df = pd.DataFrame(bookings_data)[bookings_columns]
        measurements_df = pd.DataFrame(measurements_data)[measurements_columns]
        materials_df = pd.DataFrame(materials_data)[materials_columns]

        # Step 4: Deduplicate within each DataFrame and count duplicates
        print(f"Step 4: Deduplicating data and counting duplicates for chunk {chunk_idx + 1}")

        def deduplicate_and_count(df, file_name):
            if df.empty:
                return df, {}
            duplicates_count = {}
            dedup_df = df.drop_duplicates(keep='first')
            duplicates = df[df.duplicated(keep=False)]
            for sn in tqdm(serial_chunk, desc=f"Counting duplicates in {file_name}", unit="serial_number_id"):
                duplicates_count[sn] = len(duplicates[duplicates['serial_number_id'] == sn])
            return dedup_df, duplicates_count

        bookings_df_dedup, bookings_dup_counts = deduplicate_and_count(bookings_df, "bookings")
        measurements_df_dedup, measurements_dup_counts = deduplicate_and_count(measurements_df, "measurements")
        materials_df_dedup, materials_dup_counts = deduplicate_and_count(materials_df, "materials")

        # Collect duplicates for this chunk
        if not bookings_df_dedup.empty and len(bookings_df) > len(bookings_df_dedup):
            duplicates.extend(
                bookings_df[bookings_df.duplicated(keep=False)].assign(source_file="bookings").to_dict('records'))
        if not measurements_df_dedup.empty and len(measurements_df) > len(measurements_df_dedup):
            duplicates.extend(
                measurements_df[measurements_df.duplicated(keep=False)].assign(source_file="measurements").to_dict(
                    'records'))
        if not materials_df_dedup.empty and len(materials_df) > len(materials_df_dedup):
            duplicates.extend(
                materials_df[materials_df.duplicated(keep=False)].assign(source_file="materials").to_dict('records'))

        # Step 5: Prepare detailed output for this chunk
        print(f"Step 5: Preparing detailed Parquet for chunk {chunk_idx + 1}")
        output_rows = []

        # Helper function to create a row with empty values for other sections
        def create_row(file_name, sn, row, file_columns, dup_count, return_string=False):
            row_data = {'Parquet_file': file_name, 'serial_number_id': str(sn)}
            for col in bookings_columns + measurements_columns + materials_columns:
                row_data[col] = ''
            for col in file_columns:
                row_data[col] = str(row.get(col, ''))
            row_data['duplicate_count'] = str(dup_count.get(sn, 0))
            if return_string:
                return "|".join([row_data.get(col, '') for col in all_columns])
            return row_data

        for sn in tqdm(serial_chunk, desc="Processing serial numbers for detailed output", unit="serial_number_id"):
            bookings_rows = bookings_df_dedup[bookings_df_dedup['serial_number_id'] == sn]
            for _, row in bookings_rows.iterrows():
                output_rows.append(create_row("bookings", sn, row, bookings_columns, bookings_dup_counts))

            measurements_rows = measurements_df_dedup[measurements_df_dedup['serial_number_id'] == sn]
            for _, row in measurements_rows.iterrows():
                output_rows.append(create_row("measurements", sn, row, measurements_columns, measurements_dup_counts))

            materials_rows = materials_df_dedup[materials_df_dedup['serial_number_id'] == sn]
            for _, row in materials_rows.iterrows():
                output_rows.append(create_row("materials", sn, row, materials_columns, materials_dup_counts))

            # Update summary for this serial number
            summary[sn] = {
                "bookings": len(bookings_df_dedup[bookings_df_dedup['serial_number_id'] == sn]),
                "measurements": len(measurements_df_dedup[measurements_df_dedup['serial_number_id'] == sn]),
                "materials": len(materials_df_dedup[materials_df_dedup['serial_number_id'] == sn]),
                "bookings.dup": bookings_dup_counts.get(sn, 0),
                "measurements.dup": measurements_dup_counts.get(sn, 0),
                "materials.dup": materials_dup_counts.get(sn, 0)
            }

        # Append to detailed Parquet
        chunk_df = pd.DataFrame(output_rows)
        if chunk_idx == 0:
            chunk_df.to_parquet(detailed_parquet, engine='pyarrow', index=False)
        else:
            chunk_df.to_parquet(detailed_parquet, engine='pyarrow', index=False, partition_cols=None, mode='append')
        print(f"Step 5: Chunk {chunk_idx + 1} appended to {detailed_parquet}")

    print("Step 5 Complete: Detailed Parquet saved")

    # Step 6: Save summary as CSV
    print("Step 6: Building summary CSV")
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_csv = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, encoding='utf-8')
    print(f"Step 6 Complete: Summary CSV saved as {summary_csv}")

    # Step 7: Save duplicates with detailed breakdown
    print("Step 7: Processing duplicates")
    duplicates_csv = f"{output_prefix}_duplicates.csv"
    if duplicates:
        duplicates_df = pd.DataFrame(duplicates)
        grouped_duplicates = duplicates_df.groupby(['serial_number_id', 'source_file'])
        duplicates_summary = [("|".join(all_columns[:-1]))]  # Header without duplicate_count
        for (sn, source_file), group in tqdm(grouped_duplicates, desc="Writing duplicates", unit="group"):
            duplicates_summary.append(
                f"Serial Number ID: {sn}, Source File: {source_file}, Duplicate Rows: {len(group)}")
            for _, row in group.iterrows():
                row_dict = row.drop('source_file').to_dict()
                duplicates_summary.append(create_row(source_file, sn, row_dict,
                                                     bookings_columns if source_file == "bookings" else
                                                     measurements_columns if source_file == "measurements" else materials_columns,
                                                     {}, return_string=True))
            duplicates_summary.append("")  # Empty line for readability
        with open(duplicates_csv, 'w', encoding='utf-8') as f:
            f.write("\n".join(duplicates_summary))
    else:
        with open(duplicates_csv, 'w', encoding='utf-8') as f:
            f.write("No duplicates found.")
    print(f"Step 7 Complete: Duplicates CSV saved as {duplicates_csv}")

    print(f"\nProcessing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Files saved as:")
    print(f"- Detailed: {detailed_parquet}")
    print(f"- Summary: {summary_csv}")
    print(f"- Duplicates: {duplicates_csv}")


# Example usage with current date and time
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
timestamp = "20250525_1043"  # Current date and time: May 25, 2025, 10:43 AM
process_data_with_timespan(base_path, timestamp, num_serial_ids=5000, chunk_size=10000)
