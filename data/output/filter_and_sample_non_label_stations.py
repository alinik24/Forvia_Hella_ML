import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
bookings_file = os.path.join(base_path, "bookings_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Output file paths
measurements_non_labeled_parquet = os.path.join(output_dir, f"measurements_non_labeled.parquet")
measurements_summary_csv = os.path.join(output_dir, f"measurements_station_desc_summary_{timestamp}.csv")
bookings_non_labeled_parquet = os.path.join(output_dir, f"bookings_non_labeled.parquet")
bookings_summary_csv = os.path.join(output_dir, f"bookings_station_desc_summary_{timestamp}.csv")
unique_ids_csv = os.path.join(output_dir, f"unique_ids_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define labeled station_desc
labeled_station_desc = [
    'station b17ad0f9', 'station 5a3f0928', 'station 031c4441', 'station 14473147', 'station eef8a574',
    'station 9a991014', 'station 0bde46ac', 'station 4b1b68dd', 'station 507926e9', 'station c06ba294',
    'station e40d07f7', 'station 0d84218d', 'station 94ff9bdd', 'station a24958aa', 'station 0883123a',
    'station b94b00ce', 'station 8ce235cf', 'station 63afba48', 'station bc01f8be', 'station d6806593', 'station de579af6'
]

# Verify file existence
for file_path in [measurements_file, bookings_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at {file_path}. Please check the file path or name.")

# Initialize batch size
batch_size = 2000000

# Function to process a parquet file and generate filtered parquet
def process_file(file_path, file_type, time_column):
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    print(f"Processing {file_type} file with {total_rows} rows")

    # Initialize storage
    station_desc_metrics = {}
    non_labeled_row_count = 0
    original_min_time = None
    original_max_time = None
    filtered_min_time = None
    filtered_max_time = None
    row_counter = 0

    # Initialize parquet writer
    non_labeled_writer = pq.ParquetWriter(
        measurements_non_labeled_parquet if file_type == "measurements" else bookings_non_labeled_parquet,
        schema,
        compression='snappy'
    )

    with tqdm(total=total_rows, desc=f"Processing {file_type} rows", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=True):
            df_batch = batch.to_pandas()
            df_batch = df_batch.fillna("")
            batch_row_count = len(df_batch)
            batch_indices = range(row_counter, row_counter + batch_row_count)

            # Write non-labeled rows to parquet
            non_labeled_batch = df_batch[~df_batch['station_desc'].isin(labeled_station_desc)]
            if not non_labeled_batch.empty:
                non_labeled_writer.write_table(pa.Table.from_pandas(non_labeled_batch, schema=schema))
                non_labeled_row_count += len(non_labeled_batch)

                # Update filtered timestamps
                if time_column in non_labeled_batch.columns:
                    batch_filtered_min = non_labeled_batch[time_column].min()
                    batch_filtered_max = non_labeled_batch[time_column].max()
                    if batch_filtered_min and (filtered_min_time is None or batch_filtered_min < filtered_min_time):
                        filtered_min_time = batch_filtered_min
                    if batch_filtered_max and (filtered_max_time is None or batch_filtered_max > filtered_max_time):
                        filtered_max_time = batch_filtered_max

            # Process station_desc metrics
            for station, group in df_batch.groupby('station_desc'):
                if station not in station_desc_metrics:
                    station_desc_metrics[station] = {
                        'original_total_rows': 0,
                        'original_unique_rows': set(),
                        'filtered_total_rows': 0,
                        'filtered_unique_rows': set()
                    }
                station_desc_metrics[station]['original_total_rows'] += len(group)
                group_tuples = [tuple(row) for row in group.to_dict('records')]
                station_desc_metrics[station]['original_unique_rows'].update(group_tuples)
                if station not in labeled_station_desc:
                    station_desc_metrics[station]['filtered_total_rows'] += len(group)
                    station_desc_metrics[station]['filtered_unique_rows'].update(group_tuples)

            # Update original timestamps
            if time_column in df_batch.columns:
                batch_min = df_batch[time_column].min()
                batch_max = df_batch[time_column].max()
                if batch_min and (original_min_time is None or batch_min < original_min_time):
                    original_min_time = batch_min
                if batch_max and (original_max_time is None or batch_max > original_max_time):
                    original_max_time = batch_max

            row_counter += batch_row_count
            pbar.update(batch_row_count)
            del df_batch

    non_labeled_writer.close()

    # Finalize unique row counts
    for station in station_desc_metrics:
        station_desc_metrics[station]['original_unique_rows'] = len(station_desc_metrics[station]['original_unique_rows'])
        station_desc_metrics[station]['filtered_unique_rows'] = len(station_desc_metrics[station]['filtered_unique_rows'])

    return {
        'total_rows': total_rows,
        'non_labeled_rows': non_labeled_row_count,
        'station_desc_metrics': station_desc_metrics,
        'original_min_time': original_min_time,
        'original_max_time': original_max_time,
        'filtered_min_time': filtered_min_time,
        'filtered_max_time': filtered_max_time
    }

# Function to collect unique IDs
def collect_unique_ids(file_path, file_type):
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Collecting unique IDs from {file_type} file with {total_rows} rows")

    unique_serial_ids = set()
    unique_booking_ids = set()

    with tqdm(total=total_rows, desc=f"Processing {file_type} rows for IDs", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['serial_number_id', 'booking_id'], use_threads=True):
            df_batch = batch.to_pandas()
            df_batch = df_batch.fillna("")

            unique_serial_ids.update(df_batch['serial_number_id'].dropna().astype(str))
            unique_booking_ids.update(df_batch['booking_id'].dropna().astype(str))

            pbar.update(len(df_batch))
            del df_batch

    return unique_serial_ids, unique_booking_ids

# Process files
measurements_results = process_file(measurements_file, "measurements", "created_at")
bookings_results = process_file(bookings_file, "bookings", "book_stamp")

# Collect unique IDs
bookings_serial_ids, bookings_booking_ids = collect_unique_ids(bookings_file, "bookings")
measurements_serial_ids, measurements_booking_ids = collect_unique_ids(measurements_file, "measurements")

# Prepare summary CSV
def prepare_summary_csv(results, file_type, summary_csv_path):
    output_data = []
    total_rows = results['total_rows']
    non_labeled_rows = results['non_labeled_rows']

    # Add station_desc rows
    for station in sorted(results['station_desc_metrics'].keys()):
        metrics = results['station_desc_metrics'][station]
        output_data.append({
            'station_desc': station,
            'original_total_rows': metrics['original_total_rows'],
            'filtered_total_rows': metrics['filtered_total_rows'],
            'original_unique_rows': metrics['original_unique_rows'],
            'filtered_unique_rows': metrics['filtered_unique_rows']
        })

    # Create DataFrame
    df_output = pd.DataFrame(output_data, columns=[
        'station_desc', 'original_total_rows', 'filtered_total_rows',
        'original_unique_rows', 'filtered_unique_rows'
    ])

    # Description for CSV header
    description = f"""\
# Script Purpose:
# Processes {file_type}_single_line.parquet to generate a non-labeled parquet file and a summary CSV with station_desc metrics.
# Filters out rows with labeled station_desc values, keeping only non-labeled rows.
# Labeled station_desc: {', '.join(labeled_station_desc)}
# Total Rows in Original File: {total_rows}
# Total Rows in Filtered Parquet: {non_labeled_rows}
# Earliest Time in Original File: {results['original_min_time']}
# Latest Time in Original File: {results['original_max_time']}
# Earliest Time in Filtered Parquet: {results['filtered_min_time']}
# Latest Time in Filtered Parquet: {results['filtered_max_time']}
"""
    with open(summary_csv_path, 'w', encoding='utf-8') as f:
        f.write(description)
        df_output.to_csv(f, index=False, lineterminator='\n')

# Prepare unique IDs CSV
def prepare_unique_ids_csv():
    output_data = []

    # Add bookings unique IDs
    all_bookings_serial_ids = sorted(bookings_serial_ids)
    all_bookings_booking_ids = sorted(bookings_booking_ids)
    max_len_bookings = max(len(all_bookings_serial_ids), len(all_bookings_booking_ids))
    for i in range(max_len_bookings):
        output_data.append({
            'file': 'bookings',
            'serial_number_id': all_bookings_serial_ids[i] if i < len(all_bookings_serial_ids) else '',
            'booking_id': all_bookings_booking_ids[i] if i < len(all_bookings_booking_ids) else ''
        })

    # Add measurements unique IDs
    all_measurements_serial_ids = sorted(measurements_serial_ids)
    all_measurements_booking_ids = sorted(measurements_booking_ids)
    max_len_measurements = max(len(all_measurements_serial_ids), len(all_measurements_booking_ids))
    for i in range(max_len_measurements):
        output_data.append({
            'file': 'measurements',
            'serial_number_id': all_measurements_serial_ids[i] if i < len(all_measurements_serial_ids) else '',
            'booking_id': all_measurements_booking_ids[i] if i < len(all_measurements_booking_ids) else ''
        })

    # Create DataFrame
    df_output = pd.DataFrame(output_data, columns=['file', 'serial_number_id', 'booking_id'])

    # Description for CSV header
    description = f"""\
# Script Purpose:
# Lists all unique serial_number_id and booking_id values from bookings_single_line.parquet and measurements_single_line.parquet.
# Total Unique Serial IDs in Bookings: {len(bookings_serial_ids)}
# Total Unique Booking IDs in Bookings: {len(bookings_booking_ids)}
# Total Unique Serial IDs in Measurements: {len(measurements_serial_ids)}
# Total Unique Booking IDs in Measurements: {len(measurements_booking_ids)}
"""
    with open(unique_ids_csv, 'w', encoding='utf-8') as f:
        f.write(description)
        df_output.to_csv(f, index=False, lineterminator='\n')

# Generate outputs
prepare_summary_csv(measurements_results, "measurements", measurements_summary_csv)
prepare_summary_csv(bookings_results, "bookings", bookings_summary_csv)
prepare_unique_ids_csv()

# Verify total row counts
for file_path, file_type in [(measurements_file, "measurements"), (bookings_file, "bookings")]:
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    processed_rows = 0
    with tqdm(total=total_rows, desc=f"Verifying {file_type} rows", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['station_desc'], use_threads=True):
            df_batch = batch.to_pandas()
            processed_rows += len(df_batch)
            pbar.update(len(df_batch))
            del df_batch

    print(f"Processed rows for {file_type}: {processed_rows}")
    print(f"Total rows in {file_type} file: {total_rows}")
    if processed_rows == total_rows:
        print(f"Verification successful for {file_type}: All rows processed.")
    else:
        print(f"Verification failed for {file_type}: Processed {processed_rows} rows, expected {total_rows}.")

print(f"Processing complete. Results saved to:")
print(f"Measurements: {measurements_non_labeled_parquet}, {measurements_summary_csv}")
print(f"Bookings: {bookings_non_labeled_parquet}, {bookings_summary_csv}")
print(f"Unique IDs CSV: {unique_ids_csv}")