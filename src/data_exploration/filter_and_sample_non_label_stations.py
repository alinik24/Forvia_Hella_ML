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
measurements_unique_ids_csv = os.path.join(output_dir, f"measurements_unique_ids_{timestamp}.csv")
bookings_non_labeled_parquet = os.path.join(output_dir, f"bookings_non_labeled.parquet")
bookings_summary_csv = os.path.join(output_dir, f"bookings_station_desc_summary_{timestamp}.csv")
bookings_unique_ids_csv = os.path.join(output_dir, f"bookings_unique_ids_{timestamp}.csv")

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
    station_desc_metrics = {station: {
        'original_total_rows': 0,
        'original_unique_rows': set(),
        'filtered_total_rows': 0,
        'filtered_unique_rows': set(),
        'first_row': None
    } for station in labeled_station_desc}
    non_labeled_row_count = 0
    original_min_time = None
    original_max_time = None
    filtered_min_time = None
    filtered_max_time = None
    original_min_time_row = None
    original_max_time_row = None
    filtered_min_time_row = None
    filtered_max_time_row = None
    original_min_time_row_number = None
    original_max_time_row_number = None
    filtered_min_time_row_number = None
    filtered_max_time_row_number = None
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
                        filtered_min_time_row = non_labeled_batch[non_labeled_batch[time_column] == batch_filtered_min].iloc[0].to_dict()
                        filtered_min_time_row_number = batch_indices[non_labeled_batch[time_column].idxmin()]
                    if batch_filtered_max and (filtered_max_time is None or batch_filtered_max > filtered_max_time):
                        filtered_max_time = batch_filtered_max
                        filtered_max_time_row = non_labeled_batch[non_labeled_batch[time_column] == batch_filtered_max].iloc[0].to_dict()
                        filtered_max_time_row_number = batch_indices[non_labeled_batch[time_column].idxmax()]

            # Process station_desc metrics
            for station, group in df_batch.groupby('station_desc'):
                if station not in station_desc_metrics:
                    station_desc_metrics[station] = {
                        'original_total_rows': 0,
                        'original_unique_rows': set(),
                        'filtered_total_rows': 0,
                        'filtered_unique_rows': set(),
                        'first_row': None
                    }
                station_desc_metrics[station]['original_total_rows'] += len(group)
                group_tuples = [tuple(row) for row in group.to_dict('records')]
                station_desc_metrics[station]['original_unique_rows'].update(group_tuples)
                if station not in labeled_station_desc:
                    station_desc_metrics[station]['filtered_total_rows'] += len(group)
                    station_desc_metrics[station]['filtered_unique_rows'].update(group_tuples)
                if station_desc_metrics[station]['first_row'] is None and not group.empty:
                    station_desc_metrics[station]['first_row'] = group.iloc[0].to_dict()

            # Update original timestamps
            if time_column in df_batch.columns:
                batch_min = df_batch[time_column].min()
                batch_max = df_batch[time_column].max()
                if batch_min and (original_min_time is None or batch_min < original_min_time):
                    original_min_time = batch_min
                    original_min_time_row = df_batch[df_batch[time_column] == batch_min].iloc[0].to_dict()
                    original_min_time_row_number = batch_indices[df_batch[time_column].idxmin()]
                if batch_max and (original_max_time is None or batch_max > original_max_time):
                    original_max_time = batch_max
                    original_max_time_row = df_batch[df_batch[time_column] == batch_max].iloc[0].to_dict()
                    original_max_time_row_number = batch_indices[df_batch[time_column].idxmax()]

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
        'filtered_max_time': filtered_max_time,
        'original_min_time_row': original_min_time_row,
        'original_max_time_row': original_max_time_row,
        'filtered_min_time_row': filtered_min_time_row,
        'filtered_max_time_row': filtered_max_time_row,
        'original_min_time_row_number': original_min_time_row_number,
        'original_max_time_row_number': original_max_time_row_number,
        'filtered_min_time_row_number': filtered_min_time_row_number,
        'filtered_max_time_row_number': filtered_max_time_row_number
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
    unique_station_desc = sorted(results['station_desc_metrics'].keys())
    first_rows_info = "\n".join([f"# station_desc: {station}\n# First Row: {results['station_desc_metrics'][station]['first_row']}" 
                                for station in unique_station_desc if results['station_desc_metrics'][station]['first_row']])

    # Add labeled section header
    output_data.append({
        'section': 'labeled',
        'station_desc': 'LABELED',
        'original_total_rows': '',
        'filtered_total_rows': '',
        'original_unique_rows': '',
        'filtered_unique_rows': ''
    })

    # Add all labeled station_desc
    for station in labeled_station_desc:
        metrics = results['station_desc_metrics'].get(station, {
            'original_total_rows': 0,
            'filtered_total_rows': 0,
            'original_unique_rows': 0,
            'filtered_unique_rows': 0
        })
        output_data.append({
            'section': 'labeled',
            'station_desc': station,
            'original_total_rows': metrics['original_total_rows'],
            'filtered_total_rows': metrics['filtered_total_rows'],
            'original_unique_rows': metrics['original_unique_rows'],
            'filtered_unique_rows': metrics['filtered_unique_rows']
        })

    # Add non-labeled section header
    output_data.append({
        'section': 'non-labeled',
        'station_desc': 'NON-LABELED',
        'original_total_rows': '',
        'filtered_total_rows': '',
        'original_unique_rows': '',
        'filtered_unique_rows': ''
    })

    # Add non-labeled station_desc
    for station in sorted(unique_station_desc):
        if station not in labeled_station_desc:
            metrics = results['station_desc_metrics'][station]
            output_data.append({
                'section': 'non-labeled',
                'station_desc': station,
                'original_total_rows': metrics['original_total_rows'],
                'filtered_total_rows': metrics['filtered_total_rows'],
                'original_unique_rows': metrics['original_unique_rows'],
                'filtered_unique_rows': metrics['filtered_unique_rows']
            })

    # Create DataFrame
    df_output = pd.DataFrame(output_data, columns=[
        'section', 'station_desc', 'original_total_rows', 'filtered_total_rows',
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
# Unique station_desc in Original File: {', '.join(unique_station_desc)}
# First Rows for Unique station_desc:
{first_rows_info}
# Earliest Time in Original File: {results['original_min_time']}
# Earliest Time Row Number: {results['original_min_time_row_number']}/{total_rows}
# Earliest Time Row: {results['original_min_time_row']}
# Latest Time in Original File: {results['original_max_time']}
# Latest Time Row Number: {results['original_max_time_row_number']}/{total_rows}
# Latest Time Row: {results['original_max_time_row']}
# Earliest Time in Filtered Parquet: {results['filtered_min_time']}
# Earliest Time Row Number: {results['filtered_min_time_row_number']}/{non_labeled_rows}
# Earliest Time Row: {results['filtered_min_time_row']}
# Latest Time in Filtered Parquet: {results['filtered_max_time']}
# Latest Time Row Number: {results['filtered_max_time_row_number']}/{non_labeled_rows}
# Latest Time Row: {results['filtered_max_time_row']}
"""
    with open(summary_csv_path, 'w', encoding='utf-8') as f:
        f.write(description)
        df_output.to_csv(f, index=False, lineterminator='\n')

# Prepare unique IDs CSV
def prepare_unique_ids_csv(unique_serial_ids, unique_booking_ids, file_type, unique_ids_csv_path):
    output_data = []

    # Add serial_number_id section
    output_data.append({
        'section': 'serial_number_id',
        'value': ''
    })
    for serial_id in sorted(unique_serial_ids):
        output_data.append({
            'section': 'serial_number_id',
            'value': serial_id
        })

    # Add booking_id section
    output_data.append({
        'section': 'booking_id',
        'value': ''
    })
    for booking_id in sorted(unique_booking_ids):
        output_data.append({
            'section': 'booking_id',
            'value': booking_id
        })

    # Create DataFrame
    df_output = pd.DataFrame(output_data, columns=['section', 'value'])

    # Description for CSV header
    description = f"""\
# Script Purpose:
# Lists all unique serial_number_id and booking_id values from {file_type}_single_line.parquet.
# Total Unique Serial Number IDs: {len(unique_serial_ids)}
# Total Unique Booking IDs: {len(unique_booking_ids)}
"""
    with open(unique_ids_csv_path, 'w', encoding='utf-8') as f:
        f.write(description)
        df_output.to_csv(f, index=False, lineterminator='\n')

# Generate outputs
prepare_summary_csv(measurements_results, "measurements", measurements_summary_csv)
prepare_summary_csv(bookings_results, "bookings", bookings_summary_csv)
prepare_unique_ids_csv(bookings_serial_ids, bookings_booking_ids, "bookings", bookings_unique_ids_csv)
prepare_unique_ids_csv(measurements_serial_ids, measurements_booking_ids, "measurements", measurements_unique_ids_csv)

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
print(f"Measurements: {measurements_non_labeled_parquet}, {measurements_summary_csv}, {measurements_unique_ids_csv}")
print(f"Bookings: {bookings_non_labeled_parquet}, {bookings_summary_csv}, {bookings_unique_ids_csv}")