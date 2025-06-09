import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa

# Define paths
measurements_file = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output\measurements_non_labeled.parquet"
bookings_file = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output\bookings_non_labeled.parquet"
output_dir = r"C:\Desktop\Research and Thesis\RWML projects\data_hella\output"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_file = os.path.join(output_dir, f"joined_measurements_bookings_{timestamp}.parquet")
unmatched_measurements_csv = os.path.join(output_dir, f"unmatched_measurements_{timestamp}.csv")
unmatched_bookings_csv = os.path.join(output_dir, f"unmatched_bookings_{timestamp}.csv")
summary_file = os.path.join(output_dir, f"join_summary_{timestamp}.csv")
timestamp_comparison_csv = os.path.join(output_dir, f"timestamp_comparison_{timestamp}.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Verify file existence
for file_path in [measurements_file, bookings_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at {file_path}. Please check the file path or name.")

# Initialize batch size
batch_size = 2000000

# Function to count total rows
def count_rows(file_path, file_type):
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in {file_type} file: {total_rows}")
    return total_rows

# Count rows in both files
measurements_total_rows = count_rows(measurements_file, "measurements")
bookings_total_rows = count_rows(bookings_file, "bookings")

# Function to perform left join and track unmatched rows
def perform_left_join(measurements_path, bookings_path):
    measurements_parquet = pq.ParquetFile(measurements_path)
    bookings_parquet = pq.ParquetFile(bookings_path)
    
    # Initialize result lists
    timestamp_comparison_data = []
    joined_row_count = 0
    unmatched_measurements_count = 0
    unmatched_bookings_count = 0
    unmatched_measurements_rows = []
    
    # Get schemas
    measurements_schema = measurements_parquet.schema_arrow
    bookings_schema = bookings_parquet.schema_arrow
    
    # Create combined schema: all measurements columns + all bookings columns with _b suffix (except join keys)
    combined_fields = measurements_schema
    for field in bookings_schema:
        if field.name not in ['serial_number_id', 'booking_id']:
            combined_fields = combined_fields.append(pa.field(f"{field.name}_b", field.type))
    
    # Initialize writer for joined output
    joined_writer = pq.ParquetWriter(result_file, combined_fields, compression='snappy')
    
    # Track earliest and latest dates
    earliest_measurements = {'time': None, 'row': None, 'row_number': None}
    latest_measurements = {'time': None, 'row': None, 'row_number': None}
    earliest_bookings = {'time': None, 'row': None, 'row_number': None}
    latest_bookings = {'time': None, 'row': None, 'row_number': None}
    earliest_joined = {'time': None, 'row': None, 'row_number': None}
    latest_joined = {'time': None, 'row': None, 'row_number': None}
    row_counter = 0
    
    # Load bookings into memory, renaming non-join columns with _b suffix
    bookings_df = pd.read_parquet(bookings_path)
    bookings_df = bookings_df.rename(columns={col: f"{col}_b" for col in bookings_df.columns if col not in ['serial_number_id', 'booking_id']})
    bookings_df = bookings_df.set_index(['serial_number_id', 'booking_id'])
    
    # Track matched booking indices for unmatched bookings
    matched_booking_indices = set()
    
    with tqdm(total=measurements_total_rows, desc="Processing measurements for join", unit="rows") as pbar:
        for batch in measurements_parquet.iter_batches(batch_size=batch_size, use_threads=True):
            df_batch = batch.to_pandas()
            batch_row_count = len(df_batch)
            batch_indices = range(row_counter, row_counter + batch_row_count)
            
            # Update earliest and latest measurements time
            if 'created_at' in df_batch.columns:
                batch_min = df_batch['created_at'].min()
                batch_max = df_batch['created_at'].max()
                if batch_min and (earliest_measurements['time'] is None or batch_min < earliest_measurements['time']):
                    earliest_measurements['time'] = batch_min
                    earliest_measurements['row'] = df_batch[df_batch['created_at'] == batch_min].iloc[0].to_dict()
                    earliest_measurements['row_number'] = batch_indices[df_batch['created_at'].idxmin()]
                if batch_max and (latest_measurements['time'] is None or batch_max > latest_measurements['time']):
                    latest_measurements['time'] = batch_max
                    latest_measurements['row'] = df_batch[df_batch['created_at'] == batch_max].iloc[0].to_dict()
                    latest_measurements['row_number'] = batch_indices[df_batch['created_at'].idxmax()]
            
            # Perform join
            df_batch_indexed = df_batch.set_index(['serial_number_id', 'booking_id'])
            joined_batch = df_batch_indexed.join(bookings_df, how='left')
            
            # Separate matched and unmatched
            matched = joined_batch[joined_batch.index.isin(bookings_df.index)]
            unmatched = joined_batch[~joined_batch.index.isin(bookings_df.index)]
            
            # Track matched booking indices
            matched_booking_indices.update(matched.index)
            
            # Collect timestamp comparison for matched rows
            if not matched.empty and 'created_at' in matched.columns and 'book_stamp_b' in matched.columns:
                for idx, row in matched.iterrows():
                    timestamp_comparison_data.append({
                        'serial_number_id': idx[0],
                        'booking_id': idx[1],
                        'created_at': row['created_at'],
                        'book_stamp': row['book_stamp_b']
                    })
            
            # Reset index for writing
            matched = matched.reset_index()
            unmatched = unmatched.reset_index()
            
            # Write matched rows
            if not matched.empty:
                joined_writer.write_table(pa.Table.from_pandas(matched, schema=combined_fields))
                joined_row_count += len(matched)
                
                # Update earliest and latest joined time
                if 'created_at' in matched.columns:
                    batch_joined_min = matched['created_at'].min()
                    batch_joined_max = matched['created_at'].max()
                    if batch_joined_min and (earliest_joined['time'] is None or batch_joined_min < earliest_joined['time']):
                        earliest_joined['time'] = batch_joined_min
                        earliest_joined['row'] = matched[matched['created_at'] == batch_joined_min].iloc[0].to_dict()
                        earliest_joined['row_number'] = batch_indices[matched['created_at'].idxmin()]
                    if batch_joined_max and (latest_joined['time'] is None or batch_joined_max > latest_joined['time']):
                        latest_joined['time'] = batch_joined_max
                        latest_joined['row'] = matched[matched['created_at'] == batch_joined_max].iloc[0].to_dict()
                        latest_joined['row_number'] = batch_indices[matched['created_at'].idxmax()]
            
            # Collect unmatched measurements rows
            if not unmatched.empty:
                unmatched_measurements_rows.append(unmatched.reset_index())
                unmatched_measurements_count += len(unmatched)
            
            row_counter += batch_row_count
            pbar.update(batch_row_count)
            del df_batch, joined_batch, matched, unmatched
    
    joined_writer.close()
    
    # Write unmatched measurements to CSV
    if unmatched_measurements_rows:
        unmatched_measurements_df = pd.concat(unmatched_measurements_rows, ignore_index=True)
        unmatched_measurements_df.to_csv(unmatched_measurements_csv, index=False, lineterminator='\n')
        del unmatched_measurements_df
    
    # Write unmatched bookings to CSV
    unmatched_bookings = bookings_df[~bookings_df.index.isin(matched_booking_indices)]
    unmatched_bookings_count = len(unmatched_bookings)
    if not unmatched_bookings.empty:
        unmatched_bookings = unmatched_bookings.reset_index()
        unmatched_bookings.to_csv(unmatched_bookings_csv, index=False, lineterminator='\n')
    
    # Get earliest and latest bookings time
    bookings_full_df = pd.read_parquet(bookings_path)  # Load full bookings for timestamp analysis
    if 'book_stamp' in bookings_full_df.columns:
        earliest_bookings['time'] = bookings_full_df['book_stamp'].min()
        earliest_bookings['row'] = bookings_full_df[bookings_full_df['book_stamp'] == earliest_bookings['time']].iloc[0].to_dict()
        earliest_bookings['row_number'] = bookings_full_df.index.get_loc(bookings_full_df[bookings_full_df['book_stamp'] == earliest_bookings['time']].index[0])
        latest_bookings['time'] = bookings_full_df['book_stamp'].max()
        latest_bookings['row'] = bookings_full_df[bookings_full_df['book_stamp'] == latest_bookings['time']].iloc[0].to_dict()
        latest_bookings['row_number'] = bookings_full_df.index.get_loc(bookings_full_df[bookings_full_df['book_stamp'] == latest_bookings['time']].index[0])
    
    return joined_row_count, unmatched_measurements_count, unmatched_bookings_count, earliest_measurements, latest_measurements, earliest_bookings, latest_bookings, earliest_joined, latest_joined, timestamp_comparison_data

# Perform join
joined_count, unmatched_measurements_count, unmatched_bookings_count, earliest_measurements, latest_measurements, earliest_bookings, latest_bookings, earliest_joined, latest_joined, timestamp_comparison_data = perform_left_join(measurements_file, bookings_file)

# Get 5 rows from result file
result_peek = pd.read_parquet(result_file, engine='pyarrow').head(5).to_dict('records')

# Prepare timestamp comparison CSV
timestamp_comparison_df = pd.DataFrame(timestamp_comparison_data)
description = f"""\
# Script Purpose:
# Compares created_at from measurements and book_stamp from bookings for matched rows.
# Total Matched Rows: {len(timestamp_comparison_data)}
"""
with open(timestamp_comparison_csv, 'w', encoding='utf-8') as f:
    f.write(description)
    timestamp_comparison_df.to_csv(f, index=False, lineterminator='\n')

# Prepare summary CSV
summary_data = {
    'Metric': [
        'Total Measurements Rows',
        'Total Bookings Rows',
        'Joined Rows',
        'Unmatched Measurements Rows',
        'Unmatched Bookings Rows',
        'Earliest Measurements Time',
        'Earliest Measurements Row Number',
        'Latest Measurements Time',
        'Latest Measurements Row Number',
        'Earliest Bookings Time',
        'Earliest Bookings Row Number',
        'Latest Bookings Time',
        'Latest Bookings Row Number',
        'Earliest Joined Time',
        'Earliest Joined Row Number',
        'Latest Joined Time',
        'Latest Joined Row Number'
    ],
    'Value': [
        measurements_total_rows,
        bookings_total_rows,
        joined_count,
        unmatched_measurements_count,
        unmatched_bookings_count,
        earliest_measurements['time'],
        f"{earliest_measurements['row_number']}/{measurements_total_rows}" if earliest_measurements['row_number'] is not None else None,
        latest_measurements['time'],
        f"{latest_measurements['row_number']}/{measurements_total_rows}" if latest_measurements['row_number'] is not None else None,
        earliest_bookings['time'],
        f"{earliest_bookings['row_number']}/{bookings_total_rows}" if earliest_bookings['row_number'] is not None else None,
        latest_bookings['time'],
        f"{latest_bookings['row_number']}/{bookings_total_rows}" if latest_bookings['row_number'] is not None else None,
        earliest_joined['time'],
        f"{earliest_joined['row_number']}/{joined_count}" if earliest_joined['row_number'] is not None else None,
        latest_joined['time'],
        f"{latest_joined['row_number']}/{joined_count}" if latest_joined['row_number'] is not None else None
    ]
}

# Add rows as strings to avoid CSV formatting issues
for metric, row in [
    ('Earliest Measurements Row', earliest_measurements['row']),
    ('Latest Measurements Row', latest_measurements['row']),
    ('Earliest Bookings Row', earliest_bookings['row']),
    ('Latest Bookings Row', latest_bookings['row']),
    ('Earliest Joined Row', earliest_joined['row']),
    ('Latest Joined Row', latest_joined['row'])
]:
    summary_data['Metric'].append(metric)
    summary_data['Value'].append(str(row))

# Add peek of result dataset
summary_data['Metric'].append('Result Dataset Peek (First 5 Rows)')
summary_data['Value'].append(str(result_peek))

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Write summary CSV
description = f"""\
# Script Purpose:
# Performs a left join between measurements_non_labeled.parquet and bookings_non_labeled.parquet
# based on serial_number_id and booking_id, keeping one copy of matching columns (from measurements).
# Bookings columns (except join keys) are suffixed with _b.
# Outputs joined parquet, unmatched measurements CSV, unmatched bookings CSV, timestamp comparison CSV, and this summary CSV.
# Join Results:
# Total Measurements Rows: {measurements_total_rows}
# Total Bookings Rows: {bookings_total_rows}
# Joined Rows: {joined_count}
# Unmatched Measurements Rows: {unmatched_measurements_count}
# Unmatched Bookings Rows: {unmatched_bookings_count}
# Output Files:
# Joined Result: {result_file}
# Unmatched Measurements CSV: {unmatched_measurements_csv}
# Unmatched Bookings CSV: {unmatched_bookings_csv}
# Timestamp Comparison: {timestamp_comparison_csv}
"""
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(description)
    summary_df.to_csv(f, index=False, lineterminator='\n')

print(f"Processing complete. Results saved to:")
print(f"Joined Result: {result_file}")
print(f"Unmatched Measurements CSV: {unmatched_measurements_csv}")
print(f"Unmatched Bookings CSV: {unmatched_bookings_csv}")
print(f"Timestamp Comparison: {timestamp_comparison_csv}")
print(f"Summary: {summary_file}")