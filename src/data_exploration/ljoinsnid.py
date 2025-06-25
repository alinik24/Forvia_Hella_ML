import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from pathlib import Path
import gc

# Define file paths
data_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
bookings_file = os.path.join(data_path, "bookings_single_line.parquet")
measurements_file = os.path.join(data_path, "measurements_single_line.parquet")
materials_file = os.path.join(data_path, "materials_single_line.parquet")
output_parquet = os.path.join(data_path, "joined_serial_number_id.parquet")
output_summary = os.path.join(data_path, "summary_serial_number_id.csv")

# Read metadata to get row counts and schemas
bookings_pq = pq.ParquetFile(bookings_file)
measurements_pq = pq.ParquetFile(measurements_file)
materials_pq = pq.ParquetFile(materials_file)
bookings_rows = bookings_pq.metadata.num_rows
measurements_rows = measurements_pq.metadata.num_rows
materials_rows = materials_pq.metadata.num_rows
bookings_columns = bookings_pq.schema.names
measurements_columns = measurements_pq.schema.names
materials_columns = materials_pq.schema.names

# Initialize output Parquet writer
schema = None  # Will be set after first join
writer = None
merged_rows = 0
missing_values = {}
preview_rows = []
preview_count = 5

# Load bookings fully (assuming it's smaller) and keep all columns
bookings = bookings_pq.read().to_pandas()

# Process measurements in chunks
measurements_reader = pq.ParquetFile(measurements_file)
for i in range(measurements_reader.num_row_groups):
    chunk = measurements_reader.read_row_group(i).to_pandas()
    # Perform left join with bookings
    merged_chunk = bookings.merge(chunk, on="serial_number_id", how="left", suffixes=('', '_measurements'))
    # Initialize schema and writer on first chunk
    if i == 0:
        schema = pa.Table.from_pandas(merged_chunk).schema
        writer = pq.ParquetWriter(output_parquet, schema)
    # Write chunk to Parquet
    writer.write_table(pa.Table.from_pandas(merged_chunk))
    # Update merged row count
    merged_rows += merged_chunk.shape[0]
    # Collect missing values
    for col in merged_chunk.columns:
        missing_values[col] = missing_values.get(col, 0) + merged_chunk[col].isnull().sum()
    # Collect preview rows (first 5 non-null rows)
    if len(preview_rows) < preview_count:
        preview_rows.extend(merged_chunk.head(preview_count - len(preview_rows)).to_dict('records'))
    del chunk, merged_chunk
    gc.collect()

# Close writer temporarily
if writer:
    writer.close()

# Reopen Parquet writer for materials join
writer = pq.ParquetWriter(output_parquet + ".temp", schema)
materials_reader = pq.ParquetFile(materials_file)
for i in range(materials_reader.num_row_groups):
    chunk = materials_reader.read_row_group(i).to_pandas()
    # Perform left join with previous merged result
    merged_chunk = bookings.merge(chunk, on="serial_number_id", how="left", suffixes=('', '_materials'))
    writer.write_table(pa.Table.from_pandas(merged_chunk))
    # Update missing values
    for col in merged_chunk.columns:
        missing_values[col] = missing_values.get(col, 0) + merged_chunk[col].isnull().sum()
    # Collect preview rows if not already collected
    if len(preview_rows) < preview_count:
        preview_rows.extend(merged_chunk.head(preview_count - len(preview_rows)).to_dict('records'))
    del chunk, merged_chunk
    gc.collect()

# Close writer
if writer:
    writer.close()

# Rename temp file to final output
if os.path.exists(output_parquet):
    os.remove(output_parquet)
os.rename(output_parquet + ".temp", output_parquet)

# Calculate unjoined rows
measurements_joined = merged_rows - bookings[bookings['serial_number_id'].isin(measurements['serial_number_id'])].shape[0]
measurements_unjoined = measurements_rows - measurements_joined
materials_joined = merged_rows - bookings[bookings['serial_number_id'].isin(materials['serial_number_id'])].shape[0]
materials_unjoined = materials_rows - materials_joined
bookings_unjoined = bookings_rows - merged_rows  # Should be 0 for left join

# Create summary dictionary
summary = {
    "Metric": [
        "Total rows in bookings",
        "Number of rows in merged result",
        "Bookings rows not joined",
        "Total rows in measurements",
        "Measurements rows not joined",
        "Total rows in materials",
        "Materials rows not joined",
        "Number of columns in merged result"
    ],
    "Value": [
        bookings_rows,
        merged_rows,
        bookings_unjoined,
        measurements_rows,
        measurements_unjoined,
        materials_rows,
        materials_unjoined,
        len(bookings_columns) + len(measurements_columns) + len(materials_columns) - 2  # Subtract 2 for shared join keys
    ]
}

# Create DataFrame for summary
summary_df = pd.DataFrame(summary)

# Add missing values to summary
missing_df = pd.DataFrame({
    "Metric": ["Missing values in " + col for col in missing_values.keys()],
    "Value": list(missing_values.values())
})

# Combine summary and missing values
summary_df = pd.concat([summary_df, missing_df], ignore_index=True)

# Add preview rows
preview_df = pd.DataFrame(preview_rows)
if not preview_df.empty:
    preview_df.insert(0, "Metric", [f"Preview row {i+1}" for i in range(len(preview_df))])
    summary_df = pd.concat([summary_df, preview_df], ignore_index=True)

# Save summary to CSV
summary_df.to_csv(output_summary, index=False)