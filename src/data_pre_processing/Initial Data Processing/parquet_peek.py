# This script previews the first 20 rows of each specified Parquet file by reading from the first row group using PyArrow.
# It prints a sample of the data and column names to the console for quick inspection and saves each preview to a corresponding
# CSV file in the specified output directory. This is useful for quickly examining dataset structure and content without loading
# the full file into memory.
import pandas as pd
import pyarrow.parquet as pq
import os

# Update the path to the correct location
path = 'C:/Users/alina/Downloads/data_hella/'

# Create a directory to save CSV files (if it doesn't exist)
output_dir = 'C:/Users/alina/OneDrive/Desktop/RWML projects/data_hella/output/'
os.makedirs(output_dir, exist_ok=True)

# List of Parquet files to inspect
parquet_files = ['bookings.parquet', 'materials.parquet', 'measurements.parquet', 'messages.parquet']

# Loop through each Parquet file, peek into it, and save to CSV
for file in parquet_files:
    try:
        # Read the Parquet file as a pyarrow Table
        parquet_file = pq.ParquetFile(path + file)
        # Read the first 5 rows from the first row group
        table = parquet_file.read_row_group(0).slice(0, 20)
        df = table.to_pandas()
        print(f"\n=== {file} ===")
        print(df)
        print(f"Columns in {file}:", df.columns.tolist())

        # Save the DataFrame to a CSV file
        csv_filename = os.path.join(output_dir, file.replace('.parquet', '_peek.csv'))
        df.to_csv(csv_filename, index=False)
        print(f"Saved {csv_filename}")
    except Exception as e:
        print(f"Error reading {file}: {e}")