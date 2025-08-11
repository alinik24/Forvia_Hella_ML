import pandas as pd
import pyarrow.parquet as pq
import os

# Update the path to the correct location using a raw string to handle backslashes.
path = r'C:\Users\alina\Downloads\output'

# Create a directory to save CSV files (if it doesn't exist)
output_dir = r'C:\Desktop\Research and Thesis\RWML projects\Forvia_Project\Forvia_Project\data\output'
os.makedirs(output_dir, exist_ok=True)

# List of Parquet files to inspect
parquet_files = ['encoded_bookmeas.parquet']

# Loop through each Parquet file, peek into it, and save to CSV
for file in parquet_files:
    try:
        # Correctly join the path and file name using os.path.join()
        full_path = os.path.join(path, file)
        
        # Read the Parquet file as a pyarrow Table
        parquet_file = pq.ParquetFile(full_path)
        
        # Read the first 20 rows from the first row group
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
