# This script inspects the structure of multiple Parquet files by extracting each file’s column names, data types, and total row count.
# It handles errors gracefully, logs metadata to the console, and stores the detailed results—including any read errors—in a
# structured CSV file. This provides a quick overview of the schema and size of each dataset, useful for data auditing and exploration.
import pyarrow.parquet as pq
import pandas as pd
import os

# Path to the Parquet files
path = 'C:/Users/alina/Downloads/data_hella/'

# Output directory for the CSV
output_dir = 'C:/Users/alina/OneDrive/Desktop/RWML projects/data_hella/output/'
os.makedirs(output_dir, exist_ok=True)

# Output CSV file path
output_csv = os.path.join(output_dir, 'parquet_metadata.csv')

# List of Parquet files to inspect
parquet_files = ['bookings.parquet', 'materials.parquet', 'measurements.parquet', 'messages.parquet']

# Function to get column names, types, and row count for a Parquet file
def inspect_parquet_file(file_path):
    try:
        # Read the Parquet file
        parquet_file = pq.ParquetFile(file_path)
        
        # Get the Arrow schema (column names and types)
        schema = parquet_file.schema_arrow
        columns = [(field.name, str(field.type)) for field in schema]
        
        # Get the total number of rows
        total_rows = parquet_file.metadata.num_rows
        
        return columns, total_rows
    except Exception as e:
        return None, f"Error reading {file_path}: {e}"

# Collect results for all Parquet files
results = []

for file in parquet_files:
    file_path = os.path.join(path, file)
    print(f"\n=== Inspecting {file} ===")
    
    # Get columns and row count
    columns, result = inspect_parquet_file(file_path)
    
    if columns is not None:
        print(f"Columns in {file}:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")
        print(f"Total rows: {result}")
        
        # Add to results for CSV
        for col_name, col_type in columns:
            results.append({
                'file_name': file,
                'column_name': col_name,
                'column_type': col_type,
                'total_rows': result
            })
    else:
        print(result)
        # Add error to results
        results.append({
            'file_name': file,
            'column_name': None,
            'column_type': None,
            'total_rows': None,
            'error': result
        })

# Create a DataFrame and save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)
print(f"\nSaved metadata to {output_csv}")