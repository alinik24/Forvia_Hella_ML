
# This script processes the `measurements_single_line.parquet` file to compute, for each unique `serial_number_id`, the total row count
# and the number of unique values for key attributes such as `catalog_id`, `recipe_revision_id`, `booking_id`, `product_id`,
# `product_variant_id`, `part_number`, and `teststep_id`. It processes the Parquet file in large memory-efficient batches, tracks all
# combinations, and writes the aggregated statistics to a timestamped CSV file. The script also verifies that the total row count
# matches the original dataset, ensuring data integrity.
import pyarrow.parquet as pqZ
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

# Define paths
base_path = r"C:\Desktop\Research and Thesis\RWML projects\data_hella_single_line"
measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
output_csv = os.path.join(base_path, f"serial_counts_all_{timestamp}.csv")

# Step 1: Read the Parquet file and collect data
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Step 1: Collecting serial_number_id, teststep_id, catalog_id, recipe_revision_id, booking_id, product_id, product_variant_id, and part_number data")

parquet_file = pq.ParquetFile(measurements_file)
total_rows = parquet_file.metadata.num_rows
batch_size = 2000000  # Increased batch size for better performance (adjust based on memory)

# Dictionary to store aggregated data for each serial_number_id
serial_data = {}

with tqdm(total=total_rows, desc="Processing Parquet rows", unit="rows") as pbar:
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[
        'serial_number_id', 'teststep_id', 'catalog_id', 'recipe_revision_id',
        'booking_id', 'product_id', 'product_variant_id', 'part_number'
    ], use_threads=True):
        df_batch = batch.to_pandas()
        # Replace null/NaN values with placeholders
        df_batch.fillna("unlabeled", inplace=True)
        
        # Aggregate unique values and counts using groupby
        agg = df_batch.groupby('serial_number_id').agg({
            'catalog_id': lambda x: set(x),
            'recipe_revision_id': lambda x: set(x),
            'booking_id': lambda x: set(x),
            'product_id': lambda x: set(x),
            'product_variant_id': lambda x: set(x),
            'part_number': lambda x: set(x),
            'teststep_id': lambda x: set(x),
            'serial_number_id': 'count'  # Row count
        }).rename(columns={'serial_number_id': 'row_count'})
        
        # Update serial_data
        for serial, row in agg.iterrows():
            if serial not in serial_data:
                serial_data[serial] = {
                    'catalog_ids': set(),
                    'recipe_revision_ids': set(),
                    'booking_ids': set(),
                    'product_ids': set(),
                    'product_variant_ids': set(),
                    'part_numbers': set(),
                    'teststep_ids': set(),
                    'row_count': 0
                }
            serial_data[serial]['catalog_ids'].update(row['catalog_id'])
            serial_data[serial]['recipe_revision_ids'].update(row['recipe_revision_id'])
            serial_data[serial]['booking_ids'].update(row['booking_id'])
            serial_data[serial]['product_ids'].update(row['product_id'])
            serial_data[serial]['product_variant_ids'].update(row['product_variant_id'])
            serial_data[serial]['part_numbers'].update(row['part_number'])
            serial_data[serial]['teststep_ids'].update(row['teststep_id'])
            serial_data[serial]['row_count'] += row['row_count']
        
        pbar.update(len(df_batch))
        # Free memory
        del df_batch
        del agg

print(f"Step 1 Complete: Found {len(serial_data)} unique serial_number_id values")

# Step 2: Prepare table data
print("Step 2: Preparing table data")
output_data = []

for serial in tqdm(serial_data.keys(), desc="Preparing table rows", unit="serial"):
    row = {
        'serial_number_id': serial,
        'unique_catalog_id_count': len(serial_data[serial]['catalog_ids']),
        'total_catalog_id_count': serial_data[serial]['row_count'],
        'unique_recipe_revision_id_count': len(serial_data[serial]['recipe_revision_ids']),
        'total_recipe_revision_id_count': serial_data[serial]['row_count'],
        'unique_booking_id_count': len(serial_data[serial]['booking_ids']),
        'total_booking_id_count': serial_data[serial]['row_count'],
        'unique_product_id_count': len(serial_data[serial]['product_ids']),
        'total_product_id_count': serial_data[serial]['row_count'],
        'unique_product_variant_id_count': len(serial_data[serial]['product_variant_ids']),
        'total_product_variant_id_count': serial_data[serial]['row_count'],
        'unique_part_number_count': len(serial_data[serial]['part_numbers']),
        'total_part_number_count': serial_data[serial]['row_count'],
        'unique_teststep_id_count': len(serial_data[serial]['teststep_ids']),
        'total_teststep_id_count': serial_data[serial]['row_count']
    }
    output_data.append(row)

# Sort by serial_number_id for consistent output
output_data.sort(key=lambda x: x['serial_number_id'])

print("Step 2 Complete: Table data prepared")

# Step 3: Write to CSV
print("Step 3: Writing results to CSV")
with open(output_csv, 'w', encoding='utf-8') as f:
    # Write header
    headers = (
        "serial_number_id,unique_catalog_id_count,total_catalog_id_count,"
        "unique_recipe_revision_id_count,total_recipe_revision_id_count,"
        "unique_booking_id_count,total_booking_id_count,unique_product_id_count,"
        "total_product_id_count,unique_product_variant_id_count,total_product_variant_id_count,"
        "unique_part_number_count,total_part_number_count,unique_teststep_id_count,"
        "total_teststep_id_count\n"
    )
    f.write(headers)
    
    # Write data rows
    for row in tqdm(output_data, desc="Writing table to CSV", unit="row"):
        values = [
            str(row['serial_number_id']),
            str(row['unique_catalog_id_count']),
            str(row['total_catalog_id_count']),
            str(row['unique_recipe_revision_id_count']),
            str(row['total_recipe_revision_id_count']),
            str(row['unique_booking_id_count']),
            str(row['total_booking_id_count']),
            str(row['unique_product_id_count']),
            str(row['total_product_id_count']),
            str(row['unique_product_variant_id_count']),
            str(row['total_product_variant_id_count']),
            str(row['unique_part_number_count']),
            str(row['total_part_number_count']),
            str(row['unique_teststep_id_count']),
            str(row['total_teststep_id_count'])
        ]
        f.write(','.join(values) + '\n')

print(f"Step 3 Complete: CSV saved as {output_csv}")

# Step 4: Verify total row count
print("Step 4: Verifying total row count")
total_counted_rows = sum(row['total_teststep_id_count'] for row in output_data)
print(f"Sum of total teststep counts: {total_counted_rows}")
print(f"Total rows in file: {total_rows}")
if total_counted_rows == total_rows:
    print("Verification successful: All rows are accounted for.")
else:
    print(f"Verification failed: Sum of total teststep counts ({total_counted_rows}) does not match total rows ({total_rows}).")

print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")