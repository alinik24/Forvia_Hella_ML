# config.py

import os
from datetime import datetime

# Dynamically determine the base path from the location of this script
current_script_path = os.path.abspath(__file__)
base_path = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..', '..', '..'))

# Define output directories
output_dir = os.path.join(base_path, "src", "output")
os.makedirs(output_dir, exist_ok=True)

# Define the full path to the source Parquet file
SOURCE_PARQUET_FILE = os.path.join(base_path, "4_week.parquet")

# --- Define Output Filenames ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_FILENAME = os.path.splitext(os.path.basename(SOURCE_PARQUET_FILE))[0]

# Stage 1 Outputs
UNIQUE_SERIAL_NUMBERS_CSV = os.path.join(output_dir, f"{BASE_FILENAME}_serial_numbers_for_filtering_{TIMESTAMP}.csv")
FILTERED_PARQUET_STAGE1 = os.path.join(output_dir, f"{BASE_FILENAME}_book_state_2_{TIMESTAMP}.parquet")
UNIQUE_STATIONS_CSV = os.path.join(output_dir, f"{BASE_FILENAME}_unique_stations_ids_and_descs_{TIMESTAMP}.csv")

# Stage 2 Outputs
SERIAL_NUMBER_ROWS_CSV = os.path.join(output_dir, f"{BASE_FILENAME}_serial_number_rows_all_book_states_{TIMESTAMP}.csv")
SERIAL_NUMBER_SUMMARY_CSV = os.path.join(output_dir, f"{BASE_FILENAME}_serial_number_book_state_summary_{TIMESTAMP}.csv")

# Stage 3 Outputs
ANOMALOUS_SERIAL_NUMBERS_CSV = os.path.join(output_dir, f"{BASE_FILENAME}_anomalous_serial_numbers_{TIMESTAMP}.csv")
FINAL_FILTERED_PARQUET = os.path.join(output_dir, f"{BASE_FILENAME}_final_filtered_data_{TIMESTAMP}.parquet")
DYNAMIC_EXCLUSION_REPORT = os.path.join(output_dir, f"{BASE_FILENAME}_dynamic_exclusion_report_{TIMESTAMP}.csv")
