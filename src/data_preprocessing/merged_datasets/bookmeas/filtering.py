# filtering.py

import os
import sys
import pandas as pd

# The following code block explicitly adds the required directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
utils_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'utils'))
single_datasets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'single_datasets'))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if utils_dir not in sys.path:
    sys.path.append(utils_dir)
    
if single_datasets_dir not in sys.path:
    sys.path.append(single_datasets_dir)

from utils import find_latest_file
from processing import perform_dynamic_exclusion
import config

def filter_for_anomalous_candidates(file_path, target_book_state):
    """
    Performs Stage 1 filtering, extracting unique serial numbers and stations
    that match the specified book state to be analyzed for anomalies.
    """
    print(f"Starting Stage 1: Filtering data for book_state = {target_book_state} to find anomalous candidates.")
    try:
        df = pd.read_parquet(file_path)
        df_filtered = df[df['book_state'] == target_book_state]
        
        # The line to save the intermediate parquet file has been removed.
        # df_filtered.to_parquet(config.FILTERED_PARQUET_STAGE1, index=False)
        # print(f"Filtered Parquet file saved to: {config.FILTERED_PARQUET_STAGE1}")

        # Extract and save unique station IDs and descriptions
        unique_stations_df = df_filtered[['station_id', 'station_desc']].drop_duplicates()
        unique_stations_df.to_csv(config.UNIQUE_STATIONS_CSV, index=False)
        print(f"Unique station IDs and descriptions saved to: {config.UNIQUE_STATIONS_CSV}")

        # Extract and save unique serial number IDs
        unique_serial_number_ids_filtered = df_filtered['serial_number_id'].unique()
        unique_serial_number_ids_df = pd.DataFrame(unique_serial_number_ids_filtered, columns=['serial_number_id'])
        unique_serial_number_ids_df.to_csv(config.UNIQUE_SERIAL_NUMBERS_CSV, index=False)
        print(f"Unique serial number IDs saved to: {config.UNIQUE_SERIAL_NUMBERS_CSV}")

        print("--- Stage 1: Candidate Filtering Complete ---")

    except Exception as e:
        print(f"An error occurred during Stage 1 processing: {e}")
        raise

def analyze_serial_number_book_states(source_file_path, unique_serial_numbers_csv_path):
    """
    Performs detailed analysis for Stage 2, counting book states for each serial number.
    """
    print("\nStarting Stage 2: Detailed Analysis of book_state counts for each serial number.")
    try:
        unique_serial_numbers_df = pd.read_csv(unique_serial_numbers_csv_path)
        unique_serial_number_ids = unique_serial_numbers_df['serial_number_id'].tolist()
        
        source_df = pd.read_parquet(source_file_path)
        source_df['serial_number_id'] = source_df['serial_number_id'].astype(str)

        # Filter the original dataframe to include only the rows with the unique serial number IDs
        filtered_df = source_df[source_df['serial_number_id'].isin(unique_serial_number_ids)].copy()

        # Group by 'serial_number_id' and count the occurrences of each 'book_state'
        book_state_counts = pd.crosstab(filtered_df['serial_number_id'], filtered_df['book_state'])
        
        # Ensure all book states are columns, even if they don't appear for a specific serial number
        all_book_states = source_df['book_state'].unique()
        for state in all_book_states:
            if state not in book_state_counts.columns:
                book_state_counts[state] = 0
        
        # Calculate the total number of rows for each serial number ID
        total_rows_per_serial = filtered_df.groupby('serial_number_id').size().rename('Total Rows')
        
        # Merge the counts and the totals
        summary_df = book_state_counts.merge(total_rows_per_serial, on='serial_number_id')
        
        # Rename columns for clarity
        summary_df.columns = [f'Book State {col} Count' if col != 'Total Rows' else col for col in summary_df.columns]
        summary_df.index.name = 'Serial Number ID'
        
        summary_df.to_csv(config.SERIAL_NUMBER_SUMMARY_CSV)
        print(f"Summary of serial number book state counts saved to: {config.SERIAL_NUMBER_SUMMARY_CSV}")

        print("--- Stage 2: Analysis Complete ---")

    except Exception as e:
        print(f"An error occurred during Stage 2 analysis: {e}")
        raise

def main():
    """
    Main pipeline function to execute all stages.
    """
    try:
        # --- Stage 1: Filter and Extract Candidates ---
        filter_for_anomalous_candidates(config.SOURCE_PARQUET_FILE, target_book_state=2)

        # --- Find the latest Stage 1 output to use as input for Stage 2 ---
        unique_serials_csv_path = find_latest_file(
            config.output_dir,
            prefix=f"{config.BASE_FILENAME}_serial_numbers_for_filtering_",
            extension=".csv"
        )
        
        if not unique_serials_csv_path:
            raise FileNotFoundError("Could not find the serial number IDs CSV from Stage 1.")
        
        print(f"Found latest Stage 1 output: {unique_serials_csv_path}")

        # --- Stage 2: Detailed Serial Number Analysis ---
        analyze_serial_number_book_states(config.SOURCE_PARQUET_FILE, unique_serials_csv_path)

        # --- Find the latest Stage 2 output to use as input for Stage 3 ---
        stage2_summary_csv_path = find_latest_file(
            config.output_dir,
            prefix=f"{config.BASE_FILENAME}_serial_number_book_state_summary_",
            extension=".csv"
        )
        
        if not stage2_summary_csv_path:
            raise FileNotFoundError("Could not find the serial number summary CSV from Stage 2.")
        
        print(f"Found latest Stage 2 output: {stage2_summary_csv_path}")

        # --- Stage 3: Dynamic Exclusion and Final Reporting ---
        perform_dynamic_exclusion(
            config.SOURCE_PARQUET_FILE,
            stage2_summary_csv_path
        )
        
        print("\n--- ✅ Pipeline execution completed successfully ✅ ---")
        print(f"All reports and final files are saved to: {config.output_dir}")

    except Exception as e:
        print(f"\n--- ❌ Pipeline failed: {e} ---")

if __name__ == "__main__":
    main()
