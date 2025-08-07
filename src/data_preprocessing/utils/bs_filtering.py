import os

import pandas as pd
import pyarrow.parquet as pq


def find_latest_file(directory, prefix, extension=".csv"):
    """
    Finds the most recently created file in a directory that matches a given prefix and extension.

    Args:
        directory (str): The directory to search in.
        prefix (str): The filename prefix to match.
        extension (str): The filename extension to match (default: .csv).

    Returns:
        str: The full path to the most recent matching file, or None if no file is found.
    """
    try:
        all_files = os.listdir(directory)
        relevant_files = [f for f in all_files if f.startswith(prefix) and f.endswith(extension)]

        if not relevant_files:
            return None

        # Sort files by modification time (most recent first)
        relevant_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

        return os.path.join(directory, relevant_files[0])
    except Exception as e:
        print(f"Error finding latest file: {e}")
        return None


def filter_for_anomalous_candidates(file_path, target_book_state, unique_stations, unique_serial_numbers):
    """
    Performs Stage 1 filtering, extracting unique serial numbers and stations
    that match the specified book state to be analyzed for anomalies.
    """
    print(f"Starting Stage 1: Filtering data for book_state = {target_book_state} to find anomalous candidates.")
    try:
        df = pd.read_parquet(file_path)
        df_filtered = df[df['book_state'] == target_book_state]

        # The line to save the intermediate parquet file has been removed.
        # df_filtered.to_parquet(FILTERED_PARQUET_STAGE1, index=False)
        # print(f"Filtered Parquet file saved to: {FILTERED_PARQUET_STAGE1}")

        # Extract and save unique station IDs and descriptions
        unique_stations_df = df_filtered[['station_id', 'station_desc']].drop_duplicates()
        unique_stations_df.to_csv(unique_stations, index=False)
        print(f"Unique station IDs and descriptions saved to: {unique_stations}")

        # Extract and save unique serial number IDs
        unique_serial_number_ids_filtered = df_filtered['serial_number_id'].unique()
        unique_serial_number_ids_df = pd.DataFrame(unique_serial_number_ids_filtered, columns=['serial_number_id'])
        unique_serial_number_ids_df.to_csv(unique_serial_numbers, index=False)
        print(f"Unique serial number IDs saved to: {unique_serial_numbers}")

        print("--- Stage 1: Candidate Filtering Complete ---")

    except Exception as e:
        print(f"An error occurred during Stage 1 processing: {e}")
        raise


def analyze_serial_number_book_states(source_file_path, unique_serial_numbers_csv_path, serial_number_summary_csv):
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

        summary_df.to_csv(serial_number_summary_csv)
        print(f"Summary of serial number book state counts saved to: {serial_number_summary_csv}")

        print("--- Stage 2: Analysis Complete ---")

    except Exception as e:
        print(f"An error occurred during Stage 2 analysis: {e}")
        raise
