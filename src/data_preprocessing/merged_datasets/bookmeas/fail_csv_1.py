import pandas as pd
import os
from datetime import datetime

def analyze_and_extract_data(file_path, target_book_state):
    """
    Performs a comprehensive analysis of a Parquet file:
    - Calculates overall source file statistics.
    - Filters the original data by a specific 'book_state', saves it to a Parquet file.
    - Extracts and saves unique 'station_id' and 'serial_number_id' from the filtered data.
    - Generates a summary report of all calculated statistics.

    Args:
        file_path (str): The full path to the input Parquet file (e.g., '4_week_v2.parquet').
        target_book_state (int/str): The value of 'book_state' to filter by.
    """
    output_directory = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output file paths with timestamp
    # No deduplicated_output_parquet_path as deduplication is not saved separately
    # No duplicates_all_columns_csv_path as the check is eliminated
    filtered_output_parquet_path = os.path.join(output_directory, f"{base_name}_book_state_{target_book_state}_{timestamp}.parquet")
    unique_station_ids_filtered_csv_path = os.path.join(output_directory, f"{base_name}_unique_station_ids_filtered_{timestamp}.csv")
    unique_serial_number_ids_filtered_csv_path = os.path.join(output_directory, f"{base_name}_unique_serial_number_ids_filtered_{timestamp}.csv")
    analysis_report_csv_path = os.path.join(output_directory, f"{base_name}_analysis_report_{timestamp}.csv")

    # Initialize report data
    report_data = {}

    try:
        print(f"Loading Parquet file from: {file_path}")
        df_source = pd.read_parquet(file_path)
        print("File loaded successfully.")

        # --- Initial Column Checks ---
        required_cols = ['book_state', 'station_id', 'serial_number_id', 'booking_id']
        for col in required_cols:
            if col not in df_source.columns:
                print(f"Error: Required column '{col}' not found in the Parquet file. Cannot proceed.")
                return

        # Ensure relevant columns are string type for consistent operations
        df_source['station_id'] = df_source['station_id'].astype(str)
        df_source['serial_number_id'] = df_source['serial_number_id'].astype(str)
        df_source['booking_id'] = df_source['booking_id'].astype(str)
        df_source['book_state'] = pd.to_numeric(df_source['book_state'], errors='coerce')

        # --- Removed: Duplicate Checks in Source File ---
        # As per request, assuming no duplicates and no need for explicit check/saving.
            
        # --- Removed: Deduplicate the source DataFrame and saving as a new Parquet file ---
        # All analysis will now proceed directly on df_source.


        # --- Report Section 1: Source File Statistics (Based on original df_source) ---
        print("\n--- Analyzing Source File Statistics (Original Source File) ---")
        report_data['Total Rows (Original Source File)'] = len(df_source) 
        report_data['Unique Station IDs (Original Source File)'] = df_source['station_id'].nunique()
        report_data['Unique Serial Number IDs (Original Source File)'] = df_source['serial_number_id'].nunique()
        print(f"Original source file contains {report_data['Total Rows (Original Source File)']} rows.")
        print(f"Original source file contains {report_data['Unique Station IDs (Original Source File)']} unique station IDs.")
        print(f"Original source file contains {report_data['Unique Serial Number IDs (Original Source File)']} unique serial number IDs.")


        # --- Filter ORIGINAL data by 'book_state' and save as PARQUET ---
        # Filtering is now done directly on df_source
        print(f"\n--- Filtering ORIGINAL data where 'book_state' is {target_book_state} and saving as Parquet ---")
        df_filtered = df_source[df_source['book_state'] == target_book_state].copy()

        if not df_filtered.empty:
            df_filtered.to_parquet(filtered_output_parquet_path, index=False)
            print(f"Saved {len(df_filtered)} rows with book_state={target_book_state} (from original data) to: {filtered_output_parquet_path}")

            # --- Report Section 2: Filtered Data Statistics (from the newly saved Parquet) ---
            report_data['Total Rows (Filtered Original Data)'] = len(df_filtered) 
            report_data['Unique Station IDs (Filtered Original Data)'] = df_filtered['station_id'].nunique()
            report_data['Unique Serial Number IDs (Filtered Original Data)'] = df_filtered['serial_number_id'].nunique()
            print(f"Filtered original data contains {report_data['Total Rows (Filtered Original Data)']} rows.")
            print(f"Filtered original data contains {report_data['Unique Station IDs (Filtered Original Data)']} unique station IDs.")
            print(f"Filtered original data contains {report_data['Unique Serial Number IDs (Filtered Original Data)']} unique serial number IDs.")


            # --- Extract and save separate unique station_ids and serial_number_ids from filtered original data ---
            print("\n--- Extracting and Saving Unique IDs from Filtered Original Data ---")
            
            # Unique Station IDs from filtered original data
            unique_station_ids_filtered_list = df_filtered['station_id'].unique().tolist()
            if unique_station_ids_filtered_list:
                pd.DataFrame(unique_station_ids_filtered_list, columns=['station_id']).to_csv(unique_station_ids_filtered_csv_path, index=False)
                print(f"Saved {len(unique_station_ids_filtered_list)} unique Station IDs (from filtered original data) to: {unique_station_ids_filtered_csv_path}")
            else:
                print("No unique Station IDs found in the filtered original data.")

            # Unique Serial Number IDs from filtered original data
            unique_serial_number_ids_filtered_list = df_filtered['serial_number_id'].unique().tolist()
            if unique_serial_number_ids_filtered_list:
                pd.DataFrame(unique_serial_number_ids_filtered_list, columns=['serial_number_id']).to_csv(unique_serial_number_ids_filtered_csv_path, index=False)
                print(f"Saved {len(unique_serial_number_ids_filtered_list)} unique Serial Number IDs (from filtered original data) to: {unique_serial_number_ids_filtered_csv_path}")
            else:
                print("No unique Serial Number IDs found in the filtered original data.")

        else:
            print(f"No rows found with book_state={target_book_state} in original data. No filtered Parquet or unique IDs generated.")
            # Populate report for filtered data as 0 if no rows found
            report_data['Total Rows (Filtered Original Data)'] = 0
            report_data['Unique Station IDs (Filtered Original Data)'] = 0
            report_data['Unique Serial Number IDs (Filtered Original Data)'] = 0


        # --- Generate and Save Analysis Report ---
        print("\n--- Generating Analysis Report ---")
        # Convert dictionary to a DataFrame for easy saving as CSV
        # Transpose to make the metric names a column
        report_df = pd.DataFrame.from_dict(report_data, orient='index', columns=['Value'])
        report_df.index.name = 'Metric' # Name the index column
        report_df.to_csv(analysis_report_csv_path)
        print(f"Analysis report saved to: {analysis_report_csv_path}")
        print("\n--- Analysis Complete ---")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column '{e}' in the Parquet file. Please ensure all required columns are present.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Define the path to the input Parquet file
    input_parquet_file = "/home/alinzk/Forvia_Hella/4_week_v2.parquet"

    # Define the book_state value to filter by
    target_state = 2

    # Call the main analysis function
    analyze_and_extract_data(input_parquet_file, target_state)
