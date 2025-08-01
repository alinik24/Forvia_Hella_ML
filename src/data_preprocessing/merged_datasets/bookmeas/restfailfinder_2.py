import pandas as pd
import os
from datetime import datetime
from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_4w

def analyze_serial_numbers(source_file_path, unique_serial_numbers_csv_path):
    """
    Reads a list of unique serial number IDs, loads the original source data,
    extracts all rows for each serial number ID, saves these rows to a formatted CSV,
    and generates a summary report of book state counts and total rows for each serial number ID.
    It also generates a global summary of book state counts from the entire source file.

    Args:
        source_file_path (str): Path to the original source Parquet file (e.g., '4_week_v2.parquet').
        unique_serial_numbers_csv_path (str): Path to the CSV file containing unique serial_number_ids
                                               (e.g., '4_week_v2_unique_serial_number_ids_filtered_TIMESTAMP.csv').
    """
    output_directory = os.path.dirname(source_file_path)
    base_name = os.path.splitext(os.path.basename(source_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output file paths
    # Filename adjusted to reflect no exclusion
    serial_number_rows_output_csv_path = os.path.join(output_directory, f"{base_name}_serial_number_rows_all_book_states_{timestamp}.csv")
    serial_number_book_state_summary_csv_path = os.path.join(output_directory, f"{base_name}_serial_number_book_state_summary_{timestamp}.csv")
    global_book_state_summary_csv_path = os.path.join(output_directory, f"{base_name}_global_book_state_summary_{timestamp}.csv") # New output file

    try:
        print(f"Loading original source Parquet file from: {source_file_path}")
        df_source = pd.read_parquet(source_file_path)
        print("Original source file loaded successfully.")

        print(f"Loading unique serial number IDs from: {unique_serial_numbers_csv_path}")
        df_unique_serial_numbers = pd.read_csv(unique_serial_numbers_csv_path)
        unique_serial_number_ids = df_unique_serial_numbers['serial_number_id'].tolist()
        print(f"Loaded {len(unique_serial_number_ids)} unique serial number IDs.")

        # Ensure required columns are present and correctly typed
        required_cols = ['serial_number_id', 'book_state']
        for col in required_cols:
            if col not in df_source.columns:
                print(f"Error: Required column '{col}' not found in the source Parquet file. Cannot proceed.")
                return
        df_source['serial_number_id'] = df_source['serial_number_id'].astype(str)
        df_source['book_state'] = pd.to_numeric(df_source['book_state'], errors='coerce')


        # Prepare data for summary report per serial number
        summary_report_data = []

        print(f"\n--- Extracting all rows for each unique serial number ID ---")
        # Open the output CSV file for formatted writing
        with open(serial_number_rows_output_csv_path, 'w', newline='') as f:
            for serial_id in unique_serial_number_ids:
                # Filter rows for the current serial_id (NO EXCLUSION of book_state)
                df_current_serial_rows = df_source[
                    (df_source['serial_number_id'] == serial_id)
                ].copy()

                if not df_current_serial_rows.empty:
                    f.write(f"Serial NumberID: {serial_id}\n")
                    # Write header for the data block, then the data
                    df_current_serial_rows.to_csv(f, index=False, header=True)
                    f.write("----------------------\n") # Separator

                    # Calculate book state counts for the current serial_id (all book states)
                    book_state_counts = df_current_serial_rows['book_state'].value_counts().to_dict()
                    total_rows_for_serial = len(df_current_serial_rows)

                    summary_entry = {
                        'Serial Number ID': serial_id,
                        'Total Rows': total_rows_for_serial, # Adjusted column name
                    }
                    # Dynamically add book state counts
                    for state, count in book_state_counts.items():
                        summary_entry[f'Book State {int(state)} Count'] = count
                    
                    summary_report_data.append(summary_entry)
                    print(f"Processed serial number: {serial_id} (Total rows: {total_rows_for_serial})")
                else:
                    print(f"No rows found for serial number: {serial_id}.")

        print(f"\nExtracted rows saved to: {serial_number_rows_output_csv_path}")

        # Generate and save the book state summary report per serial number
        if summary_report_data:
            summary_df = pd.DataFrame(summary_report_data)
            # Fill NaN for book states not present in all serial numbers with 0
            summary_df = summary_df.fillna(0)
            # Ensure integer types for counts
            for col in summary_df.columns:
                if 'Count' in col or 'Total Rows' in col:
                    summary_df[col] = summary_df[col].astype(int)
            
            summary_df.to_csv(serial_number_book_state_summary_csv_path, index=False)
            print(f"Book state summary report per serial number saved to: {serial_number_book_state_summary_csv_path}")
        else:
            print("No data to generate book state summary report per serial number.")

        # --- Global Book State Counts from Source File ---
        print("\n--- Calculating Global Book State Counts from Source File ---")
        global_book_state_counts = df_source['book_state'].value_counts().to_dict()
        global_summary_data = {
            'Metric': 'Total Rows in Source File',
            'Value': len(df_source)
        }
        for state, count in global_book_state_counts.items():
            global_summary_data[f'Book State {int(state)} Count (Source)'] = count

        global_summary_df = pd.DataFrame([global_summary_data])
        global_summary_df.to_csv(global_book_state_summary_csv_path, index=False)
        print(f"Global book state summary saved to: {global_book_state_summary_csv_path}")

        print("\n--- Serial Number Analysis Complete ---")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}")
    except KeyError as e:
        print(f"Error: Missing expected column '{e}' in one of the data files. Please check column names.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution for Serial Number Analysis ---
if __name__ == "__main__":
    # Define the path to the original source Parquet file
    original_source_parquet_file = INPUT_FILE_BOOKMEAS_4w

    # Define the path to the CSV containing unique serial number IDs from the first script's output.
    # IMPORTANT: You will need to replace 'YOUR_TIMESTAMP_HERE' with the actual timestamp
    # from the file generated by the first script.
    # Example: "/home/alinzk/Forvia_Hella/4_week_v2_unique_serial_number_ids_filtered_20240722_233000.csv"
    # It is recommended to copy the exact filename from your directory after running the first script.
    unique_serial_numbers_input_csv = "/home/alinzk/Forvia_Hella/4_week_v2_unique_serial_number_ids_filtered_20250722_234527.csv" # Example timestamp

    # The 'book_state_to_exclude' parameter is no longer used in the filtering logic
    # but is kept as a placeholder in the function signature for compatibility if needed elsewhere.
    # Its value is effectively ignored for row extraction now.
    book_state_to_exclude_from_analysis = 2 

    # Call the serial number analysis function
    analyze_serial_numbers(original_source_parquet_file, unique_serial_numbers_input_csv)
