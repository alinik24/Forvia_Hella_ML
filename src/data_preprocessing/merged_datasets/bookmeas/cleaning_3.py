import pandas as pd
import os
from datetime import datetime

# --- Original analyze_serial_numbers function (kept for reference, but not called in main) ---
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
    serial_number_rows_output_csv_path = os.path.join(output_directory, f"{base_name}_serial_number_rows_all_book_states_{timestamp}.csv")
    serial_number_book_state_summary_csv_path = os.path.join(output_directory, f"{base_name}_serial_number_book_state_summary_{timestamp}.csv")
    global_book_state_summary_csv_path = os.path.join(output_directory, f"{base_name}_global_book_state_summary_{timestamp}.csv")

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
        with open(serial_number_rows_output_csv_path, 'w', newline='') as f:
            for serial_id in unique_serial_number_ids:
                df_current_serial_rows = df_source[
                    (df_source['serial_number_id'] == serial_id)
                ].copy()

                if not df_current_serial_rows.empty:
                    f.write(f"Serial NumberID: {serial_id}\n")
                    df_current_serial_rows.to_csv(f, index=False, header=True)
                    f.write("----------------------\n")

                    book_state_counts = df_current_serial_rows['book_state'].value_counts().to_dict()
                    total_rows_for_serial = len(df_current_serial_rows)

                    summary_entry = {
                        'Serial Number ID': serial_id,
                        'Total Rows': total_rows_for_serial,
                    }
                    for state, count in book_state_counts.items():
                        summary_entry[f'Book State {int(state)} Count'] = count
                    
                    summary_report_data.append(summary_entry)
                    print(f"Processed serial number: {serial_id} (Total rows: {total_rows_for_serial})")
                else:
                    print(f"No rows found for serial number: {serial_id}.")

        print(f"\nExtracted rows saved to: {serial_number_rows_output_csv_path}")

        if summary_report_data:
            summary_df = pd.DataFrame(summary_report_data)
            summary_df = summary_df.fillna(0)
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


# --- NEW FUNCTION: filter_and_report_serial_numbers ---
def filter_and_report_serial_numbers(source_file_path, exclusion_csv_path):
    """
    Reads the source Parquet file, identifies serial numbers with specific book states (1 or 2),
    excludes those present in an exclusion CSV (with exceptions), eliminates all rows corresponding to
    the remaining serial numbers from the source, and saves the final remaining rows
    to a new Parquet file. It also generates a step-by-step report.

    Args:
        source_file_path (str): Path to the original source Parquet file.
        exclusion_csv_path (str): Path to the CSV file containing serial numbers to exclude.
    """
    output_directory = os.path.dirname(source_file_path)
    base_name = os.path.splitext(os.path.basename(source_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output file paths
    initial_target_serials_csv_path = os.path.join(output_directory, f"{base_name}_initial_target_serials_bs1_or_bs2_{timestamp}.csv")
    final_filtered_parquet_path = os.path.join(output_directory, f"{base_name}_filtered_by_serial_exclusion_{timestamp}.parquet")
    exclusion_analysis_report_csv_path = os.path.join(output_directory, f"{base_name}_exclusion_analysis_report_{timestamp}.csv")

    report_data = {}

    try:
        print(f"Loading original source Parquet file from: {source_file_path}")
        df_source = pd.read_parquet(source_file_path)
        print("Original source file loaded successfully.")

        # --- Initial Column Checks and Type Conversions ---
        required_cols = ['serial_number_id', 'book_state']
        for col in required_cols:
            if col not in df_source.columns:
                print(f"Error: Required column '{col}' not found in the Parquet file. Cannot proceed.")
                return
        df_source['serial_number_id'] = df_source['serial_number_id'].astype(str)
        df_source['book_state'] = pd.to_numeric(df_source['book_state'], errors='coerce')


        # --- Report Step 1: Original Source File Statistics and Unique Serial Counts ---
        print("\n--- Initial Source File Analysis ---")
        report_data['Total Rows (Original Source)'] = len(df_source)
        original_book_state_counts = df_source['book_state'].value_counts().to_dict()
        for state, count in original_book_state_counts.items():
            report_data[f'Original Book State {int(state)} Count'] = count
        print(f"Original source has {report_data['Total Rows (Original Source)']} rows.")

        # New: Unique serial numbers with book_state 1
        unique_serials_bs1 = df_source[df_source['book_state'] == 1]['serial_number_id'].unique().tolist()
        report_data['Unique Serial Numbers with Book State 1 (Original Source)'] = len(unique_serials_bs1)
        print(f"Unique serial numbers with book_state=1 in source: {len(unique_serials_bs1)}")

        # New: Unique serial numbers with book_state 2
        unique_serials_bs2 = df_source[df_source['book_state'] == 2]['serial_number_id'].unique().tolist()
        report_data['Unique Serial Numbers with Book State 2 (Original Source)'] = len(unique_serials_bs2)
        print(f"Unique serial numbers with book_state=2 in source: {len(unique_serials_bs2)}")

        # New: Unique serial numbers with both book_state 1 AND book_state 2
        serials_bs1_set = set(unique_serials_bs1)
        serials_bs2_set = set(unique_serials_bs2)
        serials_bs1_and_bs2 = list(serials_bs1_set.intersection(serials_bs2_set))
        report_data['Unique Serial Numbers with Book State 1 AND 2 (Original Source)'] = len(serials_bs1_and_bs2)
        print(f"Unique serial numbers with book_state=1 AND book_state=2 in source: {len(serials_bs1_and_bs2)}")


        # --- Step 2: Identify all unique serial numbers with book_state 1 OR 2 ---
        print(f"\n--- Identifying all unique serial numbers with book_state 1 OR 2 ---")
        # Combine serials with book_state 1 or 2
        serials_with_bs1_or_bs2 = df_source[df_source['book_state'].isin([1, 2])]['serial_number_id'].unique().tolist()
        report_data['Serial Numbers with Book State 1 or 2 (Initial Target List)'] = len(serials_with_bs1_or_bs2)
        
        # Save this initial combined list to a new CSV
        if serials_with_bs1_or_bs2:
            pd.DataFrame(serials_with_bs1_or_bs2, columns=['serial_number_id']).to_csv(initial_target_serials_csv_path, index=False)
            print(f"Saved {len(serials_with_bs1_or_bs2)} unique serial numbers with book_state 1 or 2 to: {initial_target_serials_csv_path}")
        else:
            print("No serial numbers found with book_state 1 or 2 in the source file.")


        # --- Step 3: Load serial numbers to exclude ---
        print(f"\n--- Loading serial numbers to exclude from: {exclusion_csv_path} ---")
        df_exclusion_serials = pd.read_csv(exclusion_csv_path)
        # Corrected: Use 'serial_number_id' as the column name
        if 'serial_number_id' not in df_exclusion_serials.columns:
            print(f"Error: 'serial_number_id' column not found in exclusion CSV: {exclusion_csv_path}. Cannot proceed with exclusion.")
            return
        
        exclusion_serial_ids = set(df_exclusion_serials['serial_number_id'].astype(str).tolist())
        report_data['Serial Numbers to Exclude (from Exclusion CSV)'] = len(exclusion_serial_ids)
        print(f"Loaded {len(exclusion_serial_ids)} serial numbers for exclusion.")

        # Define the serial numbers that should NOT be excluded, even if they are in the exclusion list
        force_include_serials_list = {
            'c89a5b10d8d20f74', 'dd23695c2f264822', '7bd8b2015f02a251',
            '4244e83d68afcc54', 'd00cca4f962041be', '22485fc02e2bb798'
        }
        report_data['Serial Numbers Force-Included (despite exclusion list)'] = len(force_include_serials_list)
        print(f"Defined {len(force_include_serials_list)} serial numbers to force-include in elimination list.")


        # --- Step 4: Exclude serial numbers present in the exclusion CSV from the initial target list ---
        # But ensure force_include_serials_list are NOT excluded
        print("\n--- Excluding serial numbers from the initial target list (with force-include exceptions) ---")
        
        # Calculate the effective exclusion set: those in exclusion_serial_ids MINUS force_include_serials_list
        effective_exclusion_set = exclusion_serial_ids - force_include_serials_list

        serial_numbers_to_eliminate_rows_for = [
            sid for sid in serials_with_bs1_or_bs2 # Filter from the combined list
            if sid not in effective_exclusion_set # Apply the effective exclusion
        ]
        report_data['Serial Numbers Remaining After Exclusion (for Elimination)'] = len(serial_numbers_to_eliminate_rows_for)
        print(f"After exclusion (with exceptions), {len(serial_numbers_to_eliminate_rows_for)} serial numbers remain for row elimination.")


        # --- Step 5: Eliminate all rows corresponding to the remaining serial numbers ---
        print("\n--- Eliminating rows corresponding to the remaining serial numbers ---")
        # Create a boolean mask: True for rows to KEEP, False for rows to ELIMINATE
        rows_to_keep_mask = ~df_source['serial_number_id'].isin(serial_numbers_to_eliminate_rows_for)
        df_final_remaining_rows = df_source[rows_to_keep_mask].copy()

        rows_eliminated_count = len(df_source) - len(df_final_remaining_rows)
        report_data['Rows Eliminated (Corresponding to Remaining Serial Numbers)'] = rows_eliminated_count
        print(f"Eliminated {rows_eliminated_count} rows from the source file.")


        # --- Step 6: Save the final remaining rows to a new Parquet file ---
        print(f"\n--- Saving final remaining rows to Parquet: {final_filtered_parquet_path} ---")
        df_final_remaining_rows.to_parquet(final_filtered_parquet_path, index=False)
        report_data['Total Rows (Final Filtered Parquet)'] = len(df_final_remaining_rows)
        print(f"Final filtered Parquet file saved with {len(df_final_remaining_rows)} rows.")


        # --- Report Step 7: Final Parquet File Statistics ---
        print("\n--- Analyzing Final Parquet File Statistics ---")
        final_book_state_counts = df_final_remaining_rows['book_state'].value_counts().to_dict()
        for state, count in final_book_state_counts.items():
            report_data[f'Final Book State {int(state)} Count'] = count
        
        # Ensure all expected book states (0, 1, 2) are in the report, even if count is 0
        for state in [0, 1, 2]: # Assuming these are the primary book states
            if f'Original Book State {state} Count' not in report_data:
                report_data[f'Original Book State {state} Count'] = 0
            if f'Final Book State {state} Count' not in report_data:
                report_data[f'Final Book State {state} Count'] = 0


        # --- Generate and Save Analysis Report ---
        print("\n--- Generating Exclusion Analysis Report ---")
        report_df = pd.DataFrame.from_dict(report_data, orient='index', columns=['Value'])
        report_df.index.name = 'Metric'
        report_df.to_csv(exclusion_analysis_report_csv_path)
        print(f"Exclusion analysis report saved to: {exclusion_analysis_report_csv_path}")
        print("\n--- Exclusion Analysis Complete ---")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}")
    except KeyError as e:
        print(f"Error: Missing expected column '{e}' in one of the data files. Please check column names.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    # --- Configuration for the NEW filter_and_report_serial_numbers function ---
    source_parquet_file = "/home/alinzk/Forvia_Hella/4_week_v2.parquet"
    
    # IMPORTANT: Replace this with the ACTUAL path to the unique serial number IDs CSV
    # generated by the first script (analyze_data.py) which contains serial_number_id column.
    # Example: "/home/alinzk/Forvia_Hella/4_week_v2_unique_serial_number_ids_filtered_20250722_234527.csv"
    serial_numbers_to_exclude_csv = "/home/alinzk/Forvia_Hella/src/data_exploration/merge/output/4_week_v2_unique_serial_number_ids_filtered_20250722_234527.csv"
    
    # Call the new function. Note: target_book_states_for_filter is now handled internally
    # within the function to always be [1, 2] for the initial identification step.
    filter_and_report_serial_numbers(source_parquet_file, serial_numbers_to_exclude_csv)

    # --- Original calls (commented out) ---
    # original_source_parquet_file = "/home/alinzk/Forvia_Hella/4_week_v2.parquet"
    # unique_serial_numbers_input_csv = "/home/alinzk/Forvia_Hella/4_week_v2_unique_serial_number_ids_filtered_20250722_234527.csv"
    # analyze_serial_numbers(original_source_parquet_file, unique_serial_numbers_input_csv)
