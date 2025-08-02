# processing.py

import pandas as pd
import os
import sys

# Add parent directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
single_datasets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'single_datasets'))
if single_datasets_dir not in sys.path:
    sys.path.append(single_datasets_dir)

import config

def perform_dynamic_exclusion(source_file_path, stage2_summary_csv_path):
    """
    Performs dynamic exclusion based on the new anomaly detection logic (IQR method)
    and generates a comprehensive report.

    Args:
        source_file_path (str): Path to the original source Parquet file.
        stage2_summary_csv_path (str): Path to the summary CSV from Stage 2.
    """
    print("\nStarting Stage 3: Performing Dynamic Exclusion and Final Reporting")
    report_data = {}

    try:
        # Load the original source data for "before" metrics
        df_source = pd.read_parquet(source_file_path)
        # Preserve original dtypes to be used later
        original_dtypes = df_source.dtypes
        # Ensure data types are consistent for merging and filtering
        df_source['serial_number_id'] = df_source['serial_number_id'].astype(str)
        df_source['book_state'] = pd.to_numeric(df_source['book_state'], errors='coerce')
        df_source['station_desc'] = df_source['station_desc'].astype('category')

        # Load the summary report from Stage 2
        df_summary = pd.read_csv(stage2_summary_csv_path)
        df_summary['serial_number_id'] = df_summary['Serial Number ID'].astype(str)

        # --- Anomaly Detection Logic (IQR Method) ---
        book_state_1_col = 'Book State 1 Count'
        
        if book_state_1_col not in df_summary.columns:
            print(f"Error: Missing expected column '{book_state_1_col}' in summary file. Cannot proceed with anomaly detection.")
            return

        q1 = df_summary[book_state_1_col].quantile(0.25)
        q3 = df_summary[book_state_1_col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        print(f"Calculated Q1: {q1}, Q3: {q3}, IQR: {iqr}, Upper Bound: {upper_bound:.2f}")

        anomalous_serials = df_summary[
            df_summary[book_state_1_col] > upper_bound
        ]['serial_number_id'].tolist()

        anomalies_df = pd.DataFrame(anomalous_serials, columns=['Anomalous Serial Number ID'])
        anomalies_df.to_csv(config.ANOMALOUS_SERIAL_NUMBERS_CSV, index=False)
        print(f"Anomalous serial number IDs saved to: {config.ANOMALOUS_SERIAL_NUMBERS_CSV}")
        
        # --- Final Filtering and Parquet File Generation ---
        df_final_filtered = df_source[
            ~df_source['serial_number_id'].isin(anomalous_serials)
        ].copy()
        
        # Restore original data types
        df_final_filtered = df_final_filtered.astype(original_dtypes)

        # Save the final filtered data
        df_final_filtered.to_parquet(config.FINAL_FILTERED_PARQUET, index=False)
        print(f"Final filtered data saved to: {config.FINAL_FILTERED_PARQUET}")

        # --- Generate the Comprehensive Report with new metrics ---
        report_data['Metric'] = []
        report_data['Value'] = []

        # --- Original Data (Before Filtering) ---
        report_data['Metric'].append('--- Source Data (Before Exclusion) ---')
        report_data['Value'].append('')
        report_data['Metric'].append('Total Rows')
        report_data['Value'].append(len(df_source))
        report_data['Metric'].append('Unique serial number IDs')
        report_data['Value'].append(df_source['serial_number_id'].nunique())
        report_data['Metric'].append('Unique station_desc')
        report_data['Value'].append(df_source['station_desc'].nunique())
        
        original_book_state_counts = df_source['book_state'].value_counts().to_dict()
        for state in sorted(original_book_state_counts.keys()):
            report_data['Metric'].append(f'Book State {int(state)} Row Count (Original)')
            report_data['Value'].append(original_book_state_counts[state])

        # New metrics for original data
        serials_with_state_1_orig = set(df_source[df_source['book_state'] == 1]['serial_number_id'].unique())
        serials_with_state_2_orig = set(df_source[df_source['book_state'] == 2]['serial_number_id'].unique())
        
        report_data['Metric'].append('Total Unique IDs with either book state 1 or 2 (Original)')
        report_data['Value'].append(len(serials_with_state_1_orig.union(serials_with_state_2_orig)))
        
        report_data['Metric'].append('Total Unique IDs with both book state 1 and 2 (Original)')
        report_data['Value'].append(len(serials_with_state_1_orig.intersection(serials_with_state_2_orig)))
        
        report_data['Metric'].append('Total Unique IDs with only book state 1 (Original)')
        report_data['Value'].append(len(serials_with_state_1_orig.difference(serials_with_state_2_orig)))

        # --- Final Filtered Data (After Filtering) ---
        report_data['Metric'].append('--- Final Filtered Data (After Exclusion) ---')
        report_data['Value'].append('')
        report_data['Metric'].append('Total Rows')
        report_data['Value'].append(len(df_final_filtered))
        report_data['Metric'].append('Unique serial number IDs')
        report_data['Value'].append(df_final_filtered['serial_number_id'].nunique())
        report_data['Metric'].append('Unique station_desc')
        report_data['Value'].append(df_final_filtered['station_desc'].nunique())

        final_book_state_counts = df_final_filtered['book_state'].value_counts().to_dict()
        for state in sorted(final_book_state_counts.keys()):
            report_data['Metric'].append(f'Book State {int(state)} Row Count (Final Filtered)')
            report_data['Value'].append(final_book_state_counts[state])
        
        # New metrics for final filtered data
        serials_with_state_1_final = set(df_final_filtered[df_final_filtered['book_state'] == 1]['serial_number_id'].unique())
        serials_with_state_2_final = set(df_final_filtered[df_final_filtered['book_state'] == 2]['serial_number_id'].unique())
        
        report_data['Metric'].append('Total Unique IDs with either book state 1 or 2 (Final Filtered)')
        report_data['Value'].append(len(serials_with_state_1_final.union(serials_with_state_2_final)))
        
        report_data['Metric'].append('Total Unique IDs with both book state 1 and 2 (Final Filtered)')
        report_data['Value'].append(len(serials_with_state_1_final.intersection(serials_with_state_2_final)))
        
        report_data['Metric'].append('Total Unique IDs with only book state 1 (Final Filtered)')
        report_data['Value'].append(len(serials_with_state_1_final.difference(serials_with_state_2_final)))

        # --- Summary of Exclusions ---
        report_data['Metric'].append('--- Summary of Exclusions ---')
        report_data['Value'].append('')
        report_data['Metric'].append('Total Rows Eliminated')
        report_data['Value'].append(len(df_source) - len(df_final_filtered))
        report_data['Metric'].append('Serial Number IDs Eliminated')
        report_data['Value'].append(df_source['serial_number_id'].nunique() - df_final_filtered['serial_number_id'].nunique())
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(config.DYNAMIC_EXCLUSION_REPORT, index=False)

        print(f"Dynamic exclusion complete. Final filtered data saved to: {config.FINAL_FILTERED_PARQUET}")
        print(f"Exclusion report saved to: {config.DYNAMIC_EXCLUSION_REPORT}")

    except Exception as e:
        print(f"An error occurred during Stage 3 processing: {e}")
