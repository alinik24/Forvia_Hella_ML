from src.data_preprocessing.utils.bs_analysis_utils import analyze_and_extract_data
from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_4w

# --- Main execution ---
if __name__ == "__main__":
    # Define the path to the input Parquet file
    input_parquet_file = INPUT_FILE_BOOKMEAS_4w

    # Define the book_state value to filter by
    target_state = 2

    # Call the main analysis function
    analyze_and_extract_data(input_parquet_file, target_state)
