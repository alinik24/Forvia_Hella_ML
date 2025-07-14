from src.data_preprocessing.config import INPUT_FILE_MATERIALS_2w, OUTPUT_FILE_MATERIALS_2w_b
from src.data_preprocessing.utils.balance_utils import balance_by_state_per_serial

def main():
    balance_by_state_per_serial(
        input_main_path=INPUT_FILE_MATERIALS_2w,
        output_main_path=OUTPUT_FILE_MATERIALS_2w_b,
        serial_col="serial_number_id",
        bookstate_col="book_state",
        zero_to_nonzero_ratio=2,  # keep up to x zeros
        random_state=42
    )

if __name__ == "__main__":
    main()