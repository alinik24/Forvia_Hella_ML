from src.data_preprocessing.config import FINAL_FILE_BOOKMEASMAT_2w, FINAL_FILE_BOOKMEASMAT_2w_b
from src.data_preprocessing.utils.balance_utils import balance_by_state_per_serial


# TODO
def main():
    balance_by_state_per_serial(
        input_main_path=FINAL_FILE_BOOKMEASMAT_2w,
        output_main_path=FINAL_FILE_BOOKMEASMAT_2w_b,
        serial_col="serial_number_id",
        bookstate_col="book_state",
        zero_to_nonzero_ratio=2,  # keep up to x zeros
        random_state=42
    )


if __name__ == "__main__":
    main()
