import sys
from src.data_preprocessing.config import INPUT_FILE_MATERIALS_2w_b, INPUT_FILE_BOOKMEAS_2w_b, FINAL_FILE_BOOKMEASMAT_2w
from src.data_preprocessing.utils.merging_utils import merge_balanced_datasets

if __name__ == "__main__":
    try:
        merge_balanced_datasets(
            materials_path=INPUT_FILE_MATERIALS_2w_b,
            bookmeas_path=INPUT_FILE_BOOKMEAS_2w_b,
            output_path=FINAL_FILE_BOOKMEASMAT_2w
        )
    except Exception as e:
        print(f"!Fatal error in main process: {str(e)}!")
        sys.exit(1)
