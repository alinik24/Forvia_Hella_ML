from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_2w, INPUT_FILE_MATERIALS_2w, OUTPUT_FILE_BOOKMEAS_2w_b, \
    OUTPUT_FILE_MATERIALS_2w_b
from src.data_preprocessing.utils.balance_utils import balance_datasets

if __name__ == "__main__":
    balance_datasets(
        input_materials_path=INPUT_FILE_MATERIALS_2w,
        input_bookmeas_path=INPUT_FILE_BOOKMEAS_2w,
        output_materials_path=OUTPUT_FILE_MATERIALS_2w_b,
        output_bookmeas_path=OUTPUT_FILE_BOOKMEAS_2w_b,
        factor=5
    )
