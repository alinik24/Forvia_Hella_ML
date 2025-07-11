# ---- Preprocessing of materials dataset ----

from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_4w, OUTPUT_FILE_BOOKMEAS_1w, OUTPUT_FILE_BOOKMEAS_2w
from src.data_preprocessing.utils.time_filtering import analyze_and_transform_data, create_n_day_dataset

if __name__ == "__main__":
    #analyze_and_transform_data(INPUT_FILE_BOOKMEAS_4w)
    #create_n_day_dataset(n_days=7, input_path=INPUT_FILE_BOOKMEAS_4w, output_path=OUTPUT_FILE_BOOKMEAS_1w)
    create_n_day_dataset(n_days=14, input_path=INPUT_FILE_BOOKMEAS_4w, output_path=OUTPUT_FILE_BOOKMEAS_2w)
