# ---- Preprocessing of materials dataset ----

from src.data_preprocessing.config import INPUT_FILE_MATERIALS_4w, INPUT_FILE_MATERIALS_1w, INPUT_FILE_MATERIALS_2w
from src.data_preprocessing.utils.time_filtering import analyze_and_transform_data, create_n_day_dataset

if __name__ == "__main__":
    #analyze_and_transform_data(INPUT_FILE_MATERIALS_4w)
    #create_n_day_dataset(n_days=7, input_path=INPUT_FILE_MATERIALS_4w, output_path=INPUT_FILE_MATERIALS_1w)
    create_n_day_dataset(n_days=11, input_path=INPUT_FILE_MATERIALS_4w, output_path=INPUT_FILE_MATERIALS_2w) # Only 11 days available
