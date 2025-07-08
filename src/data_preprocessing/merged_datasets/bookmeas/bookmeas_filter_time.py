# ---- Preprocessing of materials dataset ----
import os

from src.data_preprocessing.utils.time_filtering import analyze_and_transform_data, create_n_day_dataset

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "bookmeas", "4_week.parquet")
OUTPUT_FILE_1_WEEK = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week.parquet")
OUTPUT_FILE_2_WEEKS = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks.parquet")

if __name__ == "__main__":
    analyze_and_transform_data(INPUT_FILE)
    #create_n_day_dataset(n_days=7, input_path=INPUT_FILE, output_path=OUTPUT_FILE_1_WEEK)
    #create_n_day_dataset(n_days=14, input_path=INPUT_FILE, output_path=OUTPUT_FILE_2_WEEKS)
