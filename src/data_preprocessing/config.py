# config.py
import os

# Base directory for all datasets
#TODO Change to your path
DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"

# Bookmeas dataset input files
INPUT_FILE_BOOKMEAS_1w = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week.parquet")
INPUT_FILE_BOOKMEAS_1w_b = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week_balanced.parquet")
INPUT_FILE_BOOKMEAS_1w_enc = os.path.join(DATA_DIR, "bookmeas_1_week_encoded.parquet")
INPUT_FILE_BOOKMEAS_1w_enc_f = os.path.join(DATA_DIR, "bookmeas_1_week_filtered.parquet")
INPUT_FILE_BOOKMEAS_2w = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks.parquet")
INPUT_FILE_BOOKMEAS_2w_enc = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks_encoded.parquet")
INPUT_FILE_BOOKMEAS_2w_enc_unbalanced = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks_encoded_unbalanced.parquet")
INPUT_FILE_BOOKMEAS_2w_b = os.path.join(DATA_DIR, "bookmeas_2_weeks_balanced.parquet")
INPUT_FILE_BOOKMEAS_4w = os.path.join(DATA_DIR, "bookmeas", "4_week.parquet")

# Bookings single line dataset input files
INPUT_FILE_BOOKINGS_SL = os.path.join(DATA_DIR, "data_hella_single_line", "bookings_single_line.parquet")
INPUT_FILE_BOOKINGS_SL_enc = os.path.join(DATA_DIR, "bookings_encoded.parquet")
INPUT_FILE_BOOKINGS_SL_enc_f = os.path.join(DATA_DIR, "bookings_filtered.parquet")

# Materials dataset input files
INPUT_FILE_MATERIALS_1w = os.path.join(DATA_DIR, "materials", "materials_1_week.parquet")
INPUT_FILE_MATERIALS_1w_enc = os.path.join(DATA_DIR, "materials_1_week_encoded.parquet")
INPUT_FILE_MATERIALS_1w_enc_f = os.path.join(DATA_DIR, "materials_1_week_filtered.parquet")
INPUT_FILE_MATERIALS_2w = os.path.join(DATA_DIR, "materials", "materials_2_weeks.parquet")
INPUT_FILE_MATERIALS_2w_b = os.path.join(DATA_DIR, "materials_2_weeks_balanced.parquet")
INPUT_FILE_MATERIALS_2w_enc = os.path.join(DATA_DIR, "materials_2_weeks_encoded.parquet")
INPUT_FILE_MATERIALS_4w = os.path.join(DATA_DIR, "materials", "materials_4_weeks.parquet")

# Measurements single line dataset input files
INPUT_FILE_MEAS_SL = os.path.join(DATA_DIR, "data_hella_single_line", "measurements_single_line.parquet")
INPUT_FILE_MEAS_SL_enc = os.path.join(DATA_DIR, "measurements_encoded.parquet")
INPUT_FILE_MEAS_SL_enc_f = os.path.join(DATA_DIR, "measurements_filtered.parquet")

# Measurements single line dataset output files
OUTPUT_FILE_MEAS_SL_enc = os.path.join(DATA_DIR, "measurements", "measurements_encoded.parquet")
OUTPUT_FILE_MEAS_SL_b = os.path.join(DATA_DIR, "measurements", "measurements_balanced.parquet")
OUTPUT_FILE_MEAS_SL_enc_f = os.path.join(DATA_DIR, "measurements", "measurements_filtered.parquet")

# Materials dataset output files
OUTPUT_FILE_MATERIALS_1w = os.path.join(DATA_DIR, "materials", "materials_1_week.parquet")
OUTPUT_FILE_MATERIALS_1w_enc = os.path.join(DATA_DIR, "materials_1_week_encoded.parquet")
OUTPUT_FILE_MATERIALS_1w_enc_f = os.path.join(DATA_DIR, "materials_1_week_filtered.parquet")
OUTPUT_FILE_MATERIALS_2w = os.path.join(DATA_DIR, "materials", "materials_2_weeks.parquet")
OUTPUT_FILE_MATERIALS_2w_enc = os.path.join(DATA_DIR, "materials_2_weeks_encoded.parquet")
OUTPUT_FILE_MATERIALS_2w_enc_f = os.path.join(DATA_DIR, "materials_2_weeks_filtered.parquet")
OUTPUT_FILE_MATERIALS_2w_b = os.path.join(DATA_DIR, "materials_2_weeks_balanced.parquet")

# Bookings single line dataset output files
OUTPUT_FILE_BOOKINGS_SL_b = os.path.join(DATA_DIR, "bookings", "bookings_balanced.parquet")
OUTPUT_FILE_BOOKINGS_SL_enc = os.path.join(DATA_DIR, "bookings", "bookings_encoded.parquet")
OUTPUT_FILE_BOOKINGS_SL_f = os.path.join(DATA_DIR, "bookings",  "bookings_filtered.parquet")

# Correlation results files
OUTPUT_BOOKMEAS_FILE_PEARSON = os.path.join(DATA_DIR, "bookmeas_cleaned_pearson.parquet")
OUTPUT_BOOKMEAS_FILE_SPEARMAN = os.path.join(DATA_DIR, "bookmeas_cleaned_spearman.parquet")
OUTPUT_BOOKMEASMAT_FILE_PEARSON = os.path.join(DATA_DIR, "bookmeasmat_cleaned_pearson.parquet")
OUTPUT_BOOKMEASMAT_FILE_SPEARMAN = os.path.join(DATA_DIR, "bookmeasmat_cleaned_spearman.parquet")

# Bookmeas dataset output files
OUTPUT_FILE_BOOKMEAS_1w = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week.parquet")
OUTPUT_FILE_BOOKMEAS_1w_b = os.path.join(DATA_DIR, "bookmeas", "bookmeas_1_week_balanced.parquet")
OUTPUT_FILE_BOOKMEAS_1w_enc = os.path.join(DATA_DIR, "bookmeas_1_week_encoded.parquet")
OUTPUT_FILE_BOOKMEAS_1w_enc_f = os.path.join(DATA_DIR, "bookmeas_1_week_filtered.parquet")
OUTPUT_FILE_BOOKMEAS_2w = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks.parquet")
OUTPUT_FILE_BOOKMEAS_2w_enc = os.path.join(DATA_DIR, "bookmeas", "bookmeas_2_weeks_encoded.parquet")
OUTPUT_FILE_BOOKMEAS_2w_enc_f = os.path.join(DATA_DIR, "bookmeas_2_weeks_filtered.parquet")
OUTPUT_FILE_BOOKMEAS_2w_b = os.path.join(DATA_DIR, "bookmeas_2_weeks_balanced.parquet")

# Bookmeasmat dataset output files
FINAL_FILE_BOOKMEASMAT_2w = os.path.join(DATA_DIR, "final_bookmeasmat_2_weeks.parquet")
FINAL_FILE_BOOKMEASMAT_2w_b = os.path.join(DATA_DIR, "final_bookmeasmat_2_weeks.parquet")
FINAL_FILE_BOOKMEASMAT_2w_enc = os.path.join(DATA_DIR, "final_bookmeasmat_2_weeks_encoded.parquet")

# Paths for saving skrubs vectorizers
VECTORIZER_PATH_MEAS = os.path.join(DATA_DIR, "meas_vectorizer.pkl")
VECTORIZER_PATH_BOOKINGS = os.path.join(DATA_DIR, "bookings_vectorizer.pkl")
VECTORIZER_PATH_MATERIALS = os.path.join(DATA_DIR, "materials_vectorizer.pkl")
VECTORIZER_PATH_BOOKMEAS = os.path.join(DATA_DIR, "bookmeas_vectorizer.pkl")
VECTORIZER_PATH_BOOKMEASMAT = os.path.join(DATA_DIR, "bookmeasmat_vectorizer.pkl")

# Heatmap pictures
HEATMAP_FILE_BOOKINGS = os.path.join(DATA_DIR, "bookings_correlation_heatmap.png")
HEATMAP_FILE_MATERIALS= os.path.join(DATA_DIR, "materials_correlation_heatmap.png")
HEATMAP_FILE_MEAS = os.path.join(DATA_DIR, "meas_correlation_heatmap.png")