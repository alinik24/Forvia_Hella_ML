import dask.dataframe as dd
import os
from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_4w_snid_f, INPUT_FILE_BOOKINGS_SL, INPUT_FILE_MATERIALS_2w, INPUT_FILE_MEAS_SL, INPUT_FILE_MATERIALS_1w_enc, INPUT_FILE_BOOKMEAS_2w, FINAL_FILE_BOOKMEASMAT_2w

FILES = {
    #"bookings": INPUT_FILE_BOOKINGS_SL,
    #"materials": INPUT_FILE_MATERIALS_2w,
    #"materialsenc": INPUT_FILE_MATERIALS_1w_enc,
    #"measurements": INPUT_FILE_MEAS_SL,
    #"bookmeas": INPUT_FILE_BOOKMEAS_2w,
    #"bookmeasmat": FINAL_FILE_BOOKMEASMAT_2w,
    "bookmeas_filtered": INPUT_FILE_BOOKMEAS_4w_snid_f
}

def inspect_parquet_with_dask(file_path, sample_size=5):
    print(f"Inspecting {file_path}...\n")
    df = dd.read_parquet(file_path)

    print("Columns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSample data:")
    print(df.head(sample_size))

    print("\n" + "="*80 + "\n")

def main():
    for name, path in FILES.items():
        print(f"--- Dataset: {name} ---")
        inspect_parquet_with_dask(path)

if __name__ == '__main__':
    main()


