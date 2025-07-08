import dask.dataframe as dd
import os

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
FILES = {
    #"bookings": os.path.join(DATA_DIR, "data_hella_single_line", "bookings_single_line.parquet"),
    "materials": os.path.join(DATA_DIR, "materials", "materials_1_week.parquet"),
    #"materialsenc": os.path.join(DATA_DIR, "materials_1_week_encoded.parquet"),
    #"measurements": os.path.join(DATA_DIR, "data_hella_single_line", "measurements_single_line.parquet"),
    #"bookmeas": os.path.join(DATA_DIR, "bookmeas", "1_week.parquet")
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


