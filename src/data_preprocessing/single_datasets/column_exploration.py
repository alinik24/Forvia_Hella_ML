import dask.dataframe as dd
import os

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
FILES = {
    "bookings": os.path.join(DATA_DIR, "data_hella_single_line", "bookings_single_line.parquet"),
    "materials": os.path.join(DATA_DIR, "data_hella_single_line", "materials_single_line.parquet"),
    "measurements": os.path.join(DATA_DIR, "data_hella_single_line", "measurements_single_line.parquet")
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
