from pathlib import Path

import dask.dataframe as dd


def inspect_parquet_with_dask(file_path, sample_size=5):
    print(f"Inspecting {file_path}...\n")
    df = dd.read_parquet(file_path)

    print("Columns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSample data:")
    print(df.head(sample_size))

    # === Book State and Serial Number Summary ===
    if "book_state" in df.columns:
        print("\nBook State Counts:")
        book_state_counts = df["book_state"].value_counts().compute()
        print(book_state_counts)

    if "serial_number_id" in df.columns:
        print("\nUnique Serial Numbers:")
        unique_serials = df["serial_number_id"].nunique().compute()
        print(f"Total unique serial numbers: {unique_serials}")

    print("\n" + "=" * 100 + "\n")


def main():
    # Ask user for parquet path
    file_path = input("Enter path to parquet file: ").strip()
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    inspect_parquet_with_dask(str(file_path))


if __name__ == '__main__':
    main()
