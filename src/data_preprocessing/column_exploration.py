import dask.dataframe as dd
from pathlib import Path
from src.data_preprocessing.utils.load_paths import load_paths


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


def get_latest_final_filtered_file(output_dir: str, base_filename: str) -> Path:
    pattern = f"{base_filename}_final_filtered_*.parquet"
    output_path = Path(output_dir) / "bookmeas"
    candidates = sorted(output_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(f"No final filtered files found in {output_path} with pattern {pattern}")

    return candidates[0]


def main():
    # Load paths from YAML config interactively
    paths = load_paths()

    # Dynamically find latest Stage 3 output
    latest_final_filtered_file = get_latest_final_filtered_file(paths["output_dir"], "4_week")

    # Example: Inspect bookmeas filtered parquet path dynamically from config
    # NOTE: You can extend this dict with more datasets if needed
    files_to_inspect = {
        "Latest Final Filtered (Stage 3 output)": str(latest_final_filtered_file),
    }

    for name, file_path in files_to_inspect.items():
        # Check if path exists (handle path as string or Path object)
        if not Path(file_path).exists():
            print(f"Warning: File {file_path} for dataset '{name}' does not exist.")
            continue

        print(f"--- Dataset: {name} ---")
        inspect_parquet_with_dask(file_path)


if __name__ == '__main__':
    main()
