# ---- Balancing of bookings dataset ----
from pathlib import Path

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.balance_utils import balance_by_state_per_serial
from src.data_preprocessing.utils.load_paths import load_last_run


def main():
    print("Loading config...")
    reuse_last = input("Reuse last input/output paths? (y/n): ").strip().lower() == "y"

    if reuse_last:
        paths = load_last_run()
        input_main_path = paths["input_path"]
        output_main_path = paths["output_path"]
    else:
        base_input_dir = input("Enter base input directory: ").strip()
        specific_input_filename = input(
            "Enter specific input filename (e.g. bookmeas_1_week_2025-03-01_to_2025-03-11.parquet): ").strip()
        input_main_path = str(Path(base_input_dir) / specific_input_filename)

        # REVISION: Automatically create the output path with a suffix
        input_path_obj = Path(input_main_path)
        output_filename = f"{input_path_obj.stem}_balanced{input_path_obj.suffix}"
        output_main_path = str(input_path_obj.parent / output_filename)
        print(f"Output path automatically set to: {output_main_path}")

    balance_by_state_per_serial(
        input_main_path=input_main_path,
        output_main_path=output_main_path,
        serial_col="serial_number_id",
        bookstate_col="book_state",
        zero_to_nonzero_ratio=1,  # keep up to x zeros
        random_state=42
    )

    save_last_run(input_path=input_main_path, output_path=output_main_path, version="v1")


if __name__ == "__main__":
    main()
