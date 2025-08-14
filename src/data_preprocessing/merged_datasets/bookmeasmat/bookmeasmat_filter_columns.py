from pathlib import Path

import pyarrow.parquet as pq

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.utils import filter_parquet_columns_contains


def main():
    print("=== Column Filtering for Encoded Bookmeas Dataset ===")
    reuse_last = input("Reuse last input path? (y/n): ").strip().lower() == "y"

    if reuse_last:
        last_paths = load_last_run()
        input_path = last_paths.get("input_path")
        if not input_path:
            print("No last input path found in config. Please enter manually.")
            input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()
    else:
        input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()

    input_path = Path(input_path).resolve()
    output_path = input_path.with_name(input_path.stem + "_filtered.parquet")

    # Base column names to preserve
    base_keep_columns = [
        "booking_id",
        "book_state",
        "serial_number_id",
        "station_id",
        "station_diag_id",
        "workorder_id",
        "part_group",
        "erp_group_id",
        "workstep_id",
        "line_id",
        "part_desc",
        "workorder_desc",
        "measure_value",
        "catalog_id",
        "teststep_id",
        "measurement_name",
        "measurement_unit",
        "measurement_type",
        "lower_limit",
        "upper_limit",
        "target",
        "component_id",
        "supplier_id",
        "container_number",
        "supplier_order_date_code",
        "supplier_order_number"
    ]

    print(f"Filtering columns from: {input_path}")
    try:
        filter_parquet_columns_contains(str(input_path), str(output_path), base_keep_columns)
    except Exception as e:
        print(f"Error filtering dataset: {e}")
        return

    # Save last input path
    save_last_run(input_path=str(input_path), output_path=str(output_path), version="v1")


if __name__ == '__main__':
    main()