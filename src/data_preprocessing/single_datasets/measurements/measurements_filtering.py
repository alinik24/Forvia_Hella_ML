# filter_measurements.py

import os

import pyarrow.parquet as pq
from src.data_preprocessing.single_datasets.config import base_path, output_dir

from src.data_preprocessing.utils.utils import filter_parquet_columns


def main():
    measurements_file = os.path.join(base_path, "measurements_single_line.parquet")
    filtered_output_file = os.path.join(output_dir, "filtered_measurements.parquet")

    # TODO according to lasso
    keep_columns = [
        "measure_step_number", "measure_value", "created_at", "booking_id",
        "book_state", "serial_number_id",
        "station_id", "measurement_name", "measurement_unit",
        "lower_limit", "upper_limit"
    ]

    filter_parquet_columns(measurements_file, filtered_output_file, keep_columns)


if __name__ == '__main__':
    main()
