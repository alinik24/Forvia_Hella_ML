# --- Filter columns from the bookings single line dataset ---

import os

import pyarrow.parquet as pq

from src.data_preprocessing.single_datasets.config import base_path, output_dir
from src.data_preprocessing.utils.utils import filter_parquet_columns


def main():
    bookings_file = os.path.join(base_path, "bookings_single_line.parquet")
    filtered_output_file = os.path.join(output_dir, "filtered_bookings.parquet")

    # TODO according to lasso
    keep_columns = [
        "book_stamp", "book_state", "booking_id", "serial_number_id",
        "workstep_number_mes", "station_id", "part_group", "created_at"
    ]

    filter_parquet_columns(bookings_file, filtered_output_file, keep_columns)


if __name__ == '__main__':
    main()
