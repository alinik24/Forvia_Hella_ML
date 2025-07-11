# --- Filter columns from the bookings single line dataset ---

import os

import pyarrow.parquet as pq

from src.data_preprocessing.config import BASE_PATH_SL, OUTPUT_DIR_FILTERED_SL
from src.data_preprocessing.utils.utils import filter_parquet_columns


def main():
    bookings_file = os.path.join(BASE_PATH_SL, "bookings_single_line.parquet")
    filtered_output_file = os.path.join(OUTPUT_DIR_FILTERED_SL, "filtered_bookings.parquet")

    # TODO according to lasso
    keep_columns = [
        "book_stamp", "book_state", "booking_id", "serial_number_id",
        "workstep_number_mes", "station_id", "part_group", "created_at"
    ]

    filter_parquet_columns(bookings_file, filtered_output_file, keep_columns)


if __name__ == '__main__':
    main()
