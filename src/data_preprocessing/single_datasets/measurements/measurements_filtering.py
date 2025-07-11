# filter_measurements.py

import pyarrow.parquet as pq

from src.data_preprocessing.config import INPUT_FILE_MEAS_SL, OUTPUT_FILE_MEAS_SL_enc_f
from src.data_preprocessing.utils.utils import filter_parquet_columns


def main():

    # TODO according to lasso
    keep_columns = [
        "measure_step_number", "measure_value", "created_at", "booking_id",
        "book_state", "serial_number_id",
        "station_id", "measurement_name", "measurement_unit",
        "lower_limit", "upper_limit"
    ]

    filter_parquet_columns(INPUT_FILE_MEAS_SL, OUTPUT_FILE_MEAS_SL_enc_f, keep_columns)


if __name__ == '__main__':
    main()
