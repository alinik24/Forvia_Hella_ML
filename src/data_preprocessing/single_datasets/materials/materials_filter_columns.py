import pyarrow.parquet as pq

from src.data_preprocessing.config import INPUT_FILE_MATERIALS_1w_enc, INPUT_FILE_MATERIALS_1w_enc_f
from src.data_preprocessing.utils.utils import filter_parquet_columns


def main():
    # TODO according to feature selection
    keep_columns = [
        "component_position", "component_id", "serial_number_id",
        "station_id", "supplier_id", "mounting_place",
        "container_number", "panel_position", "created_at", "book_state"
    ]

    filter_parquet_columns(INPUT_FILE_MATERIALS_1w_enc, INPUT_FILE_MATERIALS_1w_enc_f, keep_columns)


if __name__ == '__main__':
    main()
