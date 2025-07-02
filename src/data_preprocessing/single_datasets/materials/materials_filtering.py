# filter_materials.py

import os

import pyarrow.parquet as pq

from src.data_preprocessing.single_datasets.config import base_path, output_dir
from src.data_preprocessing.single_datasets.utils.utils import filter_parquet_columns


def main():
    materials_file = os.path.join(base_path, "materials_single_line.parquet")
    filtered_output_file = os.path.join(output_dir, "filtered_materials.parquet")

    # TODO according to lasso
    keep_columns = [
        "component_position", "component_id", "serial_number_id",
        "station_id", "supplier_id", "mounting_place",
        "container_number", "panel_position", "created_at", "book_state"
    ]

    filter_parquet_columns(materials_file, filtered_output_file, keep_columns)


if __name__ == '__main__':
    main()