# filter_materials.py

import os

import pyarrow.parquet as pq

from src.data_preprocessing.single_datasets.config import base_path, output_dir
from src.data_preprocessing.utils.utils import filter_parquet_columns
from src.data_preprocessing.utils.preprocessing_utils import load_parquet_dataset

DATA_DIR = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data"
INPUT_FILE = os.path.join(DATA_DIR, "materials_1_week_encoded.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "materials_1_week_encoded.parquet")


def main():
    df = load_parquet_dataset(INPUT_FILE)

    # TODO according to lasso
    keep_columns = [
        "component_position", "component_id", "serial_number_id",
        "station_id", "supplier_id", "mounting_place",
        "container_number", "panel_position", "created_at", "book_state"
    ]

    filter_parquet_columns(INPUT_FILE, OUTPUT_FILE, keep_columns)


if __name__ == '__main__':
    main()