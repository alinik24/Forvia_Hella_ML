# ---- Preprocessing of bookmeasmat dataset ----

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.time_filtering import analyze_and_transform_data, create_n_day_dataset


def main():
    reuse_last = input("Reuse last paths? (y/n): ").strip().lower() == "y"

    if reuse_last:
        paths = load_last_run()
        input_path = paths["input_path"]
        output_path = paths["output_path"]
    else:
        input_path = input("Enter desired path for input file: ").strip()
        output_path = input("Enter desired path for output file: ").strip()

    analyze_and_transform_data(input_path)
    create_n_day_dataset(
        n_days=7,
        input_path=input_path,
        output_path=output_path
    )

    save_last_run(input_path=input_path, output_path=output_path, version="v1")


if __name__ == "__main__":
    main()