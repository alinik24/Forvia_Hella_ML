from src.data_preprocessing.config import save_last_run


def main():
    input_path = input("Enter full path to input dataset: ").strip()
    output_path = input("Enter full path to output destination: ").strip()
    version = input("Enter version name [default: v1]: ").strip() or "v1"

    save_last_run(input_path, output_path, version)


if __name__ == "__main__":
    main()
