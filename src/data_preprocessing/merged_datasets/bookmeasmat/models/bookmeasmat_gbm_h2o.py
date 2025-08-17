import h2o
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.model_utils.gbm_H2O_utils import plot_feature_importance
from src.data_preprocessing.utils.model_utils.gbm_H2O_utils import train_gbm_model


def main():
    # NOTE: if it doesnt start at the first try, try again !!!
    h2o.init(max_mem_size_GB=28)

    # Load and split data
    print("Loading bookmeasmat dataset...")
    reuse_last = input("Reuse last input path? (y/n): ").strip().lower() == "y"

    if reuse_last:
        last_paths = load_last_run()
        input_path = last_paths.get("input_path")
        if not input_path:
            print("No last input path found in config. Please enter manually.")
            input_path = input("Enter path to encoded bookmeasmat dataset (parquet): ").strip()
    else:
        input_path = input("Enter path to encoded bookmeasmat dataset (parquet): ").strip()

    print(f"Loading bookmeasmat dataset from {input_path} ...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model, features = train_gbm_model(X_train, y_train, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(model)

    # Save last input path
    save_last_run(input_path=input_path, output_path="", version="v1")

    # Shutdown H2O
    h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
