import pandas as pd
from sklearn.model_selection import train_test_split

import h2o
from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.model_utils.gbm_H2O_utils import plot_feature_importance
from src.data_preprocessing.utils.model_utils.gbm_H2O_utils import train_gbm_model

from src.data_preprocessing.utils.model_utils.h2o_utils import (
    initialize_h2o_cluster,
)


def an():
    # Start H2O cluster
    h2o.init(max_mem_size_GB=24)

    # Load and split data
    print("Loading bookmeas dataset...")
    reuse_last = input("Reuse last input path? (y/n): ").strip().lower() == "y"

    if reuse_last:
        last_paths = load_last_run()
        input_path = last_paths.get("input_path")
        if not input_path:
            print("No last input path found in config. Please enter manually.")
            input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()
    else:
        input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()

    print(f"Loading bookmeas dataset from {input_path} ...")
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

    # Shutdown H2O (optional)
    h2o.shutdown(prompt=False)


def main():
    initialize_h2o_cluster(max_mem_gb=14, max_retries=5)
    #h2o.init(ip="localhost", port="8080", max_mem_size_GB=24)
    h2o.demo("glm")


if __name__ == "__main__":
    main()
