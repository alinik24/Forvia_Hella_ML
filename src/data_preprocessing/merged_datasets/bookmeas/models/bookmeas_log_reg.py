import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.model_utils.log_reg_utils import *


def main():
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

    # Separate features and target
    X = df.drop(columns=['target'])
    columns_to_drop = ['has_failures', 'sequence_number']
    for column in columns_to_drop:
        if column in X.columns:
            X = X.drop(columns=[column])
    y = df['target']

    # Split into training and testing sets
    print("Creating train and test splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )

    # Run logistic regression with the scaled features
    print("Running logistic regression model...")
    # best_model = run_logistic_regression_gridsearch(X_train, y_train, X_test, y_test)
    best_model = run_logistic_regression(X_train, y_train, X_test, y_test)

    # Plot coefficients for each class
    plot_coefficients(best_model, X_train.columns, class_index=0)
    plot_coefficients(best_model, X_train.columns, class_index=1)
    plot_coefficients(best_model, X_train.columns, class_index=2)

    # Save last input path
    save_last_run(input_path=input_path, output_path="", version="v1")


if __name__ == "__main__":
    main()
