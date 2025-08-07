from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.model_utils.ridge_utils import *
from src.data_preprocessing.utils.utils import print_nan_columns_info


def main():
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

    X = df.drop(columns=['target'])
    y = df['target']

    print("Reporting missing values...")
    print_nan_columns_info(X)
    print("Cleaning up missing values...")
    X = X.dropna(axis=1, how='any')

    print("Running Ridge classification on bookmeas dataset...")
    ridge_model = run_ridge(X, y)

    print(f"Best alpha: {ridge_model.alpha_}")
    print(f"Coefficient norm: {np.linalg.norm(ridge_model.coef_)}")

    print("Plotting top features by absolute coefficient magnitude...")
    plot_feature_importance(ridge_model, X.columns)

    print("Plotting coefficient path...")
    plot_ridge_path(X, y)

    print("Plotting coefficient values...")
    plot_nonzero_coef_distribution(ridge_model, X.columns)

    print("Plotting distribution of coefficients...")
    plot_nonzero_coef_distribution(ridge_model)

    # Save last input path
    save_last_run(input_path=input_path, output_path="", version="v1")


if __name__ == "__main__":
    main()
