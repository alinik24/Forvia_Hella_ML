from src.data_preprocessing.config import save_last_run
from src.data_preprocessing.utils.load_paths import load_last_run
from src.data_preprocessing.utils.model_utils.lasso_utils import *
from src.data_preprocessing.utils.utils import print_nan_columns_info


def main():
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

    X = df.drop(columns=['target'])
    X = X.drop(columns=['has_failures'])
    y = df['target']

    print("Reporting missing values...")
    print_nan_columns_info(X)
    print("Cleaning up missing values...")
    X = X.dropna(axis=1, how='any')

    print("Running Lasso regression on bookmeasmat dataset...")
    lasso_model = run_lasso(X, y)

    print(f"Best alpha: {lasso_model.alpha_}")
    print(f"Selected features: {(lasso_model.coef_ != 0).sum()} / {len(X.columns)}")

    print("Plotting cross-validation MSE curve...")
    plot_cv_mse(lasso_model)

    print("Plotting top features by absolute coefficient magnitude...")
    plot_feature_importance(lasso_model, X.columns)

    print("Plotting coefficient path...")
    plot_lasso_path(X, y)

    print("Plotting coefficient values including zeros...")
    plot_coefficients(lasso_model, X.columns)

    print("Plotting distribution of non-zero coefficients...")
    plot_nonzero_coef_distribution(lasso_model)

    print("Plotting all non-zero coefficients sorted by magnitude...")
    plot_all_nonzero_coefs_sorted(lasso_model, X.columns)

    # Save last input path
    save_last_run(input_path=input_path, output_path="", version="v1")


if __name__ == "__main__":
    main()
