from src.data_preprocessing.single_datasets.utils.lasso_utils import *

DATA_PATH = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data\measurements_encoded_data.parquet"


def main():
    print("Loading measurements dataset...")
    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=['target'])
    y = df['target']

    print("Running Lasso regression on measurements dataset...")
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


if __name__ == "__main__":
    main()
