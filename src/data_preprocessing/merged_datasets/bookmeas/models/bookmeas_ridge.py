from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_2w_enc
from src.data_preprocessing.utils.ridge_utils import *
from src.data_preprocessing.utils.utils import print_nan_columns_info


def main():
    print("Loading bookmeas dataset...")
    df = pd.read_parquet(INPUT_FILE_BOOKMEAS_2w_enc)

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


if __name__ == "__main__":
    main()
