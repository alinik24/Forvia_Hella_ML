from src.data_preprocessing.config import INPUT_FILE_BOOKINGS_SL_enc
from src.data_preprocessing.utils.model_utils.log_reg_utils import *


def main():
    print("Loading bookings dataset...")
    df = pd.read_parquet(INPUT_FILE_BOOKINGS_SL_enc)

    X = df.drop(columns=['target'])
    X = X.drop(columns=['has_failures'])
    y = df['target']

    print("Running Lasso regression on bookings dataset...")
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
