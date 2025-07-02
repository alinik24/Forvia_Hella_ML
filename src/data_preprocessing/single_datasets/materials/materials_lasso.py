from src.data_preprocessing.single_datasets.utils.lasso_utils import *

DATA_PATH = r"D:\Universität\Master\Semester2\RealWorld_ML_Problems\Data\materials_encoded_data.parquet"


def main():
    print("Loading materials dataset...")
    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=['target'])
    y = df['target']

    print("Running Lasso regression on materials dataset...")
    lasso_model = run_lasso(X, y)

    print(f"Best alpha: {lasso_model.alpha_}")
    print(f"Selected features: {(lasso_model.coef_ != 0).sum()} / {len(X.columns)}")

    visualize_lasso_coefficients(lasso_model, X.columns)
    visualize_all_lasso_coefs(lasso_model, X.columns)
    coef_distribution(lasso_model)
    plot_lasso_path(X, y)
    feature_importance_summary(lasso_model, X.columns)

    save_selected_features(lasso_model, X.columns, filename="materials_selected_features.csv")


if __name__ == "__main__":
    main()
