from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_2w_enc, INPUT_FILE_BOOKMEAS_1w_enc
from src.data_preprocessing.utils.log_reg_utils import *
from sklearn.model_selection import train_test_split


def main():
    print("Loading bookmeas dataset...")
    # Load only the 2-week encoded data
    data = pd.read_parquet(INPUT_FILE_BOOKMEAS_2w_enc)

    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y              # Maintain class distribution
    )

    # Run logistic regression with the scaled features
    #best_model = run_logistic_regression_gridsearch(X_train, y_train, X_test, y_test)
    best_model = run_logistic_regression(X_train, y_train, X_test, y_test)

    # Plot coefficients for each class
    plot_coefficients(best_model, X_train.columns, class_index=0)
    plot_coefficients(best_model, X_train.columns, class_index=1)
    plot_coefficients(best_model, X_train.columns, class_index=2)


if __name__ == "__main__":
    main()
