from src.data_preprocessing.config import INPUT_FILE_BOOKMEAS_2w_enc, INPUT_FILE_BOOKMEAS_1w_enc
from src.data_preprocessing.utils.log_reg_utils import *


def main():
    print("Loading bookmeas dataset...")
    train_data = pd.read_parquet(INPUT_FILE_BOOKMEAS_2w_enc)
    test_data = pd.read_parquet(INPUT_FILE_BOOKMEAS_1w_enc)

    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']

    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']

    # Run logistic regression with the scaled features
    best_model = run_logistic_regression(X_train, y_train, X_test, y_test)

    # Plot coefficients for each class
    plot_coefficients(best_model, X_train.columns, class_index=0)
    plot_coefficients(best_model, X_train.columns, class_index=1)
    plot_coefficients(best_model, X_train.columns, class_index=2)


if __name__ == "__main__":
    main()
