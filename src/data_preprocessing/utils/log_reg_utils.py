import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
import multiprocessing


def run_logistic_regression_gridsearch(X_train, y_train, X_test, y_test, random_state=42):
    # Define pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('logreg', LogisticRegression(
            random_state=random_state
        ))
    ])

    # Grid of C values to test
    param_grid = {
        'logreg__penalty': ['l1'], # l2
        'logreg__C': [100],
        'logreg__class_weight': ['balanced'], # None
        'logreg__solver': ['saga'],
        'logreg__max_iter': [5000], #10000
        #'logreg__tol': [1e-3, 1e-4, 1e-5]  # Optional
    }

    # Or RandomSearchCV
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=5,
        verbose=1,
        n_jobs=7
    )

    # Fit
    grid.fit(X_train, y_train)

    # Best prediction_models
    best_model = grid.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return best_model

def run_logistic_regression(X_train, y_train, X_test, y_test, random_state=42):
    # Use all available cores minus one
    n_jobs = max(multiprocessing.cpu_count() - 1, 1)

    # Define pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('logreg', LogisticRegression(
            penalty='l1',
            C=0.5,
            class_weight='balanced',
            solver='saga',
            max_iter=5000,
            n_jobs=n_jobs,
            random_state=random_state
        ))
    ])

    # Fit pipeline
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def plot_coefficients(best_model, feature_names, class_index):
    # Access prediction_models inside pipeline
    logistic_model = best_model.named_steps['logreg']

    coefs = logistic_model.coef_[class_index]

    if len(coefs) != len(feature_names):
        raise ValueError("Mismatch in feature names and coefficient count")

    coef_series = pd.Series(coefs, index=feature_names).sort_values()
    plt.figure(figsize=(10, 6))
    coef_series.plot(kind='barh')
    plt.title(f'Important Coefficients for Class {class_index}')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()
