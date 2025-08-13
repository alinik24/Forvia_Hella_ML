# lasso_utils.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.metrics import mean_squared_error
import textwrap


def run_lasso(X, y, alphas=100, n_jobs=4, cv=5, max_iter=10000, random_state=42):
    lasso = LassoCV(cv=cv, random_state=random_state, alphas=alphas, n_jobs=n_jobs, max_iter=max_iter, tol=0.0001).fit(X, y)
    return lasso


def plot_cv_mse(lasso_cv):
    """Plot mean squared error across alphas from LassoCV"""
    mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
    alphas = lasso_cv.alphas_

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, mse_mean, marker='o')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Cross-Validation MSE vs Alpha')
    plt.grid(True)
    plt.show()


def plot_feature_importance(lasso_model, feature_names, top_n=20):
    """Rank features by absolute coefficient magnitude"""
    coefs = pd.Series(lasso_model.coef_, index=feature_names)
    ranked = coefs.abs().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ranked.values, y=ranked.index, palette='coolwarm')
    plt.title(f'Top {top_n} Features by Absolute Coefficient Value')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_lasso_path(X, y):
    """Plot coefficient path as alpha varies"""
    alphas, coefs, _ = lasso_path(X, y)

    plt.figure(figsize=(12, 8))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[i], lw=1)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficients')
    plt.title('Lasso Coefficient Paths')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_coefficients(lasso_model, feature_names):
    coefs = pd.Series(lasso_model.coef_, index=feature_names).sort_values(key=abs, ascending=True)

    # Wrap long feature names
    wrapped_labels = ['\n'.join(textwrap.wrap(label, 30)) for label in coefs.index]
    coefs.index = wrapped_labels

    # Plot
    plt.figure(figsize=(12, len(coefs) * 0.25))  # Adjust height based on number of features
    coefs.plot(kind='barh', color=coefs.apply(lambda x: 'red' if x != 0 else 'gray'))
    plt.title('Lasso Coefficients (Zero and Non-zero)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()


def plot_nonzero_coef_distribution(lasso_model):
    """Histogram of non-zero coefficient values"""
    coefs = lasso_model.coef_
    non_zero_coefs = coefs[coefs != 0]

    plt.figure(figsize=(8, 5))
    plt.hist(non_zero_coefs, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Non-zero Lasso Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_all_nonzero_coefs_sorted(lasso_model, feature_names):
    """Plot all non-zero coefficients sorted by absolute value"""
    coefs = pd.Series(lasso_model.coef_, index=feature_names)
    non_zero = coefs[coefs != 0].sort_values(key=abs)

    plt.figure(figsize=(12, len(coefs) * 0.25))
    non_zero.plot(kind='barh', color='teal')
    plt.title('All Non-zero Lasso Coefficients Sorted by Magnitude')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
