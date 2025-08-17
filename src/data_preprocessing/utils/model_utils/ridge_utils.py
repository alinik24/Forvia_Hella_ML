import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RidgeClassifierCV, ridge_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def run_ridge(X, y, alphas=np.logspace(0, 10, 100), cv=3, scoring='neg_mean_squared_error'):
    ridge = RidgeClassifierCV(alphas=alphas, scoring=scoring, cv=cv, class_weight='balanced').fit(X, y)
    return ridge


def plot_feature_importance(ridge_model, feature_names, top_n=20):
    """Rank features by absolute coefficient magnitude"""
    # Handle multiclass: average across classes
    if ridge_model.coef_.ndim > 1:
        coefs = ridge_model.coef_.mean(axis=0)
    else:
        coefs = ridge_model.coef_

    coefs = pd.Series(coefs, index=feature_names)
    ranked = coefs.abs().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ranked.values, y=ranked.index, palette='coolwarm')
    plt.title(f'Top {top_n} Features by Absolute Coefficient Value (Ridge)')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_coefficients(ridge_model, feature_names):
    if ridge_model.coef_.ndim > 1:
        coefs = ridge_model.coef_.mean(axis=0)
    else:
        coefs = ridge_model.coef_

    coefs = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=True)

    wrapped_labels = ['\n'.join(textwrap.wrap(label, 30)) for label in coefs.index]
    coefs.index = wrapped_labels

    plt.figure(figsize=(12, len(coefs) * 0.25))
    coefs.plot(kind='barh', color='navy')
    plt.title('Ridge Coefficients (All)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()


def plot_nonzero_coef_distribution(ridge_model, threshold=1e-6):
    """Histogram of coefficients above a small threshold"""
    coefs = ridge_model.coef_
    if coefs.ndim == 1:  # binary case
        coefs = coefs[np.newaxis, :]

    for i, class_name in enumerate(ridge_model.classes_):
        significant = coefs[i][np.abs(coefs[i]) > threshold]
        plt.figure(figsize=(8, 5))
        plt.hist(significant, bins=30, color='lightgreen', edgecolor='black')
        plt.title(f'Distribution of Ridge Coefficients for class {class_name} (|coef| > threshold)')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def plot_all_coefs_sorted(ridge_model, feature_names):
    """Plot all coefficients sorted by absolute value"""
    if ridge_model.coef_.ndim > 1:
        coefs = ridge_model.coef_.mean(axis=0)
    else:
        coefs = ridge_model.coef_

    coefs = pd.Series(coefs, index=feature_names)
    sorted_coefs = coefs.sort_values(key=abs)

    plt.figure(figsize=(12, len(coefs) * 0.25))
    sorted_coefs.plot(kind='barh', color='slateblue')
    plt.title('All Ridge Coefficients Sorted by Magnitude')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_ridge_path(X, y, alphas=np.logspace(-4, 4, 100)):
    """Plot Ridge coefficient paths as alpha varies"""
    X_scaled = StandardScaler().fit_transform(X)
    coefs = []

    for alpha in alphas:
        y_numeric = pd.Categorical(y).codes
        coef = ridge_regression(X_scaled, y_numeric, alpha=alpha)
        coefs.append(coef)

    coefs = np.array(coefs)

    plt.figure(figsize=(12, 8))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f'Feature {i}', lw=1)

    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficients')
    plt.title('Ridge Coefficient Paths')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
