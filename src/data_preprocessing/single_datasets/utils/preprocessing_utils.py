# preprocessing_utils.py
import os
from functools import reduce

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer


def load_parquet_dataset(file_path):
    return pd.read_parquet(file_path)


def find_high_cardinality_columns(df, threshold=50):
    """Return list of columns with unique values count above threshold (for categorical columns only)."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cardinalities = df[cat_cols].nunique(dropna=True)
    high_card_cols = cardinalities[cardinalities > threshold].index.tolist()
    return high_card_cols


def build_diverse_sample(df: pd.DataFrame, max_per_column=1000):
    """
    Create a diverse sample for fitting the vectorizer by collecting unique values per column.
    Args:
        df (pd.DataFrame): Full dataset.
        max_per_column (int): Max unique values to collect per column.
    Returns:
        pd.DataFrame: Synthetic sample with diverse values across all columns.
    """
    sample_frames = []
    total_cols = len(df.columns)

    print(f"Building diverse sample for {total_cols} columns...")

    for idx, col in enumerate(df.columns, start=1):
        print(f"[{idx}/{total_cols}] Processing column: {col}")

        unique_values = df[col].dropna().unique()[:max_per_column]
        temp_df = pd.DataFrame({col: unique_values})
        sample_frames.append(temp_df)

    # Use outer join to align all samples into one frame
    print("Merging all column samples into a single DataFrame...")
    diverse_sample = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
                            sample_frames)

    print("Diverse sample created.")
    return diverse_sample


def frequency_encode(df, columns):
    """
    Frequency encode high-cardinality categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): Columns to frequency encode.

    Returns:
        pd.DataFrame: DataFrame with frequency encoded columns (same column names).
    """
    df = df.copy()
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
    return df


def preprocess_data(
        df: pd.DataFrame,
        target_column: str,
        datetime_cols=None,
        boolean_cols=None,
        high_card_threshold=50,
        vectorizer_path='vectorizer.pkl',
):
    """
    Preprocess a mixed-type dataframe using skrub's TableVectorizer:
    - Handles datetime, categorical, text, and string columns automatically.
    - Scales the resulting numeric features.
    - Excludes high-cardinality categorical columns from fitting vectorizer sample.

    Returns:
        y (pd.Series): Target column.
        feature_names (List[str]): Names of encoded features.
        X_encoded (pd.DataFrame): Encoded feature dataframe (unscaled).
    """
    print("[1/7] Splitting target and features...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("[2/7] Parsing datetime columns...")
    datetime_cols = datetime_cols or []
    boolean_cols = boolean_cols or []

    for col in datetime_cols:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')

    for col in boolean_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)

    print("[3/7] Detecting high-cardinality columns...")
    high_card_cols = find_high_cardinality_columns(X, threshold=high_card_threshold)
    print(f"High cardinality columns (excluded from vectorizer fitting): {high_card_cols}")

    sample_fit_df = X.drop(columns=high_card_cols) if high_card_cols else X

    print(f"[4/7] Building diverse sample for vectorizer fitting on {sample_fit_df.shape[1]} columns...")
    sample_df = build_diverse_sample(sample_fit_df)

    print("[5/7] Fitting or loading TableVectorizer...")
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TableVectorizer()
        vectorizer.fit(sample_df)
        joblib.dump(vectorizer, vectorizer_path)

    print("[6/7] Transforming full dataset (excluding high-cardinality columns)...")
    X_to_transform = X.drop(columns=high_card_cols) if high_card_cols else X
    X_encoded_part = vectorizer.transform(X_to_transform)

    if high_card_cols:
        print(f"Adding back high-cardinality columns with frequency encoding: {high_card_cols}")
        X_high_card = X[high_card_cols].reset_index(drop=True).copy()
        for col in high_card_cols:
            freq = X_high_card[col].value_counts(normalize=True)
            X_high_card[col] = X_high_card[col].map(freq)

        X_encoded = pd.concat([
            pd.DataFrame(X_encoded_part, columns=vectorizer.get_feature_names_out()),
            X_high_card.reset_index(drop=True)
        ], axis=1)
    else:
        X_encoded = pd.DataFrame(X_encoded_part, columns=vectorizer.get_feature_names_out())

    print("[7/7] Scaling encoded features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    feature_names = X_encoded.columns.tolist()
    print("Preprocessing complete!")

    return y, feature_names, pd.DataFrame(X_scaled, columns=feature_names)
