# preprocessing_utils.py
import gc
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from skrub import MinHashEncoder, DatetimeEncoder, TableVectorizer, MinHashEncoder


def load_parquet_dataset(file_path):
    """Load parquet file"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def build_diverse_sample(df: pd.DataFrame, max_per_column=1000):
    """Memory-efficient diverse sample with categorical handling"""
    sample_dict = {}
    total_cols = len(df.columns)

    for idx, col in enumerate(df.columns, 1):
        print(f"[{idx}/{total_cols}] Processing {col}")
        col_data = df[col]

        # Handle categoricals differently
        if pd.api.types.is_categorical_dtype(col_data):
            uniques = col_data.cat.categories
        else:
            uniques = col_data.dropna().unique()

        uniques = uniques[:max_per_column]
        pad_count = max_per_column - len(uniques)
        padded = np.concatenate([uniques, [np.nan] * pad_count])
        sample_dict[col] = padded

    return pd.DataFrame(sample_dict)


def filter_serial_numbers(df, max_unique_snids):
    """Safely filter serial numbers with validation"""
    if max_unique_snids is None or 'serial_number_id' not in df.columns:
        return df

    try:
        unique_snids = df['serial_number_id'].dropna().unique()
        if len(unique_snids) < max_unique_snids:
            print(f"Warning: Only {len(unique_snids)} unique serial numbers available, using all")
            return df

        selected_snids = np.random.choice(unique_snids, max_unique_snids, replace=False)
        df = df[df['serial_number_id'].isin(selected_snids)]
        print(f"Using {max_unique_snids} unique serial_number_id values.")
        return df
    except Exception as e:
        print(f"Error filtering serial numbers: {e}")
        return df


def convert_columns(X, datetime_cols=None, boolean_cols=None):
    """Safe column type conversion with error handling"""
    datetime_cols = datetime_cols or []
    boolean_cols = boolean_cols or []

    for col in datetime_cols:
        if col in X.columns:
            try:
                # Simplified datetime conversion
                X[col] = pd.to_datetime(X[col], errors='coerce', utc=True)
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
                X = X.drop(columns=[col])

    for col in boolean_cols:
        if col in X.columns:
            try:
                print(f"Converting {col} from dtype {X[col].dtype}")
                # Handle mixed types and missing values
                X[col] = X[col].astype(str).str.lower().map({
                    'true': 1, '1': 1, 'yes': 1, 'y': 1,
                    'false': 0, '0': 0, 'no': 0, 'n': 0
                }).fillna(0).astype(np.int8)
                print(f"Converted {col} to int8")
            except Exception as e:
                print(f"Error converting {col} to boolean: {e}")
                X = X.drop(columns=[col])

    return X


def fit_vectorizer(df, path='vectorizer.pkl', max_sample_size=10000, cardinality_threshold=100):
    """Robust vectorizer fitting with memory management"""
    try:
        if os.path.exists(path):
            print("Loading existing vectorizer")
            return joblib.load(path)

        print("Fitting new vectorizer...")

        # Sample for fitting if dataset is large
        if len(df) > max_sample_size:
            print(f"Sampling {max_sample_size} rows for vectorizer fitting")
            df = df.sample(max_sample_size, random_state=42)

        # Configure vectorizer
        vectorizer = TableVectorizer(
            cardinality_threshold=cardinality_threshold,
            low_cardinality=make_pipeline(
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype='float32'
                )
            ),
            high_cardinality=MinHashEncoder(n_components=30),
            drop_null_fraction=1.0,  # Drop columns that are entirely null
            n_jobs=os.cpu_count() - 1,
        )

        vectorizer.fit(df)
        joblib.dump(vectorizer, path)
        print("Vectorizer successfully fitted and saved")
        return vectorizer
    except MemoryError:
        print("Memory error fitting vectorizer. Reducing sample size.")
        return fit_vectorizer(df, path, max_sample_size // 2, cardinality_threshold)
    except Exception as e:
        print(f"Error fitting vectorizer: {e}")
        # Fallback to simple configuration
        try:
            print("Attempting fallback configuration")
            vectorizer = TableVectorizer(
                cardinality_threshold=cardinality_threshold,
                n_jobs=os.cpu_count() - 1,
            )
            vectorizer.fit(df)
            joblib.dump(vectorizer, path)
            print("Fallback vectorizer successfully fitted and saved")
            return vectorizer
        except Exception as fallback_e:
            print(f"Fallback vectorizer failed: {fallback_e}")
            raise


def encode_features(X, vectorizer):
    """Safe feature encoding with memory management"""
    try:
        print("Transforming features...")
        X_transformed = vectorizer.transform(X)
        feature_names = vectorizer.get_feature_names_out()
        print(f"Generated {len(feature_names)} features")
        return X_transformed, feature_names
    except MemoryError:
        print("Memory error during encoding. Freeing resources and retrying.")
        gc.collect()
        return encode_features(X, vectorizer)  # Retry after GC
    except Exception as e:
        print(f"Error encoding features: {e}")
        raise


def scale_features(matrix, feature_names):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return pd.DataFrame(scaled, columns=feature_names)


def preprocess_data(
        df: pd.DataFrame,
        target_column: str,
        datetime_cols=None,
        boolean_cols=None,
        vectorizer_path='vectorizer.pkl',
        max_unique_snids: int = None
):
    """End-to-end preprocessing pipeline"""
    try:
        # Phase 1: Data preparation
        print("Starting preprocessing...")
        df = filter_serial_numbers(df, max_unique_snids)
        print(f"Data shape after filtering: {df.shape}")

        # Split target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        X = df.drop(columns=[target_column])
        y = df[target_column].reset_index(drop=True)
        print(f"Target distribution:\n{y.describe()}")

        # Convert column types
        X = convert_columns(X, datetime_cols, boolean_cols)

        # Phase 2: Vectorization
        vectorizer = fit_vectorizer(X, vectorizer_path)
        X_encoded, feature_names = encode_features(X, vectorizer)

        # Clean up memory
        del X
        gc.collect()

        # Phase 3: Scaling
        print("Scaling features...")
        X_processed = pd.DataFrame(X_encoded, columns=feature_names)
        X_scaled = scale_features(X_processed, feature_names)

        # Phase 3: Final processing
        print("Preprocessing complete!")
        print(f"Final feature matrix shape: {X_scaled.shape}")
        return y, feature_names, X_scaled

    except MemoryError as e:
        print(f"Memory exhausted: {e}. Try reducing max_unique_snids.")
        raise
    except Exception as e:
        print(f"Critical error in preprocessing: {e}")
        raise
