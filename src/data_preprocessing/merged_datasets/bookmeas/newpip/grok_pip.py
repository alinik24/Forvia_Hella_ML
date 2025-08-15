"""
Robust End-to-End Machine Learning Pipeline for Multi-Class Classification (Revised)

Revisions based on gap analysis:
- Split raw data first to prevent leakage.
- Made custom classes (DataPreprocessor, ImbalanceHandler) compatible with sklearn Pipeline by implementing fit/transform/fit_resample.
- Fit all preprocessing (imputation, encoding, scaling, selection) on training data only.
- Wrapped preprocessing, balancing, and modeling in sklearn Pipeline for each algorithm to ensure no leakage.
- Fixed Target Encoding to use training y only.
- Added class weights universally for models that support it, especially when resampling is skipped.
- Added redundancy removal post-engineering using correlation threshold.
- Ensured consistent transform logic for test data.
- Saved train/test processed separately.
- Added permutation importance for model-agnostic feature analysis.
- Minor enhancements: Added early stopping for boosting models where applicable; per-class F1 plots.

This addresses leakage, overfitting, and imbalance gaps for robust, generalizable results.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import json
import pickle
import gc
import os
from collections import Counter
from typing import List, Dict, Any

# Core ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             balanced_accuracy_score, f1_score, precision_recall_fscore_support)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline  # For handling resampling in pipeline

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available() else 'default')

warnings.filterwarnings('ignore')

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Handles feature creation/engineering. Pipeline-compatible."""
    
    def __init__(self, datetime_cols: List[str] = None, corr_threshold: float = 0.95, verbose: bool = True):
        self.datetime_cols = datetime_cols or []
        self.corr_threshold = corr_threshold
        self.verbose = verbose
        self.high_corr_cols = []
        
    def fit(self, X, y=None):
        """Fit: Analyze for redundant features."""
        X_eng = self._engineer_features(X)
        # Correlation analysis on numerics
        num_cols = X_eng.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            corr_matrix = X_eng[num_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.high_corr_cols = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        return self
    
    def transform(self, X):
        """Transform: Engineer and remove redundants."""
        X_eng = self._engineer_features(X)
        X_eng = X_eng.drop(columns=self.high_corr_cols)
        if self.verbose:
            print(f"✅ Engineered features; removed {len(self.high_corr_cols)} redundant")
        return X_eng
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_eng = df.copy()
        for col in self.datetime_cols:
            if col in df_eng.columns:
                dt_col = pd.to_datetime(df_eng[col], errors='coerce')
                df_eng[f'{col}_year'] = dt_col.dt.year
                df_eng[f'{col}_month'] = dt_col.dt.month
                df_eng[f'{col}_day'] = dt_col.dt.day
                df_eng[f'{col}_hour'] = dt_col.dt.hour
                df_eng = df_eng.drop(columns=[col])
        return df_eng

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Modular preprocessor for mixed-type datasets. Pipeline-compatible."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.enc_transformer = None
        self.scaler = None
        self.selector = None
        self.feature_info: Dict[str, List[str]] = {}
        self.columns_out = None
        
    def fit(self, X, y=None):
        """Fit all components on training data."""
        self._analyze_features(X)
        
        # Drop problematic in transform, but note them
        drop_cols = self.feature_info['constant'] + self.feature_info['high_missing']
        
        # Imputation (store medians for numerics)
        self.num_medians = X[self.feature_info['numeric']].median()
        
        # Encoding transformer
        cat_low = [col for col in self.feature_info['categorical_low_card'] if col not in drop_cols]
        cat_high = [col for col in self.feature_info['categorical_high_card'] if col not in drop_cols]
        num_cols = [col for col in self.feature_info['numeric'] if col not in drop_cols]
        
        transformers = [
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_low),
        ]
        if cat_high and y is not None:
            transformers.append(('target', TargetEncoder(), cat_high))
        transformers.append(('num', 'passthrough', num_cols))
        
        self.enc_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop others
        )
        X_encoded = self._impute(X).drop(columns=drop_cols)
        self.enc_transformer.fit(X_encoded, y)
        
        X_trans = self.enc_transformer.transform(X_encoded)
        encoded_cols = self.enc_transformer.get_feature_names_out()
        X_trans_df = pd.DataFrame(X_trans, columns=encoded_cols, index=X.index)
        
        # Scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_trans_df)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=encoded_cols, index=X.index)
        
        # Selector
        lr = LogisticRegression(max_iter=100, class_weight='balanced', n_jobs=-1)
        n_select = min(50, X_scaled_df.shape[1])
        self.selector = RFE(lr, n_features_to_select=n_select)
        self.selector.fit(X_scaled_df, y)
        
        self.columns_out = X_scaled_df.columns[self.selector.support_]
        
        if self.verbose:
            print(f"✅ Fitted preprocessor: {len(self.columns_out)} features selected")
            
        return self
    
    def transform(self, X):
        """Transform using fitted components."""
        drop_cols = self.feature_info['constant'] + self.feature_info['high_missing']
        X_imp = self._impute(X).drop(columns=drop_cols, errors='ignore')
        
        X_trans = self.enc_transformer.transform(X_imp)
        X_trans_df = pd.DataFrame(X_trans, columns=self.enc_transformer.get_feature_names_out(), index=X.index)
        
        X_scaled = self.scaler.transform(X_trans_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_trans_df.columns, index=X.index)
        
        X_selected = X_scaled_df[self.columns_out]
        
        return X_selected
    
    def _analyze_features(self, X: pd.DataFrame):
        """Analyze feature types (called in fit)."""
        analysis = {
            'numeric': [],
            'categorical_low_card': [],  # <10 unique
            'categorical_high_card': [],  # >=10
            'constant': [],
            'high_missing': [],
            'datetime': []  # Assumed handled upstream
        }
        
        for col in X.columns:
            if X[col].nunique() <= 1:
                analysis['constant'].append(col)
                continue
                
            missing_pct = X[col].isnull().mean()
            if missing_pct > 0.8:
                analysis['high_missing'].append(col)
                continue
                
            if pd.api.types.is_numeric_dtype(X[col]):
                analysis['numeric'].append(col)
            else:
                card = X[col].nunique()
                if card < 10:
                    analysis['categorical_low_card'].append(col)
                else:
                    analysis['categorical_high_card'].append(col)
                    
        self.feature_info = analysis
        if self.verbose:
            for k, v in analysis.items():
                print(f"   {k.capitalize()}: {len(v)}")
    
    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = X.copy()
        num_cols = self.feature_info['numeric']
        if num_cols:
            X_imp[num_cols] = X_imp[num_cols].fillna(self.num_medians)
        
        cat_cols = self.feature_info['categorical_low_card'] + self.feature_info['categorical_high_card']
        for col in cat_cols:
            X_imp[col] = X_imp[col].fillna('MISSING').astype(str)
            
        return X_imp

class ImbalanceHandler(BaseEstimator, TransformerMixin):
    """Handles class imbalance adaptively. Supports fit_resample for ImbPipeline."""
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.strategy = None
        self.sampler = None
        
    def fit(self, X, y=None):
        if y is None:
            return self
        self.strategy, _ = self._analyze(y)
        if self.strategy == 'conservative_smote':
            self.sampler = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=self.random_state)
        elif self.strategy == 'smote_enn':
            self.sampler = SMOTEENN(sampling_strategy=0.3, random_state=self.random_state)
        else:
            self.sampler = None  # No resampling
        if self.sampler:
            self.sampler.fit_resample(X, y)  # Fit sampler
        return self
    
    def fit_resample(self, X, y):
        """For ImbPipeline compatibility."""
        self.fit(X, y)
        if self.sampler:
            X_res, y_res = self.sampler.fit_resample(X, y)
            if self.verbose:
                print(f"🔄 Balanced with {self.strategy}: {Counter(y_res)}")
            return X_res, y_res
        return X, y
    
    def transform(self, X):
        return X  # No transform needed for test
    
    def _analyze(self, y: pd.Series) -> tuple:
        counts = Counter(y)
        ratio = max(counts.values()) / min(counts.values()) if min(counts.values()) > 0 else float('inf')
        if ratio > 10000:
            strategy = 'ensemble_only'
        elif ratio > 1000:
            strategy = 'conservative_smote'
        elif ratio > 100:
            strategy = 'smote_enn'
        else:
            strategy = 'class_weights'
            
        if self.verbose:
            print(f"⚖️ Imbalance ratio: {ratio:.1f}:1 → Strategy: {strategy}")
            
        return strategy, ratio

class ModelEvaluator:
    """Evaluates multiple algorithms with hyperparam tuning."""
    
    def __init__(self, output_dir: str, random_state: int = 42, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_score = 0
        
    def get_algorithms(self) -> Dict[str, tuple]:
        """Algorithms with param grids (from literature). Added early stopping where possible."""
        algos = {
            'XGBoost': (
                xgb.XGBClassifier(objective='multi:softprob', random_state=self.random_state, n_jobs=-1, enable_categorical=True, early_stopping_rounds=10),
                {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
            ),
            'LightGBM': (
                lgb.LGBMClassifier(objective='multiclass', random_state=self.random_state, n_jobs=-1),
                {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'num_leaves': [15, 31]}
            ),
            'CatBoost': (
                CatBoostClassifier(task_type='CPU', random_state=self.random_state, verbose=0, early_stopping_rounds=10),
                {'iterations': [50, 100], 'depth': [3, 6], 'learning_rate': [0.01, 0.1]}
            ),
            'RandomForest': (
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            ),
            'GradientBoosting': (
                GradientBoostingClassifier(random_state=self.random_state),
                {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            ),
            'MLP': (
                MLPClassifier(random_state=self.random_state, max_iter=200, early_stopping=True),
                {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
            ),
            'SVM': (
                SVC(probability=True, random_state=self.random_state),
                {'C': [0.1, 1], 'kernel': ['rbf', 'linear']}
            ),
            'LogisticRegression': (
                LogisticRegression(max_iter=500, random_state=self.random_state, n_jobs=-1),
                {'C': [0.1, 1], 'solver': ['liblinear', 'lbfgs']}
            )
        }
        return algos
    
    def train_evaluate(self, X_train_raw: pd.DataFrame, y_train: pd.Series, X_test_raw: pd.DataFrame, y_test: pd.Series, engineer: FeatureEngineer, preprocessor: DataPreprocessor, imb_handler: ImbalanceHandler):
        """Train and evaluate all algorithms using full pipeline."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, (model, param_grid) in self.get_algorithms().items():
            if hasattr(model, 'class_weight'):
                model.class_weight = 'balanced'
                
            full_pipe = ImbPipeline([
                ('engineer', engineer),
                ('preprocess', preprocessor),
                ('balancer', imb_handler),
                ('model', model)
            ])
            
            grid = GridSearchCV(full_pipe, {f'model__{k}': v for k, v in param_grid.items()}, 
                                cv=cv, scoring='balanced_accuracy', n_jobs=-1, refit=True)
            
            grid.fit(X_train_raw, y_train)
            
            y_pred = grid.predict(X_test_raw)
            
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            
            self.results[name] = {
                'best_params': grid.best_params_,
                'balanced_acc': bal_acc,
                'macro_f1': macro_f1,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if bal_acc > self.best_score:
                self.best_score = bal_acc
                self.best_model = grid.best_estimator_
                
            if self.verbose:
                print(f"✅ {name}: Bal Acc={bal_acc:.4f}, Macro F1={macro_f1:.4f}")
                
        # Recommend best
        best_algo = max(self.results, key=lambda k: self.results[k]['balanced_acc'])
        print(f"🏆 Best: {best_algo} with Bal Acc={self.results[best_algo]['balanced_acc']:.4f}")
        
    def generate_plots(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, class_names: List[str]):
        """Generate plots using test data."""
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Class distribution (test)
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_test)
        plt.title('Test Class Distribution')
        plt.savefig(self.output_dir / 'class_dist.png')
        plt.close()
        
        # Per-class F1
        _, _, f1, support = precision_recall_fscore_support(y_test, y_pred)
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, f1)
        plt.title('Per-Class F1 Scores')
        plt.savefig(self.output_dir / 'per_class_f1.png')
        plt.close()
        
        # Feature importance (model-specific + permutation)
        if hasattr(self.best_model.named_steps['model'], 'feature_importances_'):
            imp = pd.Series(self.best_model.named_steps['model'].feature_importances_, 
                            index=self.best_model.named_steps['preprocess'].columns_out)
            imp.sort_values(ascending=False)[:20].plot(kind='barh')
            plt.title('Top 20 Model Feature Importances')
            plt.savefig(self.output_dir / 'feature_importance_model.png')
            plt.close()
        
        # Permutation importance
        perm_imp = permutation_importance(self.best_model, X_test, y_test, n_repeats=5, random_state=self.random_state, n_jobs=-1)
        perm_df = pd.DataFrame({'importance': perm_imp.importances_mean}, index=X_test.columns)
        perm_df.sort_values('importance', ascending=False)[:20].plot(kind='barh')
        plt.title('Top 20 Permutation Importances')
        plt.savefig(self.output_dir / 'feature_importance_perm.png')
        plt.close()
        
    def save_outputs(self, X_train_proc: pd.DataFrame, y_train: pd.Series, X_test_proc: pd.DataFrame, y_test: pd.Series, timestamp: str):
        """Save processed train/test separately."""
        train_df = X_train_proc.copy()
        train_df['target'] = y_train.values
        train_df.to_parquet(self.output_dir / f'processed_train_{timestamp}.parquet')
        
        test_df = X_test_proc.copy()
        test_df['target'] = y_test.values
        test_df.to_parquet(self.output_dir / f'processed_test_{timestamp}.parquet')
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, default=str)
            
        with open(self.output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

def main_pipeline(data_path: str, target_col: str = 'target'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./output/ml_pipeline_{timestamp}'
    
    # Load data
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    print(f"✅ Loaded: {df.shape}")
    
    # Split raw data
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
    print(f"✂️ Split: Train {X_train_raw.shape}, Test {X_test_raw.shape}")
    
    # Initialize components
    engineer = FeatureEngineer(datetime_cols=['created_at', 'updated_at'])  # Example cols
    preprocessor = DataPreprocessor()
    imb_handler = ImbalanceHandler()
    
    # Evaluate models
    evaluator = ModelEvaluator(output_dir)
    evaluator.train_evaluate(X_train_raw, y_train, X_test_raw, y_test, engineer, preprocessor, imb_handler)
    
    # Get processed data for saves/plots (using fitted pipeline)
    X_train_proc = evaluator.best_model.named_steps['preprocess'].transform(
        evaluator.best_model.named_steps['engineer'].transform(X_train_raw)
    )
    X_test_proc = evaluator.best_model.named_steps['preprocess'].transform(
        evaluator.best_model.named_steps['engineer'].transform(X_test_raw)
    )
    
    # Plots and saves
    y_pred = evaluator.best_model.predict(X_test_raw)
    class_names = sorted(y.unique().astype(str))
    evaluator.generate_plots(X_test_proc, y_test, y_pred, class_names)
    evaluator.save_outputs(X_train_proc, y_train, X_test_proc, y_test, timestamp)
    
    print(f"✅ Pipeline complete! Outputs in {output_dir}")

if __name__ == '__main__':
    data_path = input("Enter dataset path (.parquet/.csv): ").strip()
    target_col = input("Enter target column: ").strip()
    main_pipeline(data_path, target_col)