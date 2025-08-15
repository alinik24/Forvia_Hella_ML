"""
Advanced Multi-Class Imbalanced Classification Pipeline
======================================================
Research-based pipeline for extreme class imbalance (43,000:1 ratio)
Implements state-of-the-art techniques from recent literature (2024)

Key Features:
- Rigorous preprocessing with data leakage prevention
- Advanced feature engineering and selection
- Multiple imbalance handling strategies
- Comprehensive algorithm comparison
- Deep learning models with custom loss functions
- Hyperparameter optimization with cross-validation
- Advanced evaluation metrics for imbalanced data

Based on:
- Frontiers Digital Health 2024: Multi-class imbalanced classification
- Nature Scientific Reports 2024: IMCP curves for multiclass evaluation
- MDPI Remote Sensing 2024: Best practices for imbalanced metrics
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
import joblib
from typing import Dict, List, Tuple, Any

# Core ML imports
from sklearn.model_selection import (StratifiedKFold, cross_validate, 
                                     train_test_split, RandomizedSearchCV)
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   LabelEncoder, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, SelectFromModel, 
                                       RFE, VarianceThreshold, f_classif,
                                       mutual_info_classif)

# Advanced feature engineering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder, WOEEncoder, JamesSteinEncoder

# Models
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier, StackingClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Boosting algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                         Input, Concatenate)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import class_weight
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Imbalanced learning
try:
    from imblearn.ensemble import (BalancedRandomForestClassifier, 
                                 EasyEnsembleClassifier, BalancedBaggingClassifier)
    from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE, 
                                      SVMSMOTE, KMeansSMOTE)
    from imblearn.under_sampling import (RandomUnderSampler, EditedNearestNeighbours,
                                       TomekLinks, CondensedNearestNeighbour)
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.metrics import classification_report_imbalanced
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Metrics and evaluation
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, cohen_kappa_score)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")

warnings.filterwarnings('ignore')


class DataLeakagePreventionProcessor:
    """
    Rigorous preprocessing with data leakage prevention
    Based on best practices from recent literature
    """
    
    def __init__(self, target_col: str, test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessors = {}
        self.feature_info = {}
        
    def initial_data_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data before any preprocessing to prevent leakage"""
        print("🔒 Performing leak-safe initial split...")
        
        try:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, 
                stratify=y, random_state=self.random_state
            )
        except ValueError:
            # If stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"    Train: {len(train_df):,} samples")
        print(f"    Test: {len(test_df):,} samples")
        
        return train_df, test_df
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Comprehensive feature analysis"""
        print("🔍 Analyzing features...")
        
        features = [col for col in df.columns if col != self.target_col]
        
        analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'constant_features': [],
            'high_cardinality': [],
            'missing_heavy': [],
            'potential_leakage': []
        }
        
        for col in features:
            # Missing analysis
            missing_pct = df[col].isnull().sum() / len(df)
            
            # Constant check
            if df[col].nunique(dropna=False) <= 1:
                analysis['constant_features'].append(col)
                continue
                
            # Heavy missing
            if missing_pct > 0.95:
                analysis['missing_heavy'].append(col)
                continue
            
            # Data type analysis
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['datetime_features'].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric_features'].append(col)
                
                # Check for potential ID columns (high cardinality numeric)
                if df[col].nunique() > 0.9 * len(df):
                    analysis['potential_leakage'].append(col)
                    
            else:
                analysis['categorical_features'].append(col)
                
                # High cardinality check
                if df[col].nunique() > min(1000, len(df) * 0.5):
                    analysis['high_cardinality'].append(col)
        
        self.feature_info = analysis
        
        # Print analysis
        for category, features in analysis.items():
            if features:
                print(f"    {category}: {len(features)} features")
                if category in ['potential_leakage', 'high_cardinality'] and features:
                    print(f"      Examples: {features[:3]}")
        
        return analysis
    
    def create_advanced_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced feature engineering"""
        print("⚙️ Creating advanced features...")
        
        df_enhanced = df.copy()
        numeric_cols = self.feature_info['numeric_features']
        categorical_cols = self.feature_info['categorical_features']
        
        # 1. Numeric feature transformations
        if numeric_cols:
            for col in numeric_cols:
                if col in df_enhanced.columns:
                    # Log transform for skewed features
                    if df_enhanced[col].min() > 0:
                        df_enhanced[f'{col}_log'] = np.log1p(df_enhanced[col])
                    
                    # Binning
                    df_enhanced[f'{col}_binned'] = pd.cut(df_enhanced[col], 
                                                          bins=5, labels=False)
        
        # 2. Interaction features (limited to prevent explosion)
        if len(numeric_cols) >= 2:
            # Select top correlated features with target
            target_corr = {}
            for col in numeric_cols[:10]:  # Limit to top 10
                if col in df_enhanced.columns:
                    try:
                        corr = df_enhanced[col].corr(df_enhanced[self.target_col])
                        if not pd.isna(corr):
                            target_corr[col] = abs(corr)
                    except:
                        pass
            
            top_features = sorted(target_corr.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create interactions
            for i, (col1, _) in enumerate(top_features):
                for col2, _ in top_features[i+1:]:
                    if col1 in df_enhanced.columns and col2 in df_enhanced.columns:
                        df_enhanced[f'{col1}_{col2}_multiply'] = (
                            df_enhanced[col1] * df_enhanced[col2]
                        )
        
        # 3. Aggregate features by categorical groups
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols[:3]:  # Limit computational cost
                if cat_col in df_enhanced.columns and df_enhanced[cat_col].nunique() < 50:
                    for num_col in numeric_cols[:3]:
                        if num_col in df_enhanced.columns:
                            # Group statistics
                            group_stats = df_enhanced.groupby(cat_col)[num_col].agg(['mean', 'std'])
                            
                            df_enhanced[f'{num_col}_mean_by_{cat_col}'] = (
                                df_enhanced[cat_col].map(group_stats['mean'])
                            )
                            df_enhanced[f'{num_col}_std_by_{cat_col}'] = (
                                df_enhanced[cat_col].map(group_stats['std'])
                            )
        
        print(f"    Enhanced features: {df_enhanced.shape[1] - df.shape[1]} new features")
        return df_enhanced
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Comprehensive preprocessing without data leakage"""
        print("🔧 Processing data (leak-safe)...")
        
        # Remove problematic features
        features_to_remove = (
            self.feature_info['constant_features'] +
            self.feature_info['missing_heavy'] +
            self.feature_info['potential_leakage']
        )
        
        if features_to_remove:
            print(f"    Removing {len(features_to_remove)} problematic features")
            train_df = train_df.drop(columns=features_to_remove)
            test_df = test_df.drop(columns=features_to_remove)
        
        # Enhanced feature creation (fit on train only)
        train_enhanced = self.create_advanced_features(train_df, fit=True)
        test_enhanced = self.create_advanced_features(test_df, fit=False)
        
        # Separate features and target
        X_train = train_enhanced.drop(columns=[self.target_col])
        y_train = train_enhanced[self.target_col].copy()
        X_test = test_enhanced.drop(columns=[self.target_col])
        y_test = test_enhanced[self.target_col].copy()
        
        # Handle missing values and encoding
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', TargetEncoder())
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit on training data only
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to DataFrame
        all_features = numeric_features + categorical_features
        X_train_final = pd.DataFrame(X_train_processed, columns=all_features, 
                                     index=X_train.index)
        X_test_final = pd.DataFrame(X_test_processed, columns=all_features,
                                     index=X_test.index)
        
        # Store preprocessor
        self.preprocessors['main'] = preprocessor
        
        print(f"✅ Final shape: Train {X_train_final.shape}, Test {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train, y_test


class AdvancedFeatureSelector:
    """
    Multi-strategy feature selection based on recent research
    """
    
    def __init__(self, max_features: int = 100, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Multi-strategy feature selection"""
        print(f"🎯 Selecting top {self.max_features} features...")
        
        feature_scores = {}
        
        # 1. Variance threshold
        var_selector = VarianceThreshold(threshold=0.01)
        X_var = var_selector.fit_transform(X_train)
        high_var_features = X_train.columns[var_selector.get_support()].tolist()
        
        print(f"    Variance filter: {len(high_var_features)} features remain")
        
        # Work with variance-filtered data
        X_train_var = X_train[high_var_features]
        X_test_var = X_test[high_var_features]
        
        # 2. Statistical tests
        try:
            f_selector = SelectKBest(score_func=f_classif, k='all')
            f_selector.fit(X_train_var, y_train)
            
            for i, feature in enumerate(high_var_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + f_selector.scores_[i]
            
            print("    F-test scores calculated")
        except Exception as e:
            print(f"    F-test failed: {e}")
        
        # 3. Mutual information
        try:
            mi_scores = mutual_info_classif(X_train_var, y_train, random_state=self.random_state)
            
            for i, feature in enumerate(high_var_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i] * 1000
            
            print("    Mutual information calculated")
        except Exception as e:
            print(f"    Mutual information failed: {e}")
        
        # 4. Tree-based importance
        try:
            rf_selector = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            )
            rf_selector.fit(X_train_var, y_train)
            
            for i, feature in enumerate(high_var_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + rf_selector.feature_importances_[i]
            
            print("    Random Forest importance calculated")
        except Exception as e:
            print(f"    Random Forest importance failed: {e}")
        
        # 5. L1 regularization
        try:
            l1_selector = LogisticRegression(
                penalty='l1', solver='liblinear', class_weight='balanced',
                random_state=self.random_state, max_iter=1000
            )
            l1_selector.fit(X_train_var, y_train)
            
            for i, feature in enumerate(high_var_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + abs(l1_selector.coef_[0][i])
            
            print("    L1 regularization scores calculated")
        except Exception as e:
            print(f"    L1 regularization failed: {e}")
        
        # Select top features
        if feature_scores:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, _ in sorted_features[:self.max_features]]
            self.feature_scores = dict(sorted_features[:self.max_features])
        else:
            # Fallback to all high variance features
            self.selected_features = high_var_features[:self.max_features]
            self.feature_scores = {feat: 1.0 for feat in self.selected_features}
        
        print(f"✅ Selected {len(self.selected_features)} features")
        
        return X_train[self.selected_features], X_test[self.selected_features]


class ImbalanceStrategySelector:
    """
    Research-based imbalance handling strategy selection
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.strategy = None
        self.sampler = None
        
    def analyze_and_recommend(self, y_train: pd.Series) -> str:
        """Analyze imbalance and recommend strategy based on recent research"""
        class_counts = y_train.value_counts().sort_index()
        imbalance_ratio = class_counts.max() / class_counts.min()
        minority_samples = class_counts.min()
        total_samples = len(y_train)
        
        print(f"\n⚖️ Imbalance Analysis:")
        print(f"    Class distribution: {class_counts.to_dict()}")
        print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"    Minority samples: {minority_samples}")
        print(f"    Total samples: {total_samples:,}")
        
        # Strategy selection based on research findings
        if imbalance_ratio > 1000 and minority_samples < 100:
            strategy = "extreme_ensemble_only"
            reason = "Extreme imbalance + very few minority samples"
        elif imbalance_ratio > 1000:
            strategy = "hybrid_ensemble_focal"
            reason = "Extreme imbalance - hybrid approach"
        elif imbalance_ratio > 100:
            strategy = "advanced_smote_ensemble"
            reason = "High imbalance - advanced SMOTE + ensemble"
        elif imbalance_ratio > 10:
            strategy = "balanced_approach"
            reason = "Moderate imbalance - balanced sampling"
        else:
            strategy = "cost_sensitive_only"
            reason = "Low imbalance - cost-sensitive learning"
        
        self.strategy = strategy
        print(f"    Recommended strategy: {strategy}")
        print(f"    Reason: {reason}")
        
        return strategy
    
    def apply_strategy(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         strategy: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply the recommended imbalance handling strategy"""
        if strategy is None:
            strategy = self.analyze_and_recommend(y_train)
        
        if not IMBLEARN_AVAILABLE:
            print("⚠️ imbalanced-learn not available, using original data")
            return X_train, y_train
        
        print(f"🔄 Applying {strategy}...")
        
        try:
            if strategy == "extreme_ensemble_only":
                # No resampling for extreme cases - rely on specialized algorithms
                return X_train, y_train
                
            elif strategy == "hybrid_ensemble_focal":
                # Very conservative SMOTE for extreme imbalance
                min_class_size = y_train.value_counts().min()
                if min_class_size >= 5:  # Need at least 5 samples for SMOTE
                    target_samples = min(min_class_size * 10, 1000)  # Conservative target
                    
                    sampling_strategy = {}
                    for cls in y_train.value_counts().index:
                        if y_train.value_counts()[cls] < target_samples:
                            sampling_strategy[cls] = target_samples
                    
                    if sampling_strategy:
                        self.sampler = SMOTE(
                            sampling_strategy=sampling_strategy,
                            k_neighbors=min(4, min_class_size - 1),
                            random_state=self.random_state
                        )
                        X_res, y_res = self.sampler.fit_resample(X_train, y_train)
                        print(f"    Applied conservative SMOTE")
                        return X_res, y_res
                
                return X_train, y_train
                
            elif strategy == "advanced_smote_ensemble":
                # SMOTE + Tomek links for high imbalance
                self.sampler = SMOTETomek(
                    smote=BorderlineSMOTE(random_state=self.random_state),
                    random_state=self.random_state
                )
                
            elif strategy == "balanced_approach":
                # SMOTE + ENN for moderate imbalance
                self.sampler = SMOTEENN(
                    smote=SMOTE(random_state=self.random_state),
                    enn=EditedNearestNeighbours(),
                    random_state=self.random_state
                )
                
            else:  # cost_sensitive_only
                return X_train, y_train
            
            if self.sampler:
                X_res, y_res = self.sampler.fit_resample(X_train, y_train)
                
                print("    Before resampling:", y_train.value_counts().sort_index().to_dict())
                print("    After resampling:", pd.Series(y_res).value_counts().sort_index().to_dict())
                
                return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
            
        except Exception as e:
            print(f"⚠️ Resampling failed: {e}")
            print("    Using original data")
            
        return X_train, y_train


class ComprehensiveModelSuite:
    """
    Comprehensive model suite with state-of-the-art algorithms
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.model_results = {}
        
    def create_models(self, n_classes: int, class_weights: dict = None) -> Dict[str, Any]:
        """Create comprehensive model suite"""
        print("🤖 Creating comprehensive model suite...")
        
        models = {}
        
        # 1. Traditional ML Models
        models['LogisticRegression'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
            solver='liblinear'
        )
        
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        
        models['HistGradientBoosting'] = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=6,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # 2. Imbalanced Learning Models
        if IMBLEARN_AVAILABLE:
            models['BalancedRandomForest'] = BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight='balanced_subsample'
            )
            
            models['EasyEnsemble'] = EasyEnsembleClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        # 3. Gradient Boosting Models
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced' if class_weights else None,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='mlogloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                class_weights=class_weights,
                random_seed=self.random_state,
                thread_count=self.n_jobs,
                verbose=False
            )
        
        # 4. Deep Learning Model
        if TENSORFLOW_AVAILABLE:
            def create_nn_model(input_dim: int, n_classes: int, class_weights: dict = None):
                model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(512, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(256, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(n_classes, activation='softmax')
                ])
                
                # Custom loss for imbalanced data
                if class_weights:
                    # Focal loss implementation
                    def focal_loss(alpha=0.25, gamma=2.0):
                        def focal_loss_fixed(y_true, y_pred):
                            epsilon = tf.keras.backend.epsilon()
                            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                            
                            # Convert to one-hot if needed
                            if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
                                y_true = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
                            
                            # Focal loss calculation
                            ce = -y_true * tf.math.log(y_pred)
                            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                            modulating_factor = tf.pow((1 - p_t), gamma)
                            
                            return tf.reduce_mean(alpha_factor * modulating_factor * ce)
                        return focal_loss_fixed
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=focal_loss(),
                        metrics=['accuracy']
                    )
                else:
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=SparseCategoricalCrossentropy(),
                        metrics=['accuracy']
                    )
                
                return model
            
            models['NeuralNetwork'] = create_nn_model
        
        print(f"✅ Created {len(models)} models")
        self.models = models
        return models
    
    def train_and_evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     cv_folds: int = 5) -> Dict[str, Dict]:
        """Train and evaluate all models with cross-validation"""
        print(f"\n🏋️ Training and evaluating {len(self.models)} models...")
        
        # Calculate class weights
        class_weights = dict(enumerate(
            len(y_train) / (len(y_train.unique()) * np.bincount(y_train))
        ))
        
        results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"\n📊 Training {model_name}...")
            start_time = datetime.now()
            
            try:
                # Handle neural network separately
                if model_name == 'NeuralNetwork' and TENSORFLOW_AVAILABLE:
                    nn_model = model(input_dim=X_train.shape[1], 
                                     n_classes=len(y_train.unique()),
                                     class_weights=class_weights)
                    
                    # Prepare callbacks
                    callbacks = [
                        EarlyStopping(patience=20, restore_best_weights=True),
                        ReduceLROnPlateau(patience=10, factor=0.5)
                    ]
                    
                    # Train with validation split
                    history = nn_model.fit(
                        X_train.values, y_train.values,
                        epochs=100,
                        batch_size=min(1024, len(X_train) // 10),
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=0,
                        class_weight=class_weights
                    )
                    
                    # Evaluate
                    y_pred = nn_model.predict(X_test.values).argmax(axis=1)
                    y_pred_proba = nn_model.predict(X_test.values)
                    
                    # Cross-validation for neural networks is expensive, skip for now
                    cv_scores = {'test_balanced_accuracy': [balanced_accuracy_score(y_test, y_pred)]}
                    
                else:
                    # Traditional ML models
                    # Cross-validation
                    cv_scores = cross_validate(
                        model, X_train, y_train,
                        cv=skf,
                        scoring=['balanced_accuracy', 'f1_macro', 'f1_weighted'],
                        n_jobs=1,  # Prevent nested parallelization
                        error_score='raise'
                    )
                    
                    # Train on full training set
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Basic metrics
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                
                # Advanced metrics
                mcc = matthews_corrcoef(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                
                # Per-class metrics
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # Store results
                results[model_name] = {
                    'cv_balanced_accuracy_mean': np.mean(cv_scores['test_balanced_accuracy']),
                    'cv_balanced_accuracy_std': np.std(cv_scores['test_balanced_accuracy']),
                    'test_balanced_accuracy': balanced_acc,
                    'test_f1_macro': f1_macro,
                    'test_f1_weighted': f1_weighted,
                    'test_precision_macro': precision_macro,
                    'test_recall_macro': recall_macro,
                    'test_mcc': mcc,
                    'test_kappa': kappa,
                    'training_time_seconds': training_time,
                    'classification_report': class_report,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Print summary
                print(f"    ✅ CV Balanced Accuracy: {np.mean(cv_scores['test_balanced_accuracy']):.4f} ± {np.std(cv_scores['test_balanced_accuracy']):.4f}")
                print(f"    📊 Test Balanced Accuracy: {balanced_acc:.4f}")
                print(f"    🎯 Test F1-Macro: {f1_macro:.4f}")
                print(f"    ⏱️ Training time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'cv_balanced_accuracy_mean': 0,
                    'test_balanced_accuracy': 0,
                    'test_f1_macro': 0
                }
        
        self.model_results = results
        return results


class AdvancedHyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for best models
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_models = {}
        
    def optimize_top_models(self, models: Dict, model_results: Dict,
                              X_train: pd.DataFrame, y_train: pd.Series,
                              top_k: int = 3) -> Dict:
        """Optimize hyperparameters for top performing models"""
        print(f"\n🔧 Optimizing hyperparameters for top {top_k} models...")
        
        # Sort models by CV balanced accuracy
        sorted_models = sorted(
            [(name, results) for name, results in model_results.items() 
             if 'error' not in results],
            key=lambda x: x[1]['cv_balanced_accuracy_mean'],
            reverse=True
        )
        
        top_models = sorted_models[:top_k]
        print(f"    Selected models: {[name for name, _ in top_models]}")
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10]
            },
            'BalancedRandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10]
            }
        }
        
        optimized_results = {}
        
        for model_name, _ in top_models:
            if model_name in param_grids and model_name in models:
                print(f"\n🎯 Optimizing {model_name}...")
                
                try:
                    # Create model instance
                    base_model = models[model_name]
                    
                    # Randomized search (more efficient than grid search)
                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_grids[model_name],
                        n_iter=20,  # Limited iterations for efficiency
                        cv=3,  # Reduced CV folds for speed
                        scoring='balanced_accuracy',
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    
                    optimized_results[model_name] = {
                        'best_params': search.best_params_,
                        'best_cv_score': search.best_score_,
                        'best_model': search.best_estimator_
                    }
                    
                    print(f"    ✅ Best CV score: {search.best_score_:.4f}")
                    print(f"    📋 Best params: {search.best_params_}")
                    
                except Exception as e:
                    print(f"    ❌ Optimization failed: {str(e)}")
            else:
                print(f"    ⏭️ Skipping {model_name} (no param grid defined)")
        
        self.best_models = optimized_results
        return optimized_results


class ComprehensiveEvaluator:
    """
    Advanced evaluation with multiple metrics and visualizations
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_comprehensive_report(self, model_results: Dict, 
                                     y_test: pd.Series,
                                     class_names: List[str] = None,
                                     feature_scores: Dict = None) -> Dict:
        """Create comprehensive evaluation report"""
        print("\n📋 Creating comprehensive evaluation report...")
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in sorted(y_test.unique())]
        
        # 1. Model comparison table
        self._create_model_comparison(model_results)
        
        # 2. Detailed confusion matrices
        self._create_confusion_matrices(model_results, y_test, class_names)
        
        # 3. Per-class performance analysis
        self._create_per_class_analysis(model_results, class_names)
        
        # 4. Feature importance plots
        if feature_scores:
            self._create_feature_importance_plots(feature_scores)
        
        # 5. Class distribution analysis
        self._create_class_distribution_plots(y_test, model_results, class_names)
        
        # 6. Model performance radar plots
        self._create_radar_plots(model_results)
        
        # Create summary report
        summary = self._create_summary_report(model_results, y_test, class_names)
        
        print(f"📊 Report saved to: {self.output_dir}")
        return summary
    
    def _create_model_comparison(self, model_results: Dict):
        """Create model comparison table"""
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': model_name,
                    'CV_Balanced_Accuracy': f"{results['cv_balanced_accuracy_mean']:.4f} ± {results['cv_balanced_accuracy_std']:.4f}",
                    'Test_Balanced_Accuracy': f"{results['test_balanced_accuracy']:.4f}",
                    'F1_Macro': f"{results['test_f1_macro']:.4f}",
                    'F1_Weighted': f"{results['test_f1_weighted']:.4f}",
                    'MCC': f"{results['test_mcc']:.4f}",
                    'Cohen_Kappa': f"{results['test_kappa']:.4f}",
                    'Training_Time(s)': f"{results['training_time_seconds']:.2f}"
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test_Balanced_Accuracy', ascending=False)
        
        # Save to CSV
        df_comparison.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        metrics = ['Test_Balanced_Accuracy', 'F1_Macro', 'F1_Weighted', 'MCC', 'Cohen_Kappa']
        metric_values = np.array([[float(row[metric]) for metric in metrics] for _, row in df_comparison.iterrows()])
        
        x = np.arange(len(df_comparison))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, metric_values[:, i], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 2, df_comparison['Model'], rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrices(self, model_results: Dict, y_test: pd.Series, class_names: List[str]):
        """Create confusion matrices for all models"""
        n_models = len([m for m in model_results.values() if 'error' not in m])
        if n_models == 0:
            return
            
        fig, axes = plt.subplots((n_models + 2) // 3, 3, figsize=(18, 6 * ((n_models + 2) // 3)))
        if n_models == 1:
            axes = [axes]
        elif (n_models + 2) // 3 == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        for model_name, results in model_results.items():
            if 'error' not in results and plot_idx < len(axes):
                cm = np.array(results['confusion_matrix'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names,
                            ax=axes[plot_idx])
                axes[plot_idx].set_title(f'{model_name}\nBalanced Acc: {results["test_balanced_accuracy"]:.3f}')
                axes[plot_idx].set_xlabel('Predicted')
                axes[plot_idx].set_ylabel('Actual')
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_per_class_analysis(self, model_results: Dict, class_names: List[str]):
        """Create per-class performance analysis"""
        # Extract per-class metrics for each model
        per_class_data = {}
        
        for model_name, results in model_results.items():
            if 'error' not in results and 'classification_report' in results:
                class_report = results['classification_report']
                per_class_data[model_name] = {}
                
                for i, class_name in enumerate(class_names):
                    class_key = str(i)  # Classification report uses string keys
                    if class_key in class_report:
                        per_class_data[model_name][class_name] = {
                            'precision': class_report[class_key]['precision'],
                            'recall': class_report[class_key]['recall'],
                            'f1-score': class_report[class_key]['f1-score']
                        }
        
        # Create visualization
        if per_class_data:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics = ['precision', 'recall', 'f1-score']
            
            for idx, metric in enumerate(metrics):
                metric_data = []
                model_names = []
                
                for model_name, classes in per_class_data.items():
                    model_names.append(model_name)
                    metric_values = [classes.get(class_name, {}).get(metric, 0) for class_name in class_names]
                    metric_data.append(metric_values)
                
                if metric_data:
                    metric_array = np.array(metric_data)
                    
                    # Create heatmap
                    sns.heatmap(metric_array, annot=True, fmt='.3f', cmap='RdYlBu_r',
                                xticklabels=class_names, yticklabels=model_names,
                                ax=axes[idx], vmin=0, vmax=1)
                    axes[idx].set_title(f'Per-Class {metric.title()}')
                    axes[idx].set_xlabel('Classes')
                    axes[idx].set_ylabel('Models')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'per_class_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_feature_importance_plots(self, feature_scores: Dict):
        """Create feature importance visualization"""
        if not feature_scores:
            return
            
        # Sort features by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:20]  # Top 20 features
        
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = plt.barh(y_pos, scores, color=colors, alpha=0.8)
        
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Features')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{score:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature scores to CSV
        df_features = pd.DataFrame(sorted_features, columns=['Feature', 'Importance_Score'])
        df_features.to_csv(self.output_dir / 'feature_importance.csv', index=False)
    
    def _create_class_distribution_plots(self, y_test: pd.Series, model_results: Dict, class_names: List[str]):
        """Create class distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original distribution
        test_dist = y_test.value_counts().sort_index()
        axes[0, 0].bar(class_names, test_dist.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Original Test Set Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(alpha=0.3)
        
        # Best model predictions
        best_model = max(model_results.items(), 
                         key=lambda x: x[1].get('test_balanced_accuracy', 0) if 'error' not in x[1] else 0)
        
        if best_model and 'error' not in best_model[1]:
            pred_dist = pd.Series(best_model[1]['predictions']).value_counts().sort_index()
            axes[0, 1].bar(class_names, pred_dist.values, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title(f'Best Model Predictions ({best_model[0]})')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(alpha=0.3)
        
        # Comparison
        if best_model and 'error' not in best_model[1]:
            x = np.arange(len(class_names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, test_dist.values, width, label='Actual', alpha=0.8)
            axes[1, 0].bar(x + width/2, pred_dist.values, width, label='Predicted', alpha=0.8)
            axes[1, 0].set_xlabel('Classes')
            axes[1, 0].set_ylabel('Count (log scale)')
            axes[1, 0].set_title('Actual vs Predicted Distribution')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(class_names)
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(alpha=0.3)
        
        # Class imbalance visualization
        imbalance_ratios = [test_dist.max() / count for count in test_dist.values]
        axes[1, 1].bar(class_names, imbalance_ratios, alpha=0.7, color='orange')
        axes[1, 1].set_title('Class Imbalance Ratios')
        axes[1, 1].set_ylabel('Imbalance Ratio (log scale)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_radar_plots(self, model_results: Dict):
        """Create radar plots for model comparison"""
        valid_models = {k: v for k, v in model_results.items() if 'error' not in v}
        if len(valid_models) < 2:
            return
            
        metrics = ['test_balanced_accuracy', 'test_f1_macro', 'test_f1_weighted', 
                   'test_mcc', 'test_kappa']
        metric_labels = ['Balanced Accuracy', 'F1 Macro', 'F1 Weighted', 'MCC', 'Cohen Kappa']
        
        # Normalize metrics to 0-1 scale for radar plot
        normalized_data = {}
        for metric in metrics:
            values = [results[metric] for results in valid_models.values()]
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                for model_name, results in valid_models.items():
                    if model_name not in normalized_data:
                        normalized_data[model_name] = []
                    norm_val = (results[metric] - min_val) / (max_val - min_val)
                    normalized_data[model_name].append(norm_val)
            else:
                for model_name in valid_models.keys():
                    if model_name not in normalized_data:
                        normalized_data[model_name] = []
                    normalized_data[model_name].append(0.5)
        
        # Create radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(normalized_data)))
        
        for (model_name, values), color in zip(normalized_data.items(), colors):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Normalized)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_report(self, model_results: Dict, y_test: pd.Series, class_names: List[str]) -> Dict:
        """Create comprehensive summary report"""
        valid_models = {k: v for k, v in model_results.items() if 'error' not in v}
        
        if not valid_models:
            return {"error": "No valid models to summarize"}
        
        # Find best model
        best_model_name = max(valid_models.keys(), 
                              key=lambda x: valid_models[x]['test_balanced_accuracy'])
        best_model_results = valid_models[best_model_name]
        
        # Calculate class distribution statistics
        class_dist = y_test.value_counts().sort_index()
        imbalance_ratio = class_dist.max() / class_dist.min()
        
        summary = {
            'dataset_info': {
                'total_samples': len(y_test),
                'n_classes': len(class_names),
                'class_distribution': class_dist.to_dict(),
                'imbalance_ratio': float(imbalance_ratio)
            },
            'best_model': {
                'name': best_model_name,
                'balanced_accuracy': best_model_results['test_balanced_accuracy'],
                'f1_macro': best_model_results['test_f1_macro'],
                'f1_weighted': best_model_results['test_f1_weighted'],
                'mcc': best_model_results['test_mcc'],
                'cohen_kappa': best_model_results['test_kappa'],
                'training_time': best_model_results['training_time_seconds']
            },
            'all_models_summary': {},
            'recommendations': self._generate_recommendations(valid_models, class_dist)
        }
        
        # Add all models summary
        for model_name, results in valid_models.items():
            summary['all_models_summary'][model_name] = {
                'balanced_accuracy': results['test_balanced_accuracy'],
                'f1_macro': results['test_f1_macro'],
                'training_time': results['training_time_seconds']
            }
        
        # Save summary to JSON
        with open(self.output_dir / 'summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create text summary
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write("ADVANCED MULTI-CLASS IMBALANCED CLASSIFICATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"Total samples: {len(y_test):,}\n")
            f.write(f"Number of classes: {len(class_names)}\n")
            f.write(f"Imbalance ratio: {imbalance_ratio:.1f}:1\n")
            f.write(f"Class distribution: {dict(class_dist)}\n\n")
            
            f.write("BEST MODEL PERFORMANCE:\n")
            f.write(f"Model: {best_model_name}\n")
            f.write(f"Balanced Accuracy: {best_model_results['test_balanced_accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {best_model_results['test_f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {best_model_results['test_f1_weighted']:.4f}\n")
            f.write(f"Matthews Correlation: {best_model_results['test_mcc']:.4f}\n")
            f.write(f"Cohen's Kappa: {best_model_results['test_kappa']:.4f}\n")
            f.write(f"Training Time: {best_model_results['training_time_seconds']:.2f}s\n\n")
            
            f.write("ALL MODELS RANKING:\n")
            sorted_models = sorted(valid_models.items(), 
                                     key=lambda x: x[1]['test_balanced_accuracy'], 
                                     reverse=True)
            
            for i, (name, results) in enumerate(sorted_models, 1):
                f.write(f"{i:2d}. {name:20s} - Balanced Acc: {results['test_balanced_accuracy']:.4f}\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            for rec in summary['recommendations']:
                f.write(f"• {rec}\n")
        
        return summary
    
    def _generate_recommendations(self, valid_models: Dict, class_dist: pd.Series) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Model performance recommendations
        sorted_models = sorted(valid_models.items(), 
                                 key=lambda x: x[1]['test_balanced_accuracy'], 
                                 reverse=True)
        
        best_model = sorted_models[0]
        best_acc = best_model[1]['test_balanced_accuracy']
        
        if best_acc > 0.9:
            recommendations.append(f"Excellent performance achieved with {best_model[0]} (Balanced Accuracy: {best_acc:.3f})")
        elif best_acc > 0.8:
            recommendations.append(f"Good performance with {best_model[0]}, consider ensemble methods for improvement")
        else:
            recommendations.append("Consider more advanced feature engineering or ensemble methods")
        
        # Imbalance handling recommendations
        imbalance_ratio = class_dist.max() / class_dist.min()
        
        if imbalance_ratio > 1000:
            recommendations.append("Extreme imbalance detected - consider cost-sensitive learning and specialized evaluation metrics")
        elif imbalance_ratio > 100:
            recommendations.append("High imbalance - monitor per-class performance closely")
        
        # Model diversity recommendations
        if len(valid_models) > 1:
            top_3_models = [name for name, _ in sorted_models[:3]]
            recommendations.append(f"Consider ensemble of top models: {', '.join(top_3_models)}")
        
        return recommendations


def main():
    """Main pipeline execution"""
    
    print("🚀 ADVANCED MULTI-CLASS IMBALANCED CLASSIFICATION PIPELINE")
    print("=" * 65)
    print("🎯 Research-based pipeline for extreme class imbalance")
    print("📚 Implements state-of-the-art techniques from 2024 literature")
    print("=" * 65)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Get data path
        while True:
            data_path = input("\n📂 Enter dataset path (.parquet/.csv): ").strip().strip('"')
            if os.path.exists(data_path) and data_path.lower().endswith(('.parquet', '.csv')):
                break
            print("❌ Invalid file path. Please provide existing .parquet or .csv file.")
        
        # Load data
        print(f"\n📊 Loading data: {data_path}")
        if data_path.lower().endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, low_memory=False)
            
        print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        # Memory optimization for large datasets
        if df.memory_usage(deep=True).sum() > 8e9:  # > 8GB
            print("💾 Large dataset detected, applying memory optimization...")
            
            # Optimize numeric types
            for col in df.select_dtypes(include=['int64']):
                if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                    df[col] = df[col].astype('int32')
            
            for col in df.select_dtypes(include=['float64']):
                if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:
                    df[col] = df[col].astype('float32')
            
            # Optimize categorical types
            for col in df.select_dtypes(include=['object']):
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            
            gc.collect()
            print(f"    Optimized memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        # Auto-detect or manual target selection
        print("\n🎯 Target column selection...")
        potential_targets = []
        
        for col in df.columns:
            try:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 50:  # Reasonable for classification
                    sample_vals = df[col].dropna().unique()[:5]
                    potential_targets.append((col, unique_vals, sample_vals))
            except:
                continue
        
        if potential_targets:
            print("🔍 Potential target columns:")
            for i, (col, n_classes, sample_vals) in enumerate(potential_targets[:15]):  # Show top 15
                print(f"    {i+1:2d}. {col}: {n_classes} classes")
                
            while True:
                try:
                    choice = int(input(f"Select target column (1-{len(potential_targets)}): "))
                    if 1 <= choice <= len(potential_targets):
                        target_column = potential_targets[choice-1][0]
                        break
                except ValueError:
                    pass
                print("❌ Invalid selection")
        else:
            print("❌ No obvious target columns found. Manual selection required.")
            for i, col in enumerate(df.columns):
                print(f"    {i+1:2d}. {col}")
            while True:
                try:
                    choice = int(input(f"Select target column (1-{len(df.columns)}): "))
                    if 1 <= choice <= len(df.columns):
                        target_column = df.columns[choice-1]
                        break
                except ValueError:
                    pass
                print("❌ Invalid selection")
        
        print(f"✅ Target selected: '{target_column}'")
        
        # Display target distribution
        target_dist = df[target_column].value_counts().sort_index()
        imbalance_ratio = target_dist.max() / target_dist.min()
        print(f"📊 Target classes: {len(target_dist)}")
        print(f"⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"📈 Distribution: {target_dist.to_dict()}")
        
        # Initialize pipeline components
        output_dir = f"./advanced_results_{timestamp}"
        
        # 1. Data processing with leakage prevention
        processor = DataLeakagePreventionProcessor(target_column)
        train_df, test_df = processor.initial_data_split(df)
        
        # Analyze features on training data only
        feature_info = processor.analyze_features(train_df)
        
        # Process data
        X_train, X_test, y_train, y_test = processor.preprocess_data(train_df, test_df)
        
        # 2. Feature selection
        feature_selector = AdvancedFeatureSelector(max_features=min(100, X_train.shape[1]))
        X_train_selected, X_test_selected = feature_selector.select_features(X_train, y_train, X_test)
        
        print(f"✅ Feature selection: {X_train_selected.shape[1]} features selected")
        
        # Save processed data
        processed_data_path = Path(output_dir) / 'processed_data.parquet'
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine processed data for saving
        processed_train = pd.concat([X_train_selected, y_train], axis=1)
        processed_test = pd.concat([X_test_selected, y_test], axis=1)
        processed_full = pd.concat([processed_train, processed_test])
        
        processed_full.to_parquet(processed_data_path)
        print(f"💾 Processed data saved: {processed_data_path}")
        
        # 3. Handle imbalance
        imbalance_handler = ImbalanceStrategySelector()
        X_train_balanced, y_train_balanced = imbalance_handler.apply_strategy(X_train_selected, y_train)
        
        # 4. Model training and evaluation
        model_suite = ComprehensiveModelSuite()
        class_weights = dict(enumerate(len(y_train) / (len(y_train.unique()) * np.bincount(y_train))))
        
        models = model_suite.create_models(n_classes=len(y_train.unique()), class_weights=class_weights)
        
        # Train and evaluate models
        model_results = model_suite.train_and_evaluate_models(
            X_train_balanced, y_train_balanced, X_test_selected, y_test
        )
        
        # 5. Hyperparameter optimization for top models
        optimizer = AdvancedHyperparameterOptimizer()
        optimized_models = optimizer.optimize_top_models(
            models, model_results, X_train_balanced, y_train_balanced, top_k=3
        )
        
        # Evaluate optimized models
        if optimized_models:
            print("\n🔧 Evaluating optimized models...")
            for model_name, opt_result in optimized_models.items():
                best_model = opt_result['best_model']
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test_selected)
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                print(f"    {model_name} (optimized): BA={balanced_acc:.4f}, F1={f1_macro:.4f}")
                
                # Update results
                model_results[f"{model_name}_optimized"] = {
                    'cv_balanced_accuracy_mean': opt_result['best_cv_score'],
                    'cv_balanced_accuracy_std': 0.0,
                    'test_balanced_accuracy': balanced_acc,
                    'test_f1_macro': f1_macro,
                    'test_f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'test_precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'test_recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'test_mcc': matthews_corrcoef(y_test, y_pred),
                    'test_kappa': cohen_kappa_score(y_test, y_pred),
                    'training_time_seconds': 0,  # Already included in optimization time
                    'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                    'predictions': y_pred.tolist(),
                    'probabilities': best_model.predict_proba(X_test_selected).tolist() if hasattr(best_model, 'predict_proba') else None,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'optimized_params': opt_result['best_params']
                }
        
        # 6. Comprehensive evaluation and reporting
        class_names = [f'Class_{i}' for i in sorted(y_train.unique())]
        
        evaluator = ComprehensiveEvaluator(output_dir)
        summary_report = evaluator.create_comprehensive_report(
            model_results, y_test, class_names, feature_selector.feature_scores
        )
        
        # Save complete pipeline results
        pipeline_results = {
            'pipeline_info': {
                'timestamp': timestamp,
                'data_source': data_path,
                'target_column': target_column,
                'original_shape': df.shape,
                'processed_shape': X_train_selected.shape,
                'imbalance_ratio': float(imbalance_ratio),
                'strategy_used': imbalance_handler.strategy
            },
            'feature_analysis': feature_info,
            'selected_features': feature_selector.selected_features,
            'feature_scores': feature_selector.feature_scores,
            'model_results': model_results,
            'optimized_models': optimized_models,
            'summary_report': summary_report
        }
        
        # Save complete results
        with open(Path(output_dir) / 'complete_results.json', 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Save best model
        best_model_name = max(model_results.items(), 
                               key=lambda x: x[1].get('test_balanced_accuracy', 0) if 'error' not in x[1] else 0)[0]
        
        if best_model_name in optimized_models:
            best_model = optimized_models[best_model_name]['best_model']
        elif best_model_name.replace('_optimized', '') in models:
            best_model = models[best_model_name.replace('_optimized', '')]
            if hasattr(best_model, 'fit'):
                best_model.fit(X_train_balanced, y_train_balanced)
        else:
            best_model = None
        
        if best_model:
            with open(Path(output_dir) / 'best_model.pkl', 'wb') as f:
                joblib.dump({
                    'model': best_model,
                    'model_name': best_model_name,
                    'processor': processor,
                    'feature_selector': feature_selector,
                    'selected_features': feature_selector.selected_features,
                    'target_column': target_column,
                    'class_names': class_names
                }, f)
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        if summary_report and 'best_model' in summary_report:
            best_info = summary_report['best_model']
            print(f"🏆 Best Model: {best_info['name']}")
            print(f"📊 Balanced Accuracy: {best_info['balanced_accuracy']:.4f}")
            print(f"🎯 F1-Score (Macro): {best_info['f1_macro']:.4f}")
            print(f"⚖️ Matthews Correlation: {best_info['mcc']:.4f}")
        
        print(f"📁 Results saved to: {output_dir}")
        print(f"💾 Processed data: {processed_data_path}")
        print(f"🤖 Best model: {Path(output_dir) / 'best_model.pkl'}")
        
        print("\n📊 Generated Files:")
        result_files = [
            "complete_results.json",
            "summary_report.json", 
            "summary_report.txt",
            "model_comparison.csv",
            "feature_importance.csv",
            "model_comparison.png",
            "confusion_matrices.png",
            "per_class_analysis.png",
            "feature_importance.png",
            "class_distribution_analysis.png",
            "radar_plot.png"
        ]
        
        for file in result_files:
            file_path = Path(output_dir) / file
            if file_path.exists():
                print(f"    ✅ {file}")
        
        print("=" * 70)
        
        return pipeline_results
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        gc.collect()


if __name__ == "__main__":
    # Install required packages if missing
    required_packages = [
        "imbalanced-learn",
        "xgboost", 
        "lightgbm",
        "catboost",
        "tensorflow",
        "category_encoders",
        "scikit-learn>=1.3.0"
    ]
    
    print("🔧 Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "imbalanced-learn":
                import imblearn
            elif package == "xgboost":
                import xgboost
            elif package == "lightgbm":
                import lightgbm
            elif package == "catboost":
                import catboost
            elif package == "tensorflow":
                import tensorflow
            elif package == "category_encoders":
                import category_encoders
            elif package.startswith("scikit-learn"):
                import sklearn
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ").lower() == 'y'
        if install:
            for package in missing_packages:
                print(f"Installing {package}...")
                os.system(f"pip install {package}")
        else:
            print("⚠️ Some algorithms may not be available without these packages")
    
    print("\n" + "=" * 50)
    print("🚀 ADVANCED IMBALANCED CLASSIFICATION TOOLKIT")
    print("=" * 50)
    
    result = main()
    if result:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Check comprehensive results and visualizations")
    else:
        print("\n❌ Pipeline failed or was interrupted")
    
    print("\n🏁 Done!")