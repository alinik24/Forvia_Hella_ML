import pandas as pd
import numpy as np
import logging
import time
import json
import gc
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import joblib
import psutil
import warnings
from tqdm import tqdm
from itertools import combinations
import re

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin

# Advanced libraries with error handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.ensemble import BalancedRandomForestClassifier
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    from scipy.stats import chi2_contingency, spearmanr, entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import category_encoders as ce
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class EnhancedConfig:
    """Configuration for the preprocessing pipeline"""
    target_column: str
    output_dir: str = "./enhanced_preprocessing_results"
    
    # Cardinality and encoding thresholds
    high_cardinality_threshold: float = 0.95
    cardinality_frequency_threshold: int = 10
    binary_threshold: int = 2
    onehot_threshold: int = 15
    target_encoding_threshold: int = 50
    frequency_threshold: int = 100
    hash_encoding_threshold: int = 1000
    
    # Quality filtering thresholds
    missing_threshold: float = 0.90
    variance_threshold: float = 0.0001
    correlation_threshold: float = 0.98
    
    # Memory and performance settings
    batch_size: int = 100000
    max_memory_gb: float = 50.0  # Updated to reflect your 50 GB RAM
    use_dask: bool = False
    n_jobs: int = -1
    random_state: int = 42

class IntelligentFeatureAnalyzer:
    """Intelligent feature analysis for data-driven preprocessing"""
    
    def __init__(self, config: EnhancedConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Dict]:
        """Comprehensive feature analysis with progress tracking"""
        self.logger.info("🔍 Starting intelligent feature analysis")
        
        analysis = {}
        for col in tqdm(df.columns, desc="Analyzing features", disable=len(df.columns) < 10):
            if target_col and col == target_col:
                continue
            analysis[col] = self._analyze_single_feature(df[col], col, df[target_col] if target_col else None)
        
        self.logger.info("✅ Feature analysis completed")
        return analysis
    
    def _analyze_single_feature(self, series: pd.Series, col_name: str, target: Optional[pd.Series] = None) -> Dict:
        """Analyze a single feature"""
        analysis = {
            'name': col_name,
            'dtype': str(series.dtype),
            'nunique': series.nunique(),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100
        }
        
        analysis['total_count'] = len(series)
        analysis['cardinality_ratio'] = analysis['nunique'] / analysis['total_count']
        analysis['is_numeric'] = pd.api.types.is_numeric_dtype(series)
        analysis['is_categorical'] = not analysis['is_numeric']
        analysis['feature_type'] = self._detect_feature_type(series, col_name, target)
        analysis['cardinality_level'] = self._classify_cardinality(analysis)
        
        if analysis['feature_type'] == 'numeric':
            analysis.update(self._analyze_numeric_distribution(series))
        elif analysis['feature_type'] == 'categorical':
            analysis.update(self._analyze_categorical_distribution(series))
        elif analysis['feature_type'] == 'temporal':
            analysis.update(self._analyze_temporal_distribution(series))
            
        analysis['recommended_action'] = self._recommend_action(analysis)
        
        return analysis
    
    def _detect_feature_type(self, series: pd.Series, col_name: str, target: Optional[pd.Series]) -> str:
        """Detect feature type using statistical properties"""
        nunique = series.nunique()
        cardinality_ratio = nunique / len(series)
        
        try:
            temp_series = pd.to_datetime(series, errors='coerce')
            if temp_series.notna().mean() > 0.9:
                return 'temporal'
        except:
            pass
            
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int8', 'int16', 'int32', 'int64'] and nunique <= self.config.onehot_threshold:
                return 'categorical'
            return 'numeric'
        
        if SCIPY_AVAILABLE and target is not None and pd.api.types.is_numeric_dtype(target):
            value_counts = series.value_counts()
            probabilities = value_counts / value_counts.sum()
            feature_entropy = entropy(probabilities)
            if feature_entropy < 1.0 or nunique <= self.config.onehot_threshold:
                return 'categorical'
        
        return 'categorical'
    
    def _classify_cardinality(self, analysis: Dict) -> str:
        """Classify cardinality level"""
        ratio = analysis['cardinality_ratio']
        nunique = analysis['nunique']
        
        if ratio >= self.config.high_cardinality_threshold:
            return 'very_high'
        elif nunique > self.config.hash_encoding_threshold:
            return 'high'
        elif nunique > self.config.target_encoding_threshold:
            return 'medium'
        elif nunique > self.config.onehot_threshold:
            return 'low_medium'
        else:
            return 'low'
    
    def _analyze_numeric_distribution(self, series: pd.Series) -> Dict:
        """Analyze numeric feature distribution"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'distribution_type': 'empty'}
        
        if pd.api.types.is_bool_dtype(non_null_series):
            return {
                'distribution_type': 'boolean',
                'value_counts': non_null_series.value_counts().to_dict()
            }
            
        try:
            numeric_series = pd.to_numeric(non_null_series, errors='coerce')
            if numeric_series.isna().all():
                return {
                    'distribution_type': 'non_numeric',
                    'value_counts': non_null_series.value_counts().to_dict()
                }
                
            analysis = {
                'distribution_type': 'numeric',
                'mean': numeric_series.mean(),
                'median': numeric_series.median(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'q25': numeric_series.quantile(0.25),
                'q75': numeric_series.quantile(0.75)
            }
            
            iqr = analysis['q75'] - analysis['q25']
            lower_fence = analysis['q25'] - 1.5 * iqr
            upper_fence = analysis['q75'] + 1.5 * iqr
            outliers = ((numeric_series < lower_fence) | (numeric_series > upper_fence)).sum()
            analysis['outlier_count'] = outliers
            analysis['outlier_percentage'] = outliers / len(numeric_series) * 100
            analysis['likely_categorical'] = (numeric_series.dtype in ['int8', 'int16', 'int32', 'int64'] and 
                                            series.nunique() <= self.config.onehot_threshold)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Numeric analysis failed for {series.name}: {e}")
            return {
                'distribution_type': 'error',
                'error_message': str(e),
                'value_counts': non_null_series.value_counts().to_dict()
            }
    
    def _analyze_categorical_distribution(self, series: pd.Series) -> Dict:
        """Analyze categorical feature distribution"""
        value_counts = series.value_counts()
        
        analysis = {
            'distribution_type': 'categorical',
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'frequency_distribution': value_counts.head(10).to_dict()
        }
        
        if len(value_counts) > 0:
            analysis['frequency_mean'] = value_counts.mean()
            analysis['frequency_std'] = value_counts.std()
            analysis['dominant_category_ratio'] = value_counts.iloc[0] / len(series)
            rare_categories = (value_counts < self.config.cardinality_frequency_threshold).sum()
            analysis['rare_categories_count'] = rare_categories
            analysis['rare_categories_ratio'] = rare_categories / len(value_counts) if len(value_counts) > 0 else 0
        
        return analysis
    
    def _analyze_temporal_distribution(self, series: pd.Series) -> Dict:
        """Analyze temporal feature distribution"""
        try:
            temp_series = pd.to_datetime(series, errors='coerce')
            analysis = {
                'distribution_type': 'temporal',
                'min_timestamp': temp_series.min(),
                'max_timestamp': temp_series.max(),
                'time_span_days': (temp_series.max() - temp_series.min()).days if temp_series.notna().any() else None
            }
            return analysis
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed for {series.name}: {e}")
            return {'distribution_type': 'error', 'error_message': str(e)}
    
    def _recommend_action(self, analysis: Dict) -> str:
        """Recommend preprocessing action"""
        feature_type = analysis['feature_type']
        cardinality_level = analysis['cardinality_level']
        null_percentage = analysis['null_percentage']
        
        if null_percentage > self.config.missing_threshold * 100:
            return 'drop_high_missing'
            
        if analysis['nunique'] <= 1:
            return 'drop_constant'
            
        if feature_type == 'numeric':
            return 'keep_numeric'
            
        if feature_type == 'temporal':
            return 'extract_temporal'
            
        if feature_type == 'categorical':
            if cardinality_level == 'very_high':
                return 'hash_encode'
            elif cardinality_level == 'high':
                return 'hash_encode'
            elif cardinality_level == 'medium':
                return 'target_encode'
            elif cardinality_level == 'low_medium':
                return 'onehot_encode'
            else:
                return 'binary_or_onehot_encode'
                
        return 'standard_preprocessing'
    
    def _rank_features_by_cardinality(self, df: pd.DataFrame) -> List[Dict]:
        """Rank features by cardinality with optimized batch and parallel processing"""
        self.logger.info("🔍 Ranking features by cardinality with optimized computation")
        feature_ranking = []
        
        # Calculate available memory and adjust batch size
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.logger.info(f"💾 Available memory: {available_memory_gb:.1f} GB")
        batch_size = min(self.config.batch_size, len(df) // 4)  # Use 1/4 of dataset to stay within memory limits
        batch_size = max(batch_size, 10000)  # Ensure minimum batch size for efficiency
        self.logger.info(f"📏 Using batch size: {batch_size:,} rows")

        def compute_cardinality(col: str, data: pd.DataFrame) -> Dict:
            """Compute cardinality for a single column"""
            try:
                # Estimate memory usage for column
                col_memory_mb = data[col].memory_usage(deep=True) / (1024**2)
                nunique = data[col].nunique()
                cardinality_ratio = nunique / len(data)
                feature_type = self._detect_feature_type(data[col], col, None)
                self.logger.debug(f"✅ Processed {col}: {nunique:,} unique values, {col_memory_mb:.1f} MB")
                return {
                    'name': col,
                    'cardinality': nunique,
                    'cardinality_ratio': cardinality_ratio,
                    'type': feature_type
                }
            except Exception as e:
                self.logger.warning(f"❌ Failed to analyze cardinality for {col}: {e}")
                return {
                    'name': col,
                    'cardinality': 0,
                    'cardinality_ratio': 0.0,
                    'type': 'error'
                }

        if PYARROW_AVAILABLE:
            self.logger.info("🚀 Using pyarrow for cardinality computation")
            try:
                # Convert to pyarrow table
                table = pa.Table.from_pandas(df, preserve_index=False)
                batches = [table.slice(i, batch_size) for i in range(0, len(table), batch_size)]
                
                for col in table.column_names:
                    start_time = time.time()
                    try:
                        # Use pyarrow's unique operation for efficiency
                        unique_values = set()
                        for batch in batches:
                            unique_values.update(pc.unique(batch[col]).to_pylist())
                        nunique = len(unique_values)
                        cardinality_ratio = nunique / len(df)
                        feature_type = self._detect_feature_type(df[col], col, None)  # Use pandas for type detection
                        col_memory_mb = df[col].memory_usage(deep=True) / (1024**2)
                        feature_ranking.append({
                            'name': col,
                            'cardinality': nunique,
                            'cardinality_ratio': cardinality_ratio,
                            'type': feature_type
                        })
                        self.logger.debug(f"✅ Processed {col}: {nunique:,} unique values, {col_memory_mb:.1f} MB, {time.time() - start_time:.2f}s")
                    except Exception as e:
                        self.logger.warning(f"❌ Failed to analyze cardinality for {col}: {e}")
                        feature_ranking.append({
                            'name': col,
                            'cardinality': 0,
                            'cardinality_ratio': 0.0,
                            'type': 'error'
                        })
                    gc.collect()  # Force garbage collection after each column
            except Exception as e:
                self.logger.error(f"❌ Pyarrow processing failed: {e}, falling back to pandas")
                PYARROW_AVAILABLE = False  # Disable pyarrow for fallback

        if not PYARROW_AVAILABLE:
            self.logger.info("🚀 Using pandas with parallel processing")
            from joblib import Parallel, delayed
            n_jobs = min(self.config.n_jobs if self.config.n_jobs > 0 else multiprocessing.cpu_count(), len(df.columns))
            self.logger.info(f"🔄 Using {n_jobs} parallel jobs")
            batch_size_cols = max(1, len(df.columns) // n_jobs)
            column_batches = [df.columns[i:i + batch_size_cols] for i in range(0, len(df.columns), batch_size_cols)]
            
            for batch_cols in tqdm(column_batches, desc="Processing column batches"):
                start_time = time.time()
                results = Parallel(n_jobs=n_jobs, backend='loky', mmap_mode='r')(
                    delayed(compute_cardinality)(col, df) for col in batch_cols
                )
                feature_ranking.extend(results)
                self.logger.debug(f"✅ Processed column batch: {len(batch_cols)} columns, {time.time() - start_time:.2f}s")
                gc.collect()  # Force garbage collection after each batch

        feature_ranking.sort(key=lambda x: x['cardinality'], reverse=True)
        self.logger.info("✅ Feature ranking completed")
        return feature_ranking

class EnhancedHighCardinalityHandler:
    """Handle high cardinality features with user interaction"""
    
    def __init__(self, config: EnhancedConfig, logger):
        self.config = config
        self.logger = logger
        
    def handle_high_cardinality_features(self, df: pd.DataFrame, 
                                       feature_analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Interactive handling of high cardinality features"""
        high_card_features = self._identify_high_cardinality_features(feature_analysis)
        
        if not high_card_features:
            self.logger.info("✅ No high cardinality features detected")
            return df, {'removed_features': [], 'encoding_decisions': {}}
        
        self._present_high_cardinality_analysis(high_card_features, feature_analysis)
        decisions = self._get_user_decisions(high_card_features, feature_analysis)
        df_processed, removal_info = self._apply_decisions(df, decisions, feature_analysis)
        
        return df_processed, {
            'removed_features': removal_info,
            'encoding_decisions': decisions
        }
    
    def _identify_high_cardinality_features(self, feature_analysis: Dict) -> List[str]:
        """Identify features with high cardinality"""
        high_card_features = []
        
        for feature, analysis in feature_analysis.items():
            cardinality_level = analysis['cardinality_level']
            cardinality_ratio = analysis['cardinality_ratio']
            
            if (cardinality_level in ['very_high', 'high'] or
                (cardinality_ratio > 0.8 and analysis['nunique'] > 100)):
                high_card_features.append(feature)
                
        return high_card_features
    
    def _present_high_cardinality_analysis(self, high_card_features: List[str], 
                                         feature_analysis: Dict):
        """Present analysis of high cardinality features"""
        print("\n" + "="*80)
        print("🚨 HIGH CARDINALITY FEATURES DETECTED")
        print("="*80)
        print("The following features have high cardinality and may impact model performance:")
        print()
        
        for i, feature in enumerate(high_card_features, 1):
            analysis = feature_analysis[feature]
            print(f"{i:2d}. {feature}")
            print(f"    Type: {analysis['feature_type']}")
            print(f"    Unique values: {analysis['nunique']:,} ({analysis['cardinality_ratio']:.1%} of total)")
            print(f"    Cardinality level: {analysis['cardinality_level']}")
            print(f"    Recommended action: {analysis['recommended_action']}")
            
            if analysis['feature_type'] == 'categorical' and 'frequency_distribution' in analysis:
                sample_values = list(analysis['frequency_distribution'].keys())[:5]
                print(f"    Sample values: {sample_values}")
            
            print()
        
        print("📋 RECOMMENDATIONS:")
        print("• High cardinality categorical: Consider HASH ENCODING or FREQUENCY ENCODING")
        print("• Numeric values: Keep as NUMERIC")
        print("• Temporal: Extract features (year, month, etc.)")
        print()
    
    def _get_user_decisions(self, high_card_features: List[str], 
                          feature_analysis: Dict) -> Dict[str, str]:
        """Get user decisions for high cardinality features"""
        decisions = {}
        
        print("🔧 FEATURE DECISIONS:")
        print("Options: exclude, hash, target, frequency, numeric, temporal, auto")
        print("• exclude: Remove feature")
        print("• hash: Hash encoding")
        print("• target: Target encoding")
        print("• frequency: Frequency encoding") 
        print("• numeric: Keep as numeric")
        print("• temporal: Extract temporal features")
        print("• auto: Use recommended action")
        print()
        
        for feature in high_card_features:
            analysis = feature_analysis[feature]
            recommended = analysis['recommended_action']
            
            prompt = f"{feature} (type: {analysis['feature_type']}, recommended: {recommended}): "
            decision = input(prompt).strip().lower()
            
            if not decision:
                decision = 'auto'
                
            valid_decisions = ['exclude', 'hash', 'target', 'frequency', 'numeric', 'temporal', 'auto']
            if decision not in valid_decisions:
                print(f"Invalid decision '{decision}', using 'auto'")
                decision = 'auto'
                
            decisions[feature] = decision
            
        return decisions
    
    def _apply_decisions(self, df: pd.DataFrame, decisions: Dict[str, str], 
                        feature_analysis: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """Apply user decisions"""
        removed_features = []
        
        for feature, decision in decisions.items():
            analysis = feature_analysis[feature]
            
            if decision == 'auto':
                decision = analysis['recommended_action']
                
            if decision == 'exclude':
                df = df.drop(columns=[feature])
                removed_features.append(feature)
                self.logger.info(f"🗑️ Excluded high cardinality feature: {feature}")
                self.logger.info(f"   Reason: {analysis['cardinality_level']} cardinality ({analysis['cardinality_ratio']:.1%} unique)")
                    
        return df, removed_features

class IntelligentEncoder:
    """Intelligent encoding based on feature analysis"""
    
    def __init__(self, config: EnhancedConfig, logger):
        self.config = config
        self.logger = logger
        self.encoders = {}
        self.encoding_strategies = {}
        
    def encode_features(self, df: pd.DataFrame, feature_analysis: Dict, 
                       target_info: Dict) -> pd.DataFrame:
        """Encode features with progress tracking"""
        self.logger.info("🎯 Starting intelligent feature encoding")
        
        for col in tqdm(df.columns, desc="Encoding features", disable=len(df.columns) < 10):
            if col == self.config.target_column:
                continue
                
            if col not in feature_analysis:
                continue
                
            analysis = feature_analysis[col]
            
            try:
                df = self._encode_single_feature(df, col, analysis, target_info)
            except Exception as e:
                self.logger.error(f"❌ Failed to encode {col}: {e}")
                
        self.logger.info("✅ Feature encoding completed")
        return df
    
    def _encode_single_feature(self, df: pd.DataFrame, col: str, 
                             analysis: Dict, target_info: Dict) -> pd.DataFrame:
        """Encode a single feature"""
        feature_type = analysis['feature_type']
        cardinality_level = analysis['cardinality_level']
        is_numeric = analysis['is_numeric']
        
        if feature_type == 'numeric' and not analysis.get('likely_categorical', False):
            self.encoding_strategies[col] = 'kept_numeric'
            self.logger.info(f"📊 Kept numeric: {col}")
            return df
            
        if feature_type == 'temporal':
            df = self._extract_temporal_features(df, col)
            self.encoding_strategies[col] = 'temporal_extracted'
            self.logger.info(f"🕒 Temporal features extracted: {col}")
            return df
            
        if feature_type == 'categorical' or analysis.get('likely_categorical', False):
            return self._encode_categorical_feature(df, col, analysis, target_info)
            
        return df
    
    def _encode_categorical_feature(self, df: pd.DataFrame, col: str, 
                                  analysis: Dict, target_info: Dict) -> pd.DataFrame:
        """Encode categorical features"""
        nunique = analysis['nunique']
        cardinality_level = analysis['cardinality_level']
        
        if nunique <= 1:
            df = df.drop(columns=[col])
            self.encoding_strategies[col] = 'dropped_constant'
            self.logger.info(f"🗑️ Dropped constant feature: {col}")
            return df
            
        if nunique == 2:
            df[col] = self._binary_encode(df[col])
            self.encoding_strategies[col] = 'binary'
            self.logger.info(f"🔢 Binary encoded: {col}")
            
        elif cardinality_level == 'low':
            df = self._safe_onehot_encode(df, col)
            
        elif cardinality_level in ['low_medium', 'medium']:
            if self._is_safe_for_target_encoding(analysis, target_info):
                df[col] = self._regularized_target_encode(df, col, target_info)
                self.encoding_strategies[col] = 'target_regularized'
                self.logger.info(f"🎯 Target encoded: {col}")
            else:
                df = self._merge_rare_categories(df, col)
                df[col] = self._frequency_encode(df[col])
                self.encoding_strategies[col] = 'frequency_fallback'
                self.logger.info(f"📊 Frequency encoded (target fallback): {col}")
                
        elif cardinality_level in ['high', 'very_high']:
            df = self._merge_rare_categories(df, col)
            df[col] = self._hash_encode(df[col], analysis)
            self.encoding_strategies[col] = 'hash'
            self.logger.info(f"#️⃣ Hash encoded: {col}")
            
        return df
    
    def _binary_encode(self, series: pd.Series) -> pd.Series:
        """Safe binary encoding"""
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            encoded = series.map(mapping).fillna(-1)
            self.encoders[series.name] = {'type': 'binary', 'mapping': mapping}
            return encoded
        return pd.Categorical(series).codes
    
    def _safe_onehot_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Safe one-hot encoding"""
        try:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=np.int8)
            if len(dummies.columns) == 0:
                dummy_col = f"{col}_encoded"
                df[dummy_col] = 1
                dummies = pd.DataFrame({dummy_col: df[dummy_col]})
            
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            
            self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
            self.encoding_strategies[col] = 'onehot'
            self.logger.info(f"🔥 One-hot encoded: {col} ({len(dummies.columns)} features)")
            
        except Exception as e:
            self.logger.error(f"❌ One-hot encoding failed for {col}: {e}")
            df[col] = pd.Categorical(df[col]).codes
            self.encoding_strategies[col] = 'label_fallback'
            
        return df
    
    def _is_safe_for_target_encoding(self, analysis: Dict, target_info: Dict) -> bool:
        """Check if target encoding is safe"""
        if target_info.get('imbalance_ratio', 1) > 100:
            return False
            
        avg_samples_per_category = analysis['total_count'] / analysis['nunique']
        if avg_samples_per_category < 20:
            return False
            
        if analysis.get('rare_categories_ratio', 0) > 0.5:
            return False
            
        return True
    
    def _regularized_target_encode(self, df: pd.DataFrame, col: str, target_info: Dict) -> pd.Series:
        """Regularized target encoding"""
        target_col = self.config.target_column
        global_mean = df[target_col].mean()
        
        cat_stats = df.groupby(col)[target_col].agg(['mean', 'count']).fillna(0)
        alpha = 15.0
        
        regularized_means = {}
        for category in cat_stats.index:
            cat_mean = cat_stats.loc[category, 'mean']
            cat_count = cat_stats.loc[category, 'count']
            weight = cat_count / (cat_count + alpha)
            regularized_mean = weight * cat_mean + (1 - weight) * global_mean
            regularized_means[category] = regularized_mean
        
        encoded_series = df[col].map(regularized_means).fillna(global_mean)
        
        self.encoders[col] = {
            'type': 'target_regularized',
            'mapping': regularized_means,
            'global_mean': global_mean,
            'alpha': alpha
        }
        
        return encoded_series
    
    def _frequency_encode(self, series: pd.Series) -> pd.Series:
        """Frequency encoding with smoothing"""
        freq_map = series.value_counts().to_dict()
        total_count = len(series)
        nunique = series.nunique()
        
        smoothed_freq = {k: (v + 1) / (total_count + nunique) for k, v in freq_map.items()}
        encoded = series.map(smoothed_freq).fillna(1/(total_count + nunique))
        
        self.encoders[series.name] = {
            'type': 'frequency',
            'freq_map': smoothed_freq,
            'default_value': 1/(total_count + nunique)
        }
        
        return encoded
    
    def _hash_encode(self, series: pd.Series, analysis: Dict) -> pd.Series:
        """Hash encoding for high cardinality"""
        nunique = analysis['nunique']
        
        hash_size = 64 if nunique <= 1000 else 128 if nunique <= 10000 else 256
        hash_map = {val: hash(str(val)) % hash_size for val in series.dropna().unique()}
        
        encoded = series.map(hash_map).fillna(-1)
        
        self.encoders[series.name] = {
            'type': 'hash',
            'hash_map': hash_map,
            'hash_size': hash_size,
            'default_value': -1
        }
        
        return encoded
    
    def _merge_rare_categories(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Merge rare categories into 'other'"""
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < len(df) * 0.01].index
        if len(rare_categories) > 0:
            df[col] = df[col].where(~df[col].isin(rare_categories), 'other')
            self.logger.info(f"🔧 Merged {len(rare_categories)} rare categories in {col} to 'other'")
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Extract temporal features"""
        try:
            temp_series = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"] = temp_series.dt.year
            df[f"{col}_month"] = temp_series.dt.month
            df[f"{col}_day"] = temp_series.dt.day
            df[f"{col}_hour"] = temp_series.dt.hour
            df = df.drop(columns=[col])
            self.encoders[col] = {'type': 'temporal', 'extracted': ['year', 'month', 'day', 'hour']}
            return df
        except Exception as e:
            self.logger.warning(f"Failed to extract temporal features for {col}: {e}")
            return df

class EnhancedPreprocessor:
    """Enhanced preprocessing pipeline"""
    
    def __init__(self, config: EnhancedConfig, silent: bool = False):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / 'enhanced_preprocessing.log'),
                logging.StreamHandler() if not silent else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.feature_analyzer = IntelligentFeatureAnalyzer(config, self.logger)
        self.cardinality_handler = EnhancedHighCardinalityHandler(config, self.logger)
        self.encoder = IntelligentEncoder(config, self.logger)
        
        self.removed_features = {'reasons': {}}
        self.feature_analysis = {}
        self.encoding_strategies = {}
        
        if not silent:
            self.logger.info(f"🚀 Enhanced Preprocessing Pipeline Initialized")
            self.logger.info(f"📊 Target: {config.target_column}")
            self.logger.info(f"🔧 High cardinality threshold: {config.high_cardinality_threshold}")
            self.logger.info(f"💾 Memory limit: {config.max_memory_gb}GB")
    
    def process(self, data_path: str) -> Dict[str, Any]:
        """Execute preprocessing pipeline"""
        start_time = time.time()
        
        self.logger.info("🚀 Starting enhanced preprocessing pipeline")
        
        df = self._load_data(data_path)
        df = self._optimize_dtypes(df)
        original_shape = df.shape
        
        df = self._initial_quality_filter(df)
        
        self.feature_analysis = self.feature_analyzer.analyze_features(df, self.config.target_column)
        
        df, cardinality_info = self.cardinality_handler.handle_high_cardinality_features(
            df, self.feature_analysis
        )
        
        self.feature_analysis = {k: v for k, v in self.feature_analysis.items() 
                               if k in df.columns}
        
        df, target_info = self._prepare_target(df)
        
        df = self.encoder.encode_features(df, self.feature_analysis, target_info)
        
        df = self._optimize_dtypes(df)
        
        results = self._save_results(df, original_shape, target_info, cardinality_info, start_time)
        
        self.logger.info(f"✅ Enhanced processing completed in {time.time() - start_time:.1f}s")
        return results
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data with batch processing for Parquet files"""
        self.logger.info(f"📂 Loading data: {data_path}")
        
        file_path = Path(data_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if file_path.suffix.lower() == '.parquet' and PYARROW_AVAILABLE:
            parquet_file = pq.ParquetFile(data_path)
            total_rows = parquet_file.metadata.num_rows
            batch_size = min(self.config.batch_size, total_rows)
            df_chunks = []
            
            with tqdm(total=total_rows, desc="Loading parquet") as pbar:
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    chunk = batch.to_pandas()
                    chunk = self._optimize_dtypes(chunk)
                    df_chunks.append(chunk)
                    pbar.update(len(chunk))
                    gc.collect()
            
            df = pd.concat(df_chunks, ignore_index=True)
        
        elif file_path.suffix.lower() == '.csv':
            total_lines = sum(1 for _ in open(data_path)) - 1
            batch_size = min(self.config.batch_size, total_lines)
            df_chunks = []
            with tqdm(total=total_lines, desc="Loading CSV") as pbar:
                for chunk in pd.read_csv(data_path, low_memory=False, chunksize=batch_size):
                    chunk = self._optimize_dtypes(chunk)
                    df_chunks.append(chunk)
                    pbar.update(len(chunk))
                    gc.collect()
            df = pd.concat(df_chunks, ignore_index=True)
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        self.logger.info(f"✅ Loaded: {df.shape}, {memory_mb:.1f} MB")
        
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")
        
        return df
    
    def _initial_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial quality filtering"""
        self.logger.info("🔍 Initial quality filtering")
        
        initial_cols = len(df.columns)
        
        before_dup = len(df)
        df = df.drop_duplicates()
        if before_dup != len(df):
            self.logger.info(f"🗑️ Removed {before_dup - len(df)} duplicate rows")
        
        missing_analysis = df.isnull().sum() / len(df)
        high_missing_cols = missing_analysis[missing_analysis > self.config.missing_threshold].index.tolist()
        high_missing_cols = [col for col in high_missing_cols if col != self.config.target_column]
        
        if high_missing_cols:
            self.logger.info(f"🗑️ Removing {len(high_missing_cols)} high missing columns (>{self.config.missing_threshold*100}%)")
            for col in high_missing_cols:
                self.logger.info(f"     - {col}: {missing_analysis[col]*100:.1f}% missing")
            df = df.drop(columns=high_missing_cols)
            self.removed_features['reasons']['high_missing'] = high_missing_cols
        
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            self.logger.info(f"🗑️ Removing {len(constant_cols)} constant columns")
            df = df.drop(columns=constant_cols)
            self.removed_features['reasons']['constant'] = constant_cols
        
        self._handle_missing_values(df)
        
        self.logger.info(f"📊 Initial filter: {initial_cols} → {len(df.columns)} columns")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame):
        """Handle missing values"""
        for col in tqdm(df.columns, desc="Handling missing values", disable=len(df.columns) < 10):
            if col == self.config.target_column:
                continue
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    self.logger.info(f"🔧 Filled {missing_count} missing values in {col} with median ({missing_pct:.1f}%)")
                else:
                    mode_val = df[col].mode()
                    fill_value = mode_val.iloc[0] if not mode_val.empty else 'missing'
                    df[col] = df[col].fillna(fill_value)
                    self.logger.info(f"🔧 Filled {missing_count} missing values in {col} with mode ({missing_pct:.1f}%)")
    
    def _prepare_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare target variable"""
        self.logger.info("🎯 Target preparation and analysis")
        
        y = df[self.config.target_column]
        
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            df[self.config.target_column] = label_encoder.fit_transform(y.astype(str))
            joblib.dump(label_encoder, self.output_dir / 'target_encoder.joblib')
            self.logger.info(f"🏷️ Target encoded: {len(label_encoder.classes_)} classes")
        
        value_counts = df[self.config.target_column].value_counts().sort_index()
        imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1.0
        
        target_stats = {
            'classes': len(value_counts),
            'imbalance_ratio': imbalance_ratio,
            'distribution': value_counts.to_dict(),
            'class_names': list(value_counts.index),
            'minority_class_size': value_counts.min(),
            'majority_class_size': value_counts.max(),
            'entropy': self._calculate_entropy(value_counts),
            'gini_index': self._calculate_gini_index(value_counts)
        }
        
        if imbalance_ratio >= 1000:
            severity = "EXTREME"
        elif imbalance_ratio >= 100:
            severity = "SEVERE"
        elif imbalance_ratio >= 10:
            severity = "MODERATE"
        else:
            severity = "MILD"
        
        target_stats['imbalance_severity'] = severity
        
        self.logger.info(f"🎯 Target analysis: {target_stats['classes']} classes, {severity} imbalance ({imbalance_ratio:.1f}:1)")
        self.logger.info(f"🎯 Distribution: {target_stats['distribution']}")
        self.logger.info(f"🎯 Entropy: {target_stats['entropy']:.3f}, Gini: {target_stats['gini_index']:.3f}")
        
        return df, target_stats
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of target distribution"""
        probabilities = value_counts / value_counts.sum()
        return entropy(probabilities) if SCIPY_AVAILABLE else -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_gini_index(self, value_counts: pd.Series) -> float:
        """Calculate Gini index of target distribution"""
        probabilities = value_counts / value_counts.sum()
        return 1 - np.sum(probabilities ** 2)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        self.logger.info("🗜️ Optimizing data types")
        
        initial_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        int_cols = df.select_dtypes(include=['int64']).columns
        if len(int_cols) > 0:
            for col in int_cols:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            for col in float_cols:
                test_series = df[col].astype('float32')
                if np.allclose(df[col].dropna(), test_series.dropna(), rtol=1e-6, equal_nan=True):
                    df[col] = test_series
        
        final_memory = df.memory_usage(deep=True).sum() / (1024**2)
        reduction = (initial_memory - final_memory) / initial_memory * 100 if initial_memory > 0 else 0
        
        self.logger.info(f"🗜️ Memory optimization: {initial_memory:.1f}MB → {final_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    def _save_results(self, df: pd.DataFrame, original_shape: Tuple, target_info: Dict,
                     cardinality_info: Dict, start_time: float) -> Dict:
        """Save results"""
        final_path = self.output_dir / "ENHANCED_PREPROCESSED_DATASET.parquet"
        df.to_parquet(final_path, index=False, engine='pyarrow')
        self.logger.info(f"💾 Enhanced dataset saved: {final_path}")
        
        processing_time = time.time() - start_time
        
        results = {
            'processing_info': {
                'time_seconds': processing_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pipeline_version': 'Enhanced_Data_v3.3_2025',
                'config': {
                    'target_column': self.config.target_column,
                    'high_cardinality_threshold': self.config.high_cardinality_threshold,
                    'encoding_thresholds': {
                        'binary': self.config.binary_threshold,
                        'onehot': self.config.onehot_threshold,
                        'target': self.config.target_encoding_threshold,
                        'hash': self.config.hash_encoding_threshold
                    }
                }
            },
            'transformation': {
                'original_shape': original_shape,
                'final_shape': df.shape,
                'features_removed': original_shape[1] - df.shape[1],
                'samples_change': df.shape[0] - original_shape[0],
                'memory_reduction_mb': self._calculate_memory_reduction(original_shape, df.shape)
            },
            'target_analysis': target_info,
            'cardinality_handling': cardinality_info,
            'feature_analysis': self.feature_analysis,
            'encoding_strategies': self.encoder.encoding_strategies,
            'removed_features': self.removed_features
        }
        
        with open(self.output_dir / 'enhanced_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self._generate_report(results)
        
        return results
    
    def _calculate_memory_reduction(self, original_shape: Tuple, final_shape: Tuple) -> float:
        """Calculate estimated memory reduction"""
        original_mb = (original_shape[0] * original_shape[1] * 8) / (1024**2)
        final_mb = (final_shape[0] * final_shape[1] * 8) / (1024**2)
        return original_mb - final_mb
    
    def _generate_report(self, results: Dict):
        """Generate comprehensive report"""
        report_path = self.output_dir / 'ENHANCED_PREPROCESSING_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Data Preprocessing Report v3.3\n\n")
            f.write(f"**Generated:** {results['processing_info']['timestamp']}\n")
            f.write(f"**Pipeline Version:** {results['processing_info']['pipeline_version']}\n")
            f.write(f"**Processing Time:** {results['processing_info']['time_seconds']:.1f} seconds\n\n")
            
            trans = results['transformation']
            target = results['target_analysis']
            f.write("## 📊 Executive Summary\n")
            f.write(f"- **Dataset Transformation:** {trans['original_shape'][0]:,} × {trans['original_shape'][1]} → {trans['final_shape'][0]:,} × {trans['final_shape'][1]}\n")
            f.write(f"- **Features Removed:** {trans['features_removed']}\n")
            f.write(f"- **Memory Reduction:** {trans['memory_reduction_mb']:.1f} MB\n")
            f.write(f"- **Target Classes:** {target['classes']}\n")
            f.write(f"- **Imbalance Severity:** {target['imbalance_severity']} ({target['imbalance_ratio']:.1f}:1)\n\n")
            
            cardinality = results['cardinality_handling']
            f.write("## 🔍 Feature Analysis\n")
            if cardinality['removed_features']:
                f.write(f"- **High Cardinality Features Removed:** {len(cardinality['removed_features'])}\n")
                for feature in cardinality['removed_features']:
                    f.write(f"  - {feature}\n")
                f.write("\n")
            
            encoding = results['encoding_strategies']
            f.write("## 🎯 Encoding Summary\n")
            encoding_counts = {}
            for feature, strategy in encoding.items():
                encoding_counts[strategy] = encoding_counts.get(strategy, 0) + 1
            
            for strategy, count in encoding_counts.items():
                f.write(f"- **{strategy.replace('_', ' ').title()}:** {count} features\n")
            f.write("\n")
            
            feature_types = {}
            for feature, analysis in results['feature_analysis'].items():
                ftype = analysis['feature_type']
                feature_types[ftype] = feature_types.get(ftype, 0) + 1
            
            f.write("## 🔍 Feature Type Summary\n")
            for ftype, count in feature_types.items():
                f.write(f"- **{ftype.replace('_', ' ').title()}:** {count} features\n")
            f.write("\n")
            
            f.write("## 🚀 Usage Guide\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("import joblib\n")
            f.write("from sklearn.ensemble import RandomForestClassifier\n")
            f.write("from sklearn.model_selection import train_test_split\n\n")
            
            f.write("# Load preprocessed data\n")
            f.write("df = pd.read_parquet('ENHANCED_PREPROCESSED_DATASET.parquet')\n")
            f.write(f"X = df.drop('{self.config.target_column}', axis=1)\n")
            f.write(f"y = df['{self.config.target_column}']\n\n")
            
            severity = target['imbalance_severity']
            if severity == 'EXTREME':
                f.write("# EXTREME imbalance - consider anomaly detection\n")
                f.write("from sklearn.ensemble import IsolationForest\n")
                f.write("model = IsolationForest(contamination=0.1)\n")
            elif severity in ['SEVERE', 'MODERATE']:
                f.write("# Imbalanced data - use balanced models\n")
                f.write("model = RandomForestClassifier(class_weight='balanced')\n")
            else:
                f.write("# Balanced data - standard approach\n")
                f.write("model = RandomForestClassifier()\n")
            
            f.write("```\n\n")
            
            f.write("## 💡 Recommendations\n")
            if severity == 'EXTREME':
                f.write("- Consider anomaly detection\n")
                f.write("- Use unsupervised methods\n")
                f.write("- Focus on precision metrics\n")
            elif severity == 'SEVERE':
                f.write("- Use resampling (SMOTE, ADASYN)\n")
                f.write("- Consider cost-sensitive learning\n")
                f.write("- Monitor precision-recall curves\n")
            
            if trans['features_removed'] > trans['original_shape'][1] * 0.3:
                f.write("- Many features removed - validate with domain experts\n")
        
        self.logger.info(f"📄 Report generated: {report_path}")

def main():
    """Main function with data-driven target selection"""
    print("🏭 ENHANCED DATA PREPROCESSING v3.3")
    print("Intelligent Feature Analysis • High Cardinality Handling • Optimized Encoding")
    print("=" * 90)
    
    default_path = "/home/alinzk/Forvia_Project/data/output/bookmeas/bookmeas_4_week_anonym_final_filtered_20250815_222352.parquet"
    data_path = input(f"📂 Dataset path [default: {default_path}]: ").strip().strip('"')
    if not data_path:
        data_path = default_path
    
    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        return
    
    file_path = Path(data_path)
    if file_path.suffix.lower() == '.parquet' and PYARROW_AVAILABLE:
        parquet_file = pq.ParquetFile(data_path)
        total_rows = parquet_file.metadata.num_rows
        batch_size = min(100000, total_rows)
        df_chunks = []
        
        # Initialize a single preprocessor for temporary use
        temp_processor = EnhancedPreprocessor(EnhancedConfig(target_column="temp"), silent=True)
        
        with tqdm(total=total_rows, desc="Loading parquet for target selection") as pbar:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                chunk = batch.to_pandas()
                chunk = temp_processor._optimize_dtypes(chunk)
                df_chunks.append(chunk)
                pbar.update(len(chunk))
                gc.collect()
        
        df = pd.concat(df_chunks, ignore_index=True)
    
    elif file_path.suffix.lower() == '.csv':
        total_lines = sum(1 for _ in open(data_path)) - 1
        batch_size = min(100000, total_lines)
        df_chunks = []
        temp_processor = EnhancedPreprocessor(EnhancedConfig(target_column="temp"), silent=True)
        with tqdm(total=total_lines, desc="Loading CSV for target selection") as pbar:
            for chunk in pd.read_csv(data_path, low_memory=False, chunksize=batch_size):
                chunk = temp_processor._optimize_dtypes(chunk)
                df_chunks.append(chunk)
                pbar.update(len(chunk))
                gc.collect()
        df = pd.concat(df_chunks, ignore_index=True)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    temp_config = EnhancedConfig(target_column="temp")
    temp_analyzer = IntelligentFeatureAnalyzer(temp_config, logging.getLogger(__name__))
    
    feature_ranking = temp_analyzer._rank_features_by_cardinality(df)
    
    print("\n🎯 AVAILABLE COLUMNS (Ranked by Cardinality):")
    print("=" * 80)
    print(f"{'ID':<4} {'Column Name':<30} {'Type':<15} {'Unique Values':<15} {'Cardinality Ratio':<15}")
    print("-" * 80)
    for i, feature in enumerate(feature_ranking, 1):
        print(f"{i:<4} {feature['name']:<30} {feature['type']:<15} {feature['cardinality']:<15} {feature['cardinality_ratio']:.1%}")
    print("=" * 80)
    
    while True:
        target_idx = input("🎯 Select target column by ID (or 'q' to quit): ").strip()
        if target_idx.lower() == 'q':
            return
        try:
            target_idx = int(target_idx) - 1
            if 0 <= target_idx < len(feature_ranking):
                target_column = feature_ranking[target_idx]['name']
                break
            else:
                print(f"❌ Invalid ID. Please select a number between 1 and {len(feature_ranking)}")
        except ValueError:
            print("❌ Invalid input. Please enter a valid ID number or 'q' to quit")
    
    print(f"\n🎯 Selected target column: {target_column}")
    
    print(f"\n⚙️ CONFIGURATION:")
    high_card_threshold = input(f"🔍 High cardinality threshold (0.0-1.0) [default: 0.95]: ").strip()
    high_card_threshold = float(high_card_threshold) if high_card_threshold else 0.95
    
    available_memory = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Available system memory: {available_memory:.1f}GB")
    max_memory = input(f"💾 Max memory limit (GB) [default: {min(50.0, available_memory*0.8):.1f}]: ").strip()
    max_memory = float(max_memory) if max_memory else min(50.0, available_memory*0.8)
    
    timestamp = int(time.time())
    config = EnhancedConfig(
        target_column=target_column,
        output_dir=f"./enhanced_preprocessing_results_{timestamp}",
        high_cardinality_threshold=high_card_threshold,
        max_memory_gb=max_memory
    )
    
    print(f"\n📋 FINAL CONFIGURATION:")
    print(f"📁 Output: {config.output_dir}")
    print(f"🎯 Target: {config.target_column}")
    print(f"🔍 High cardinality threshold: {config.high_cardinality_threshold}")
    print(f"💾 Memory limit: {config.max_memory_gb}GB")
    
    proceed = input(f"\n🚀 Start enhanced processing? (Y/n): ").lower()
    if proceed in ['n', 'no']:
        return
    
    try:
        processor = EnhancedPreprocessor(config)
        results = processor.process(data_path)
        
        print("\n" + "=" * 90)
        print("🎉 ENHANCED PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        
        trans = results['transformation']
        target = results['target_analysis']
        cardinality = results['cardinality_handling']
        
        print(f"📊 TRANSFORMATION:")
        print(f"   Original: {trans['original_shape'][0]:,} × {trans['original_shape'][1]}")
        print(f"   Final: {trans['final_shape'][0]:,} × {trans['final_shape'][1]}")
        print(f"   Features removed: {trans['features_removed']}")
        print(f"   Memory saved: {trans['memory_reduction_mb']:.1f} MB")
        
        print(f"\n🎯 TARGET ANALYSIS:")
        print(f"   Classes: {target['classes']}")
        print(f"   Imbalance: {target['imbalance_severity']} ({target['imbalance_ratio']:.1f}:1)")
        print(f"   Distribution: {target['distribution']}")
        
        if cardinality['removed_features']:
            print(f"\n🔍 HIGH CARDINALITY HANDLING:")
            print(f"   Features removed: {len(cardinality['removed_features'])}")
            print(f"   Removed features: {', '.join(cardinality['removed_features'][:5])}")
            if len(cardinality['removed_features']) > 5:
                print(f"   ... and {len(cardinality['removed_features'])-5} more")
        
        encoding = results['encoding_strategies']
        encoding_counts = {}
        for strategy in encoding.values():
            encoding_counts[strategy] = encoding_counts.get(strategy, 0) + 1
        
        print(f"\n🎯 ENCODING SUMMARY:")
        for strategy, count in encoding_counts.items():
            print(f"   {strategy.replace('_', ' ').title()}: {count} features")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   📊 ENHANCED_PREPROCESSED_DATASET.parquet")
        print(f"   📋 ENHANCED_PREPROCESSING_REPORT.md")
        print(f"   📋 enhanced_results.json")
        print(f"   📝 enhanced_preprocessing.log")
        
        print(f"\n🚀 QUICK START:")
        print(f"df = pd.read_parquet('{config.output_dir}/ENHANCED_PREPROCESSED_DATASET.parquet')")
        
        severity = target['imbalance_severity']
        if severity == 'EXTREME':
            print(f"# EXTREME imbalance detected - consider anomaly detection")
        elif severity in ['SEVERE', 'MODERATE']:
            print(f"# {severity} imbalance - use balanced models")
        
        print(f"\n📂 Results saved to: {config.output_dir}")
        print("=" * 90)
        
    except Exception as e:
        print(f"\n❌ PROCESSING FAILED: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   • Check target column name")
        print(f"   • Verify file format (CSV/Parquet)")
        print(f"   • Check available memory")
        print(f"   • Review log file for details")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()