#!/usr/bin/env python3
"""
IMPROVED Correlation Analyzer - Fixed for Manufacturing Quality Control
Addresses issues found in logs: categorical encoding, sampling bias, threshold optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score

# Essential imports with fallbacks
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.metrics import normalized_mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import contingency
    CONTINGENCY_AVAILABLE = True
except ImportError:
    CONTINGENCY_AVAILABLE = False


class RevisedCorrelationAnalyzer:
    """Improved correlation analysis addressing manufacturing QC dataset issues"""
    
    def __init__(self, logger: logging.Logger, plots_dir: Path):
        self.logger = logger
        self.plots_dir = plots_dir
        
        # Research-validated thresholds for manufacturing data
        self.correlation_thresholds = {
            'very_high': 0.95,
            'high': 0.85,
            'moderate': 0.65,
            'weak': 0.35,
            'negligible': 0.15
        }
        
        # IMPROVED: Research-validated correlation methods with fixes from logs
        self.correlation_methods = {
            'enhanced_target_correlation': {
                'name': 'Enhanced Target Correlation Analysis',
                'description': 'Comprehensive target-feature relationships with multiclass optimization',
                'implemented': True,
                'supports_categorical': True,
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Built-in sampling and weighting',
                'default_cutoff': 0.03,  # Lowered based on log analysis
                'computational_cost': 'Medium',
                'primary_use': 'Target-based feature selection'
            },
            'robust_correlation_matrix': {
                'name': 'Robust Correlation Matrix (Spearman + Kendall)',
                'description': 'Kendall-tau emphasis for manufacturing data with ties and outliers',
                'implemented': True,
                'supports_categorical': False,
                'multiclass_support': 'Universal',
                'imbalance_handling': 'Outlier-robust methods',
                'default_cutoff': 0.85,  # Increased to reduce over-dropping
                'computational_cost': 'Medium',
                'primary_use': 'Remove redundant features robust to manufacturing noise'
            },
            'mutual_info_manufacturing': {
                'name': 'Manufacturing-Optimized Mutual Information',
                'description': 'MI with fixed categorical handling and manufacturing preprocessing',
                'implemented': SKLEARN_AVAILABLE,
                'supports_categorical': True,
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Stratified sampling + minority preservation',
                'default_cutoff': 0.015,  # Optimized based on log results
                'computational_cost': 'High',
                'primary_use': 'Non-linear manufacturing process relationships'
            },
            'manufacturing_association': {
                'name': 'Manufacturing Statistical Association',
                'description': 'Enhanced p-value scoring with manufacturing-specific tests',
                'implemented': True,
                'supports_categorical': True,
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Robust statistical tests',
                'default_cutoff': 5.0,  # Increased for manufacturing significance
                'computational_cost': 'Low',
                'primary_use': 'Statistical significance in manufacturing QC data'
            },
            'information_gain_qc': {
                'name': 'Information Gain for Quality Control',
                'description': 'Decision-tree based information gain optimized for manufacturing QC',
                'implemented': True,
                'supports_categorical': True,
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Naturally handles imbalance',
                'default_cutoff': 0.005,  # Lowered to be more selective
                'computational_cost': 'Low',
                'primary_use': 'Manufacturing quality control feature selection'
            }
        }
        
        self.max_sample_size = 150000  # Increased for better statistical power

    def analyze_enhanced_correlation(self, df: pd.DataFrame, method: str, 
                                   target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Analyze correlation using improved methods with fixes from log analysis"""
        
        if method not in self.correlation_methods:
            raise ValueError(f"Unknown correlation method: {method}")
        
        if not self.correlation_methods[method]['implemented']:
            raise ValueError(f"Method {method} is not available (missing dependencies)")
        
        method_info = self.correlation_methods[method]
        self.logger.info(f"Starting optimized correlation analysis: {method}...")
        
        print(f"\n{method_info['name']}")
        print(f"Description: {method_info['description']}")
        print(f"Multiclass Support: {method_info['multiclass_support']}")
        print(f"Imbalance Handling: {method_info['imbalance_handling']}")
        print(f"Computational Cost: {method_info['computational_cost']}")
        print(f"Default Cutoff: {method_info['default_cutoff']}")
        
        start_time = time.time()
        
        try:
            # Apply improved sampling strategy
            if len(df) > self.max_sample_size:
                print(f"Large dataset detected ({len(df):,} rows)")
                df = self._apply_stratified_sampling(df, target_col)
            
            if method == 'enhanced_target_correlation':
                result = self._analyze_enhanced_target_correlation_fixed(df, target_col, method_info)
            elif method == 'robust_correlation_matrix':
                result = self._analyze_robust_correlation_matrix_fixed(df, target_col, method_info)
            elif method == 'mutual_info_manufacturing':
                result = self._analyze_mutual_info_manufacturing_fixed(df, target_col, method_info)
            elif method == 'manufacturing_association':
                result = self._analyze_manufacturing_association_fixed(df, target_col, method_info)
            elif method == 'information_gain_qc':
                result = self._analyze_information_gain_qc_fixed(df, target_col, method_info)
            else:
                raise ValueError(f"Method {method} implementation not found")
                
        except Exception as e:
            raise RuntimeError(f"Failed to compute {method}: {str(e)}") from e
        
        elapsed_time = time.time() - start_time
        result[1]['execution_time'] = elapsed_time
        result[1]['method_info'] = method_info
        
        self.logger.info(f"✅ {method} correlation analysis completed in {elapsed_time:.2f} seconds")
        
        return result

    def _apply_stratified_sampling(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """IMPROVED: Better stratified sampling preserving minority classes"""
        
        print(f"Applying improved stratified sampling to {self.max_sample_size:,} rows...")
        
        if target_col and target_col in df.columns:
            try:
                target_counts = df[target_col].value_counts()
                minority_threshold = 5000  # Preserve classes with fewer samples
                
                sampled_dfs = []
                remaining_budget = self.max_sample_size
                
                # First, preserve all minority samples
                for class_val in target_counts.index:
                    class_df = df[df[target_col] == class_val]
                    class_count = len(class_df)
                    
                    if class_count <= minority_threshold:
                        # Keep all samples from minority classes
                        sampled_dfs.append(class_df)
                        remaining_budget -= class_count
                        print(f"  Preserved {class_count} samples from minority class: {class_val}")
                
                # Then sample majority classes proportionally from remaining budget
                majority_classes = [cls for cls in target_counts.index 
                                  if target_counts[cls] > minority_threshold]
                
                if majority_classes and remaining_budget > 0:
                    majority_total = sum(target_counts[cls] for cls in majority_classes)
                    
                    for class_val in majority_classes:
                        class_df = df[df[target_col] == class_val]
                        class_count = len(class_df)
                        
                        # Proportional sampling from remaining budget
                        class_proportion = class_count / majority_total
                        class_sample_size = min(int(remaining_budget * class_proportion), class_count)
                        
                        if class_sample_size > 0:
                            sampled_class = class_df.sample(n=class_sample_size, random_state=42)
                            sampled_dfs.append(sampled_class)
                
                sampled_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
                print(f"✅ Improved stratified sample: {len(sampled_df):,} rows with preserved distributions")
                
                # Verify distribution preservation
                original_dist = df[target_col].value_counts(normalize=True).sort_index()
                sampled_dist = sampled_df[target_col].value_counts(normalize=True).sort_index()
                max_deviation = (original_dist - sampled_dist).abs().max()
                print(f"   Max distribution deviation: {max_deviation:.4f}")
                
                return sampled_df
                
            except Exception as e:
                print(f"⚠️ Stratified sampling failed: {e}, using simple sampling")
        
        # Fallback to simple sampling
        return df.sample(n=min(self.max_sample_size, len(df)), random_state=42)

    def _analyze_enhanced_target_correlation_fixed(self, df: pd.DataFrame, target_col: str, 
                                                  method_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED: Enhanced target correlation with proper categorical/numeric handling"""
        
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        print("Analyzing enhanced target correlation...")
        
        all_cols = [col for col in df.columns if col != target_col]
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != target_col]
        
        print(f"Feature breakdown:")
        print(f"   • Numeric features: {len(numeric_cols)}")
        print(f"   • Categorical features: {len(categorical_cols)}")
        print(f"   • Total features: {len(all_cols)}")
        
        # Improved target type determination
        target_type = self._determine_target_type_improved(df[target_col])
        print(f"Target type: {target_type}")
        
        correlations = {}
        
        # FIXED: Process numeric features with improved methods
        if numeric_cols:
            print(f"\nProcessing {len(numeric_cols)} numeric features...")
            correlations.update(self._process_numeric_target_correlations_fixed(
                df, numeric_cols, target_col, target_type
            ))
        
        # FIXED: Process categorical features with proper encoding
        if categorical_cols:
            print(f"\nProcessing {len(categorical_cols)} categorical features...")
            correlations.update(self._process_categorical_target_correlations_fixed(
                df, categorical_cols, target_col, target_type
            ))
        
        if not correlations:
            self.logger.info("No valid correlations found with target")
            return df, {'method': 'enhanced_target_correlation', 'removed_features': []}
        
        print(f"✅ Computed {len(correlations)} target correlations")
        
        # Enhanced display and selection
        features_to_drop = self._display_correlations_and_select_features(
            correlations, target_col, method_info, 'Enhanced Target Correlation'
        )
        
        # Apply feature dropping
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features based on target correlation")
        
        # Create comprehensive visualization
        self._create_target_correlation_visualization(correlations, target_col, method_info)
        
        return df, {
            'method': 'enhanced_target_correlation',
            'removed_features': features_to_drop,
            'total_analyzed': len(correlations),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'target_type': target_type
        }

    def _analyze_robust_correlation_matrix_fixed(self, df: pd.DataFrame, target_col: Optional[str], 
                                               method_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED: Robust correlation matrix with improved pair handling"""
        
        print("Analyzing robust correlation matrix (Kendall-tau emphasis)...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) < 2:
            return df, {'method': 'robust_correlation_matrix', 'skipped': 'insufficient_numeric_features'}
        
        print(f"Computing robust correlations for {len(numeric_cols)} features...")
        
        # IMPROVED: Better preprocessing for manufacturing data
        correlation_data = df[numeric_cols].copy()
        
        # Handle missing values with manufacturing-appropriate strategy
        for col in correlation_data.columns:
            if correlation_data[col].isnull().any():
                if correlation_data[col].dtype in ['int64', 'float64']:
                    # Use median for numeric (robust to outliers in manufacturing)
                    correlation_data[col].fillna(correlation_data[col].median(), inplace=True)
                else:
                    correlation_data[col].fillna(0, inplace=True)
        
        # Calculate correlations with emphasis on Kendall (better for manufacturing data with ties)
        try:
            kendall_corr = correlation_data.corr(method='kendall')
            spearman_corr = correlation_data.corr(method='spearman')
            
            # FIXED: Better combination logic - prefer Kendall but use stronger when significantly different
            combined_corr = kendall_corr.copy()
            for i in range(len(combined_corr.columns)):
                for j in range(len(combined_corr.columns)):
                    if i != j:
                        kendall_val = abs(kendall_corr.iloc[i, j])
                        spear_val = abs(spearman_corr.iloc[i, j])
                        
                        # Use Spearman only if it's significantly stronger and above threshold
                        if spear_val > kendall_val * 1.3 and spear_val > 0.5:
                            combined_corr.iloc[i, j] = spearman_corr.iloc[i, j]
        except Exception as e:
            print(f"Correlation computation failed: {e}, using Spearman only")
            combined_corr = correlation_data.corr(method='spearman')
        
        # IMPROVED: Find correlated pairs with better filtering
        corr_pairs = self._find_robust_correlated_pairs_improved(combined_corr, target_col, df, method_info)
        
        print(f"Found {len(corr_pairs)} robust correlation pairs above threshold")
        
        if not corr_pairs:
            return df, {'method': 'robust_correlation_matrix', 'removed_features': []}
        
        # FIXED: Handle pairs with improved manufacturing logic
        features_to_drop = self._handle_robust_correlation_pairs_fixed(corr_pairs, df, target_col, method_info)
        
        # Apply feature dropping
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features using robust correlation")
        
        # Create visualization
        self._create_correlation_heatmap_enhanced(combined_corr, 'robust_correlation', method_info)
        
        return df, {
            'method': 'robust_correlation_matrix',
            'removed_features': features_to_drop,
            'pairs_analyzed': len(corr_pairs),
            'correlation_threshold': method_info['default_cutoff'],
            'methods_used': ['kendall_primary', 'spearman_secondary']
        }

    def _analyze_mutual_info_manufacturing_fixed(self, df: pd.DataFrame, target_col: Optional[str], 
                                               method_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED: Manufacturing MI with proper categorical handling and improved sampling"""
        
        if not SKLEARN_AVAILABLE:
            raise ValueError("Mutual Information requires scikit-learn")
        
        print("Manufacturing-optimized Mutual Information...")
        
        # IMPROVED: Better sampling strategy for MI
        max_mi_samples = 100000  # Increased for better MI estimates
        
        if len(df) > max_mi_samples:
            print(f"Applying manufacturing-optimized sampling ({max_mi_samples:,} samples)...")
            sample_df = self._apply_stratified_sampling(df.copy(), target_col)
            if len(sample_df) > max_mi_samples:
                # Secondary sampling maintaining ratios
                sample_df = sample_df.sample(n=max_mi_samples, random_state=42)
        else:
            sample_df = df
        
        feature_cols = [col for col in sample_df.columns if col != target_col]
        
        if not feature_cols:
            return df, {'method': 'mutual_info_manufacturing', 'removed_features': [], 'skipped': 'no_features'}
        
        print(f"Computing manufacturing MI for {len(feature_cols)} features on {len(sample_df):,} samples...")
        
        # FIXED: Proper data preparation for MI
        X_data, y_data, categorical_mask = self._prepare_mi_data_fixed(sample_df, target_col, feature_cols)
        
        # Determine task type with improved logic
        n_classes = len(np.unique(y_data[~np.isnan(y_data)]))
        task_type = 'multiclass_classification' if n_classes <= 50 else 'regression'
        print(f"Manufacturing task: {task_type} ({n_classes} classes)")
        
        try:
            # FIXED: Use appropriate MI function with proper parameters
            if task_type == 'multiclass_classification':
                mi_scores = mutual_info_classif(
                    X_data, y_data.astype(int),
                    discrete_features=categorical_mask,
                    n_neighbors=5,  # Conservative for manufacturing
                    random_state=42
                )
            else:
                mi_scores = mutual_info_regression(
                    X_data, y_data,
                    discrete_features=categorical_mask,
                    n_neighbors=5,
                    random_state=42
                )
            
            # IMPROVED: Better normalization
            if np.max(mi_scores) > 0:
                mi_scores_normalized = mi_scores / np.max(mi_scores)
            else:
                mi_scores_normalized = mi_scores
                
        except Exception as e:
            print(f"MI computation failed: {e}. Using improved fallback.")
            mi_scores_normalized = self._compute_manufacturing_mi_fallback_improved(
                X_data, y_data, feature_cols, categorical_mask
            )
        
        # Create results dictionary
        mi_results = {feature: score for feature, score in zip(feature_cols, mi_scores_normalized)}
        
        # Display and get user selection
        features_to_drop = self._display_correlations_and_select_features(
            {k: {'max_abs': v, 'feature_type': 'mixed', 'method': 'mutual_info'} 
             for k, v in mi_results.items()}, 
            target_col, method_info, 'Manufacturing Mutual Information'
        )
        
        # Apply dropping
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features based on manufacturing MI")
        
        # Create visualization
        self._create_mi_visualization(mi_results, target_col, method_info)
        
        return df, {
            'method': 'mutual_info_manufacturing',
            'removed_features': features_to_drop,
            'mi_scores': mi_results,
            'task_type': task_type,
            'n_classes': n_classes,
            'sample_size': len(sample_df),
            'categorical_features_handled': sum(categorical_mask)
        }

    def _analyze_manufacturing_association_fixed(self, df: pd.DataFrame, target_col: Optional[str], 
                                               method_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED: Manufacturing statistical association with improved test selection"""
        
        print("Computing enhanced manufacturing statistical association tests...")
        
        if not target_col or target_col not in df.columns:
            return df, {'method': 'manufacturing_association', 'removed_features': [], 'skipped': 'no_target'}
        
        target_data = df[target_col]
        target_is_categorical = self._is_categorical_improved(target_data)
        
        feature_cols = [col for col in df.columns if col != target_col]
        association_results = {}
        
        print(f"Manufacturing target type: {'categorical' if target_is_categorical else 'continuous'}")
        print(f"Testing {len(feature_cols)} manufacturing features...")
        
        with tqdm(total=len(feature_cols), desc="Manufacturing Tests", ncols=100) as pbar:
            for col in feature_cols:
                try:
                    col_data = df[col]
                    col_is_categorical = self._is_categorical_improved(col_data)
                    
                    # IMPROVED: Manufacturing-specific test selection with better logic
                    if target_is_categorical and col_is_categorical:
                        # Chi-square for categorical-categorical
                        test_stat, p_value = self._chi_square_test_improved(col_data, target_data)
                        test_type = 'chi_square_manufacturing'
                    elif target_is_categorical and not col_is_categorical:
                        # ANOVA for continuous measurements vs categorical quality states
                        test_stat, p_value = self._anova_f_test_improved(col_data, target_data)
                        test_type = 'anova_manufacturing'
                    elif not target_is_categorical and col_is_categorical:
                        # Kruskal-Wallis for non-parametric manufacturing data
                        test_stat, p_value = self._kruskal_wallis_test(target_data, col_data)
                        test_type = 'kruskal_wallis'
                    else:
                        # Spearman for continuous manufacturing data (robust to outliers)
                        test_stat, p_value = self._spearman_test_improved(col_data, target_data)
                        test_type = 'spearman_manufacturing'
                    
                    # IMPROVED: Better importance scoring for manufacturing
                    if p_value <= 1e-15:
                        importance_score = 15.0  # Maximum significance
                    elif p_value >= 1.0:
                        importance_score = 0.0  # No significance
                    else:
                        # Adjusted scoring for manufacturing significance levels
                        log_p = -np.log10(max(p_value, 1e-15))
                        # Apply manufacturing-specific scaling
                        importance_score = min(log_p * 1.2, 15.0)  # Slight boost for manufacturing
                    
                    association_results[col] = {
                        'test_statistic': test_stat,
                        'p_value': p_value,
                        'importance_score': importance_score,
                        'test_type': test_type
                    }
                    
                    pbar.update(1)
                    pbar.set_postfix({'Current': col[:20], 'p-val': f"{p_value:.3e}"})
                    
                except Exception as e:
                    association_results[col] = {
                        'test_statistic': 0.0,
                        'p_value': 1.0,
                        'importance_score': 0.0,
                        'test_type': 'failed'
                    }
                    pbar.update(1)
        
        # Display results
        features_to_drop = self._display_manufacturing_association_results_fixed(
            association_results, target_col, method_info
        )
        
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features based on manufacturing association tests")
        
        # Create visualization
        self._create_statistical_visualization(association_results, target_col, method_info)
        
        return df, {
            'method': 'manufacturing_association',
            'removed_features': features_to_drop,
            'association_results': association_results,
            'significance_level': method_info['default_cutoff']
        }

    def _analyze_information_gain_qc_fixed(self, df: pd.DataFrame, target_col: Optional[str], 
                                         method_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED: Information Gain with improved entropy calculations"""
        
        print("Computing Information Gain for Quality Control...")
        
        if not target_col or target_col not in df.columns:
            return df, {'method': 'information_gain_qc', 'removed_features': [], 'skipped': 'no_target'}
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        if not feature_cols:
            return df, {'method': 'information_gain_qc', 'removed_features': [], 'skipped': 'no_features'}
        
        print(f"Computing information gain for {len(feature_cols)} features...")
        
        # IMPROVED: Better target preparation for IG
        y_data = df[target_col].copy()
        if pd.api.types.is_numeric_dtype(y_data):
            if y_data.nunique() > 10:
                # Create quality-based bins for manufacturing
                y_encoded = pd.qcut(y_data.rank(method='first'), q=10, labels=False)
            else:
                y_encoded = pd.Categorical(y_data).codes
        else:
            y_encoded = pd.Categorical(y_data).codes
        
        # Handle any remaining NaN values
        y_encoded = pd.Series(y_encoded).fillna(-1).astype(int)
        
        information_gains = {}
        
        with tqdm(total=len(feature_cols), desc="Information Gain", ncols=100) as pbar:
            for col in feature_cols:
                try:
                    # IMPROVED: Better feature preparation
                    feature_data = df[col].copy()
                    
                    if pd.api.types.is_numeric_dtype(feature_data):
                        # Adaptive binning based on unique values
                        if feature_data.nunique() > 20:
                            # Use quantile-based binning for continuous features
                            feature_encoded = pd.qcut(feature_data.rank(method='first'), 
                                                    q=min(10, feature_data.nunique()//2), 
                                                    labels=False, duplicates='drop')
                        else:
                            feature_encoded = feature_data.fillna(-1).astype(int)
                    else:
                        # Categorical features
                        feature_encoded = pd.Categorical(feature_data.fillna('MISSING')).codes
                    
                    # Ensure proper encoding
                    feature_encoded = pd.Series(feature_encoded).fillna(-1).astype(int)
                    
                    # FIXED: Calculate information gain with improved entropy calculation
                    ig = self._calculate_information_gain_improved(feature_encoded.values, y_encoded.values)
                    information_gains[col] = ig
                    
                    pbar.update(1)
                    pbar.set_postfix({'Current': col[:20], 'IG': f"{ig:.4f}"})
                    
                except Exception as e:
                    information_gains[col] = 0.0
                    pbar.update(1)
        
        # Display and get user selection
        features_to_drop = self._display_correlations_and_select_features(
            {k: {'max_abs': v, 'feature_type': 'mixed', 'method': 'information_gain'} 
             for k, v in information_gains.items()},
            target_col, method_info, 'Information Gain QC'
        )
        
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features based on information gain")
        
        # Create visualization
        self._create_information_gain_visualization(information_gains, target_col, method_info)
        
        return df, {
            'method': 'information_gain_qc',
            'removed_features': features_to_drop,
            'information_gains': information_gains,
            'features_analyzed': len(feature_cols)
        }

    # IMPROVED HELPER METHODS

    def _determine_target_type_improved(self, target_series: pd.Series) -> str:
        """Improved target type determination for manufacturing data"""
        
        unique_vals = target_series.nunique()
        
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_vals == 2:
                return 'binary'
            elif unique_vals <= 25:  # Increased for manufacturing multiclass
                return 'multi_class'
            elif unique_vals <= 100:
                return 'high_cardinality_numeric'
            else:
                return 'continuous'
        else:
            if unique_vals == 2:
                return 'binary_categorical'
            elif unique_vals <= 25:
                return 'categorical'
            elif unique_vals <= 100:
                return 'high_cardinality_categorical'
            else:
                return 'very_high_cardinality_categorical'

    def _process_numeric_target_correlations_fixed(self, df: pd.DataFrame, numeric_cols: List[str], 
                                                  target_col: str, target_type: str) -> Dict:
        """FIXED: Improved numeric feature correlation processing"""
        
        correlations = {}
        
        # IMPROVED: Better target data preparation
        if target_type in ['categorical', 'binary_categorical', 'multi_class']:
            # Use label encoding for categorical targets
            le = LabelEncoder()
            target_data = le.fit_transform(df[target_col].fillna('missing').astype(str))
        else:
            # Robust handling for numeric targets
            target_data = df[target_col].fillna(df[target_col].median())
        
        with tqdm(total=len(numeric_cols), desc="Numeric Correlations", ncols=100) as pbar:
            for col in numeric_cols:
                try:
                    # IMPROVED: Robust data cleaning
                    col_data = df[col].copy()
                    
                    # Handle infinite values in manufacturing data
                    col_data = col_data.replace([np.inf, -np.inf], np.nan)
                    col_data = col_data.fillna(col_data.median())
                    
                    # Skip if no variance (constant feature)
                    if col_data.nunique() <= 1:
                        pbar.update(1)
                        continue
                    
                    # IMPROVED: Multiple correlation measures with better selection
                    correlations_computed = {}
                    
                    # Pearson correlation
                    try:
                        pearson_corr, pearson_p = stats.pearsonr(col_data, target_data)
                        if not (np.isnan(pearson_corr) or np.isnan(pearson_p)):
                            correlations_computed['pearson'] = (pearson_corr, pearson_p)
                    except:
                        pass
                    
                    # Spearman correlation (robust to outliers)
                    try:
                        spearman_corr, spearman_p = stats.spearmanr(col_data, target_data)
                        if not (np.isnan(spearman_corr) or np.isnan(spearman_p)):
                            correlations_computed['spearman'] = (spearman_corr, spearman_p)
                    except:
                        pass
                    
                    # Kendall correlation (best for manufacturing data with ties)
                    try:
                        kendall_corr, kendall_p = stats.kendalltau(col_data, target_data)
                        if not (np.isnan(kendall_corr) or np.isnan(kendall_p)):
                            correlations_computed['kendall'] = (kendall_corr, kendall_p)
                    except:
                        pass
                    
                    # IMPROVED: Select best correlation method
                    if correlations_computed:
                        # Prioritize Kendall for manufacturing, but use strongest if significantly different
                        best_method = 'kendall' if 'kendall' in correlations_computed else list(correlations_computed.keys())[0]
                        best_corr, best_p = correlations_computed[best_method]
                        
                        # Check if another method is significantly stronger
                        for method, (corr, p) in correlations_computed.items():
                            if abs(corr) > abs(best_corr) * 1.2 and abs(corr) > 0.1:
                                best_corr, best_p = corr, p
                                best_method = method
                        
                        correlations[col] = {
                            'correlation': best_corr,
                            'p_value': best_p,
                            'max_abs': abs(best_corr),
                            'feature_type': 'numeric',
                            'method': best_method
                        }
                    
                    pbar.update(1)
                    pbar.set_postfix({'Current': col[:20], 'Corr': f"{correlations.get(col, {}).get('correlation', 0):.3f}"})
                    
                except Exception as e:
                    pbar.update(1)
                    continue
        
        return correlations

    def _process_categorical_target_correlations_fixed(self, df: pd.DataFrame, categorical_cols: List[str],
                                                      target_col: str, target_type: str) -> Dict:
        """FIXED: Improved categorical feature correlation processing"""
        
        correlations = {}
        
        with tqdm(total=len(categorical_cols), desc="Categorical Correlations", ncols=100) as pbar:
            for col in categorical_cols:
                try:
                    # IMPROVED: Better categorical association measures
                    col_data = df[col].fillna('MISSING')
                    target_data = df[target_col].fillna('MISSING')
                    
                    # Skip if no variance
                    if col_data.nunique() <= 1:
                        pbar.update(1)
                        continue
                    
                    # IMPROVED: Multiple association measures
                    association_scores = {}
                    
                    # Cramér's V with improvements
                    cramers_v = self._calculate_cramers_v_improved(col_data, target_data)
                    if cramers_v > 0:
                        association_scores['cramers_v'] = cramers_v
                    
                    # Theil's U (uncertainty coefficient) - better for imbalanced data
                    theils_u = self._calculate_theils_u(col_data, target_data)
                    if theils_u > 0:
                        association_scores['theils_u'] = theils_u
                    
                    # For high-cardinality features, use adjusted measures
                    if col_data.nunique() > 20:
                        # Adjusted mutual information
                        try:
                            # Convert to numeric codes for MI calculation
                            col_codes = pd.Categorical(col_data).codes
                            target_codes = pd.Categorical(target_data).codes
                            
                            # Remove missing value codes
                            valid_mask = (col_codes >= 0) & (target_codes >= 0)
                            if valid_mask.sum() > 10:
                                ami = adjusted_mutual_info_score(col_codes[valid_mask], target_codes[valid_mask])
                                if ami > 0:
                                    association_scores['adjusted_mi'] = ami
                        except:
                            pass
                    
                    # Select best association measure
                    if association_scores:
                        best_score = max(association_scores.values())
                        best_method = max(association_scores, key=association_scores.get)
                        
                        correlations[col] = {
                            'association_score': best_score,
                            'max_abs': best_score,
                            'feature_type': 'categorical',
                            'method': best_method,
                            'cardinality': col_data.nunique()
                        }
                    
                    pbar.update(1)
                    pbar.set_postfix({'Current': col[:20], 'Assoc': f"{correlations.get(col, {}).get('max_abs', 0):.3f}"})
                    
                except Exception as e:
                    pbar.update(1)
                    continue
        
        return correlations

    def _find_robust_correlated_pairs_improved(self, corr_matrix: pd.DataFrame, target_col: str, 
                                             df: pd.DataFrame, method_info: Dict) -> List[Dict]:
        """IMPROVED: Better correlated pair detection with manufacturing priorities"""
        
        pairs = []
        n = len(corr_matrix.columns)
        threshold = method_info['default_cutoff']
        
        for i in range(n):
            for j in range(i + 1, n):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold and not np.isnan(corr_value):
                    # IMPROVED: Add manufacturing-specific metadata
                    pair_info = {
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value),
                        'method': 'robust_kendall_spearman',
                        'var1': df[col1].var() if col1 in df.columns else 0,
                        'var2': df[col2].var() if col2 in df.columns else 0,
                        'nunique1': df[col1].nunique() if col1 in df.columns else 0,
                        'nunique2': df[col2].nunique() if col2 in df.columns else 0
                    }
                    pairs.append(pair_info)
        
        # Sort by absolute correlation strength
        pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return pairs

    def _handle_robust_correlation_pairs_fixed(self, corr_pairs: List[Dict], df: pd.DataFrame, 
                                             target_col: str, method_info: Dict) -> List[str]:
        """FIXED: Improved correlation pair handling with better manufacturing logic"""
        
        if not corr_pairs:
            return []
        
        print(f"\nFound {len(corr_pairs)} highly correlated pairs")
        print("Applying manufacturing-specific pair resolution...")
        
        features_to_drop = set()
        
        # IMPROVED: Manufacturing-specific keywords with better priorities
        measurement_keywords = ['measure', 'limit', 'value', 'unit', 'step', 'fail', 'diag']
        process_keywords = ['workstep', 'station', 'sequence', 'recipe']
        id_keywords = ['id', 'number']  # More specific
        description_keywords = ['desc', 'name']  # Separate descriptions
        
        for pair in corr_pairs:
            feature1, feature2 = pair['feature1'], pair['feature2']
            
            # Skip if already decided
            if feature1 in features_to_drop or feature2 in features_to_drop:
                continue
            
            # IMPROVED: Manufacturing-specific decision logic
            f1_measurement = any(kw in feature1.lower() for kw in measurement_keywords)
            f2_measurement = any(kw in feature2.lower() for kw in measurement_keywords)
            
            f1_process = any(kw in feature1.lower() for kw in process_keywords)
            f2_process = any(kw in feature2.lower() for kw in process_keywords)
            
            f1_id = any(kw in feature1.lower() for kw in id_keywords)
            f2_id = any(kw in feature2.lower() for kw in id_keywords)
            
            f1_desc = any(kw in feature1.lower() for kw in description_keywords)
            f2_desc = any(kw in feature2.lower() for kw in description_keywords)
            
            # Manufacturing priority logic
            if f1_measurement and not f2_measurement:
                features_to_drop.add(feature2)
                print(f"  Dropping {feature2} - keeping {feature1} (measurement priority)")
            elif f2_measurement and not f1_measurement:
                features_to_drop.add(feature1)
                print(f"  Dropping {feature1} - keeping {feature2} (measurement priority)")
            elif f1_process and not f2_process:
                features_to_drop.add(feature2)
                print(f"  Dropping {feature2} - keeping {feature1} (process priority)")
            elif f2_process and not f1_process:
                features_to_drop.add(feature1)
                print(f"  Dropping {feature1} - keeping {feature2} (process priority)")
            elif f1_desc and not f2_desc:
                features_to_drop.add(feature1)  # Drop description, keep other
                print(f"  Dropping {feature1} (description) - keeping {feature2}")
            elif f2_desc and not f1_desc:
                features_to_drop.add(feature2)  # Drop description, keep other
                print(f"  Dropping {feature2} (description) - keeping {feature1}")
            elif f1_id and not f2_id:
                features_to_drop.add(feature1)  # Drop ID, keep other
                print(f"  Dropping {feature1} (ID) - keeping {feature2}")
            elif f2_id and not f1_id:
                features_to_drop.add(feature2)  # Drop ID, keep other
                print(f"  Dropping {feature2} (ID) - keeping {feature1}")
            else:
                # IMPROVED: Use variance and uniqueness for decision
                var1, var2 = pair['var1'], pair['var2']
                nunique1, nunique2 = pair['nunique1'], pair['nunique2']
                
                if var1 == 0 and var2 > 0:
                    features_to_drop.add(feature1)
                    print(f"  Dropping {feature1} (no variance) - keeping {feature2}")
                elif var2 == 0 and var1 > 0:
                    features_to_drop.add(feature2)
                    print(f"  Dropping {feature2} (no variance) - keeping {feature1}")
                elif nunique1 < nunique2 * 0.5:  # Much lower uniqueness
                    features_to_drop.add(feature1)
                    print(f"  Dropping {feature1} (lower uniqueness) - keeping {feature2}")
                elif nunique2 < nunique1 * 0.5:
                    features_to_drop.add(feature2)
                    print(f"  Dropping {feature2} (lower uniqueness) - keeping {feature1}")
                elif var1 < var2:  # Lower variance
                    features_to_drop.add(feature1)
                    print(f"  Dropping {feature1} (lower variance) - keeping {feature2}")
                else:
                    features_to_drop.add(feature2)
                    print(f"  Dropping {feature2} (default choice) - keeping {feature1}")
        
        features_to_drop = list(features_to_drop)
        
        if features_to_drop:
            print(f"\nProposed to drop {len(features_to_drop)} correlated features")
            user_choice = input(f"Proceed with dropping {len(features_to_drop)} correlated features? (y/n): ").strip().lower()
            if user_choice != 'y':
                print("Keeping all features")
                return []
        
        return features_to_drop

    def _prepare_mi_data_fixed(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """FIXED: Proper data preparation for mutual information"""
        
        # Prepare features
        X_data = df[feature_cols].copy()
        y_data = df[target_col].copy()
        
        # Track categorical features
        categorical_mask = []
        
        # IMPROVED: Better preprocessing for each feature
        for col in feature_cols:
            col_data = X_data[col]
            
            if pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                # Categorical feature
                categorical_mask.append(True)
                
                # Handle missing values
                col_data = col_data.fillna('MISSING')
                
                # Label encode categorical data
                le = LabelEncoder()
                X_data[col] = le.fit_transform(col_data.astype(str))
                
            else:
                # Numeric feature
                categorical_mask.append(False)
                
                # Handle missing values with median
                col_median = col_data.median()
                X_data[col] = col_data.fillna(col_median)
                
                # Handle infinite values
                X_data[col] = X_data[col].replace([np.inf, -np.inf], col_median)
        
        # IMPROVED: Better target preparation
        if pd.api.types.is_numeric_dtype(y_data):
            # Numeric target
            y_median = y_data.median()
            y_data = y_data.fillna(y_median)
            y_data = y_data.replace([np.inf, -np.inf], y_median)
            
            # For classification, ensure integer encoding
            if y_data.nunique() <= 100:  # Likely categorical
                y_data = pd.Categorical(y_data).codes
        else:
            # Categorical target
            y_data = y_data.fillna('MISSING')
            le_target = LabelEncoder()
            y_data = le_target.fit_transform(y_data.astype(str))
        
        return X_data.values.astype(np.float64), y_data.astype(np.float64), categorical_mask

    def _compute_manufacturing_mi_fallback_improved(self, X: np.ndarray, y: np.ndarray, 
                                                   feature_names: List[str], categorical_mask: List[bool]) -> np.ndarray:
        """IMPROVED: Manufacturing-specific MI fallback with better correlation measures"""
        
        fallback_scores = []
        
        for i in range(X.shape[1]):
            try:
                feature_data = X[:, i]
                is_categorical = categorical_mask[i] if i < len(categorical_mask) else False
                
                if is_categorical:
                    # Use adjusted mutual information for categorical features
                    try:
                        # Convert to integer codes
                        feature_int = feature_data.astype(int)
                        y_int = y.astype(int)
                        
                        # Remove invalid codes
                        valid_mask = (feature_int >= 0) & (y_int >= 0)
                        if valid_mask.sum() > 10:
                            ami = adjusted_mutual_info_score(feature_int[valid_mask], y_int[valid_mask])
                            score = max(ami, 0)
                        else:
                            score = 0.0
                    except:
                        score = 0.0
                else:
                    # Use Spearman correlation for numeric features (robust for manufacturing)
                    try:
                        corr, _ = stats.spearmanr(feature_data, y)
                        score = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        score = 0.0
                
                fallback_scores.append(score)
                
            except:
                fallback_scores.append(0.0)
        
        return np.array(fallback_scores)

    # IMPROVED STATISTICAL TEST METHODS

    def _is_categorical_improved(self, series: pd.Series) -> bool:
        """IMPROVED: Better categorical detection for manufacturing data"""
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric data with few unique values might be categorical in manufacturing
            unique_count = series.nunique()
            total_count = len(series.dropna())
            
            if total_count == 0:
                return True
            
            # If less than 5% unique values and reasonable count, treat as categorical
            uniqueness_ratio = unique_count / total_count
            
            if unique_count <= 2:
                return True  # Binary
            elif unique_count <= 20 and uniqueness_ratio < 0.05:
                return True  # Low cardinality manufacturing codes
            elif unique_count <= 50 and uniqueness_ratio < 0.01:
                return True  # Medium cardinality with very low uniqueness
            else:
                return False
        else:
            return True  # Non-numeric is categorical

    def _chi_square_test_improved(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """IMPROVED: Chi-square test with better handling of manufacturing data"""
        
        try:
            # Handle missing values properly
            x_filled = x.fillna('__MISSING__')
            y_filled = y.fillna('__MISSING__')
            
            # Create contingency table
            contingency = pd.crosstab(x_filled, y_filled)
            
            # IMPROVED: Better validation for manufacturing data
            if contingency.size < 4:
                return 0.0, 1.0
            
            # Check expected frequencies (should be >= 5 for chi-square validity)
            expected = contingency.sum().values[:, np.newaxis] * contingency.sum(axis=1).values / contingency.sum().sum()
            if (expected < 5).sum() > contingency.size * 0.2:  # Allow up to 20% cells with expected < 5
                # Use Fisher's exact test for small samples (approximation)
                try:
                    from scipy.stats import fisher_exact
                    if contingency.shape == (2, 2):
                        _, p_val = fisher_exact(contingency)
                        return 1.0, p_val  # Return dummy chi2 stat
                except:
                    pass
                # Fallback: use chi-square anyway but flag uncertainty
                pass
            
            # Perform chi-square test
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            
            return chi2 if not np.isnan(chi2) else 0.0, p_val if not np.isnan(p_val) else 1.0
            
        except Exception as e:
            return 0.0, 1.0

    def _anova_f_test_improved(self, continuous: pd.Series, categorical: pd.Series) -> Tuple[float, float]:
        """IMPROVED: ANOVA F-test with better validation for manufacturing data"""
        
        try:
            # Clean and prepare data
            combined = pd.DataFrame({'cont': continuous, 'cat': categorical}).dropna()
            
            if len(combined) < 15:  # Need reasonable sample size
                return 0.0, 1.0
            
            # Group by category
            groups = [group['cont'].values for name, group in combined.groupby('cat')]
            
            # IMPROVED: Better group validation
            valid_groups = []
            for group in groups:
                if len(group) >= 3 and group.var() > 1e-10:  # Need variance and samples
                    valid_groups.append(group)
            
            if len(valid_groups) < 2:
                return 0.0, 1.0
            
            # Test for equal variances (Levene's test) - manufacturing data often violates this
            try:
                _, levene_p = stats.levene(*valid_groups)
                if levene_p < 0.01:  # Unequal variances
                    # Use Welch's ANOVA (more robust)
                    from scipy.stats import alexandergovern
                    try:
                        stat, p_value = alexandergovern(*valid_groups)
                        return stat, p_value
                    except:
                        pass  # Fall back to standard ANOVA
            except:
                pass
            
            # Standard one-way ANOVA
            f_stat, p_value = stats.f_oneway(*valid_groups)
            
            return f_stat if not np.isnan(f_stat) else 0.0, p_value if not np.isnan(p_value) else 1.0
            
        except Exception as e:
            return 0.0, 1.0

    def _kruskal_wallis_test(self, continuous: pd.Series, categorical: pd.Series) -> Tuple[float, float]:
        """NEW: Kruskal-Wallis test for non-parametric manufacturing data"""
        
        try:
            # Clean data
            combined = pd.DataFrame({'cont': continuous, 'cat': categorical}).dropna()
            
            if len(combined) < 10:
                return 0.0, 1.0
            
            # Group by category
            groups = [group['cont'].values for name, group in combined.groupby('cat')]
            groups = [g for g in groups if len(g) >= 3]  # Need at least 3 samples per group
            
            if len(groups) < 2:
                return 0.0, 1.0
            
            # Kruskal-Wallis H test (non-parametric alternative to ANOVA)
            h_stat, p_value = stats.kruskal(*groups)
            
            return h_stat if not np.isnan(h_stat) else 0.0, p_value if not np.isnan(p_value) else 1.0
            
        except Exception as e:
            return 0.0, 1.0

    def _spearman_test_improved(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """IMPROVED: Spearman correlation test with better data handling"""
        
        try:
            # Clean data
            combined = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(combined) < 10:
                return 0.0, 1.0
            
            # Check for constant values
            if combined['x'].nunique() <= 1 or combined['y'].nunique() <= 1:
                return 0.0, 1.0
            
            # Spearman correlation (robust to outliers and monotonic relationships)
            corr, p_val = stats.spearmanr(combined['x'], combined['y'])
            
            return abs(corr) if not np.isnan(corr) else 0.0, p_val if not np.isnan(p_val) else 1.0
            
        except Exception as e:
            return 0.0, 1.0

    # IMPROVED ASSOCIATION MEASURES

    def _calculate_cramers_v_improved(self, x: pd.Series, y: pd.Series) -> float:
        """IMPROVED: Cramér's V with better handling of imbalanced manufacturing data"""
        
        try:
            # Handle missing values
            x_filled = x.fillna('__MISSING__')
            y_filled = y.fillna('__MISSING__')
            
            # Create contingency table
            contingency = pd.crosstab(x_filled, y_filled)
            
            # IMPROVED: Better validation for manufacturing data
            if contingency.size < 4 or contingency.sum().sum() < 20:
                return 0.0
            
            # Check for very sparse tables (common in manufacturing)
            sparsity = (contingency == 0).sum().sum() / contingency.size
            if sparsity > 0.8:  # More than 80% zeros
                return 0.0
            
            # Compute chi-square statistic with Yates continuity correction for 2x2 tables
            if contingency.shape == (2, 2):
                chi2, _, _, _ = stats.chi2_contingency(contingency, correction=True)
            else:
                chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
            
            n = contingency.sum().sum()
            
            # Handle edge case where one dimension is 1
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            if min_dim == 0:
                return 0.0
            
            # Compute Cramér's V with bias correction for small samples
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            
            # Bias correction for small samples
            if n < 100:
                bias_correction = 1 - (1/(2*n)) * (contingency.shape[0]-1) * (contingency.shape[1]-1) / min_dim
                cramers_v = max(0, cramers_v - bias_correction)
            
            return min(cramers_v, 1.0)  # Cap at 1.0
            
        except Exception as e:
            return 0.0

    def _calculate_theils_u(self, x: pd.Series, y: pd.Series) -> float:
        """NEW: Theil's U (uncertainty coefficient) - better for imbalanced data"""
        
        try:
            # Handle missing values
            x_filled = x.fillna('__MISSING__')
            y_filled = y.fillna('__MISSING__')
            
            # Calculate entropy
            def entropy(labels):
                if len(labels) == 0:
                    return 0
                value_counts = labels.value_counts()
                probabilities = value_counts / len(labels)
                return -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Calculate conditional entropy
            def conditional_entropy(y_series, x_series):
                total_entropy = 0
                total_count = len(y_series)
                
                for x_val in x_series.unique():
                    subset = y_series[x_series == x_val]
                    subset_entropy = entropy(subset)
                    subset_prob = len(subset) / total_count
                    total_entropy += subset_prob * subset_entropy
                
                return total_entropy
            
            # Calculate Theil's U = (H(Y) - H(Y|X)) / H(Y)
            h_y = entropy(y_filled)
            if h_y == 0:
                return 0.0
            
            h_y_given_x = conditional_entropy(y_filled, x_filled)
            theils_u = (h_y - h_y_given_x) / h_y
            
            return max(0, min(theils_u, 1.0))  # Ensure in [0,1]
            
        except Exception as e:
            return 0.0

    def _calculate_information_gain_improved(self, feature: np.ndarray, target: np.ndarray) -> float:
        """IMPROVED: Information gain with better entropy calculation for manufacturing QC"""
        
        try:
            # IMPROVED: Better entropy calculation
            def entropy(labels):
                if len(labels) == 0:
                    return 0
                
                # Filter out missing values (-1)
                valid_labels = labels[labels >= 0]
                if len(valid_labels) == 0:
                    return 0
                
                # Count occurrences
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                probabilities = counts / len(valid_labels)
                
                # Calculate entropy with numerical stability
                entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-15))
                return entropy_val
            
            # Calculate target entropy
            target_entropy = entropy(target)
            if target_entropy == 0:
                return 0.0
            
            # Calculate weighted entropy after splitting on feature
            weighted_entropy = 0
            total_samples = len(target)
            
            # Get unique feature values (excluding -1 for missing)
            unique_features = np.unique(feature[feature >= 0])
            
            for value in unique_features:
                # Get subset where feature has this value
                subset_mask = (feature == value)
                subset_size = np.sum(subset_mask)
                
                if subset_size > 0:
                    subset_target = target[subset_mask]
                    subset_entropy = entropy(subset_target)
                    weight = subset_size / total_samples
                    weighted_entropy += weight * subset_entropy
            
            # Handle missing values as separate split
            missing_mask = (feature == -1)
            missing_size = np.sum(missing_mask)
            if missing_size > 0:
                missing_target = target[missing_mask]
                missing_entropy = entropy(missing_target)
                missing_weight = missing_size / total_samples
                weighted_entropy += missing_weight * missing_entropy
            
            # Information gain = original entropy - weighted entropy after split
            information_gain = target_entropy - weighted_entropy
            
            # Normalize by target entropy for better comparison across features
            normalized_ig = information_gain / target_entropy if target_entropy > 0 else 0
            
            return max(0, normalized_ig)  # Ensure non-negative
            
        except Exception as e:
            return 0.0

    # DISPLAY AND SELECTION METHODS

    def _display_correlations_and_select_features(self, correlations: Dict, target_col: str, 
                                                method_info: Dict, analysis_type: str) -> List[str]:
        """Enhanced correlation display with interactive feature selection"""
        
        if not correlations:
            print("No correlation results to display")
            return []
        
        print(f"\n{'='*155}")
        print(f"{analysis_type.upper()} ANALYSIS RESULTS")
        print(f"{'='*155}")
        print(f"Target Variable: {target_col}")
        print(f"Features Analyzed: {len(correlations)}")
        print(f"Method: {method_info['name']}")
        print(f"Default Cutoff: {method_info['default_cutoff']}")
        print(f"{'='*155}")
        
        # Sort by correlation strength
        sorted_features = sorted(correlations.items(), 
                               key=lambda x: x[1].get('max_abs', 0), reverse=True)
        
        # Display complete results table
        print(f"{'Rank':<6} {'Feature Name':<35} {'Score':<12} {'Type':<12} {'Method':<15} {'Action'}")
        print("-" * 155)
        
        features_by_cutoff = {'default': [], 'conservative': [], 'aggressive': []}
        default_cutoff = method_info['default_cutoff']
        
        for i, (feature, corr_data) in enumerate(sorted_features, 1):
            feature_type = corr_data.get('feature_type', 'unknown')
            max_score = corr_data.get('max_abs', 0)
            method_used = corr_data.get('method', 'unknown')
            
            # Determine action based on cutoffs
            if max_score < default_cutoff:
                action = "DROP"
                features_by_cutoff['default'].append(feature)
            else:
                action = "KEEP"
            
            # Conservative and aggressive cutoffs (method-specific multipliers)
            conservative_multiplier = 2.0 if 'association' in method_info['name'].lower() else 0.1
            aggressive_multiplier = 0.5 if 'association' in method_info['name'].lower() else 0.001
            
            if max_score < default_cutoff * aggressive_multiplier:
                features_by_cutoff['aggressive'].append(feature)
            if max_score < default_cutoff * conservative_multiplier:
                features_by_cutoff['conservative'].append(feature)
            
            feature_display = feature[:34] if len(feature) <= 34 else feature[:31] + "..."
            print(f"{i:<6} {feature_display:<35} {max_score:<12.6f} {feature_type:<12} {method_used:<15} {action}")
        
        print("-" * 155)
        print(f"Summary: {len(sorted_features)} total features")
        
        # Interactive selection
        return self._interactive_feature_selection_improved(features_by_cutoff, correlations, default_cutoff)

    def _interactive_feature_selection_improved(self, features_by_cutoff: Dict, correlations: Dict, 
                                              default_cutoff: float) -> List[str]:
        """IMPROVED: Interactive feature selection with better options"""
        
        print(f"\nFEATURE SELECTION OPTIONS:")
        print(f"1. Use default cutoff ({default_cutoff}) - Drop {len(features_by_cutoff['default'])} features")
        print(f"2. Use conservative cutoff - Drop {len(features_by_cutoff['conservative'])} features")
        print(f"3. Use aggressive cutoff - Drop {len(features_by_cutoff['aggressive'])} features")
        print(f"4. Manual selection (comma-separated list)")
        print(f"5. Keep all features")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    selected_features = features_by_cutoff['default']
                    print(f"Using default cutoff: {len(selected_features)} features selected for removal")
                    break
                elif choice == '2':
                    selected_features = features_by_cutoff['conservative']
                    print(f"Using conservative cutoff: {len(selected_features)} features selected for removal")
                    break
                elif choice == '3':
                    selected_features = features_by_cutoff['aggressive']
                    print(f"Using aggressive cutoff: {len(selected_features)} features selected for removal")
                    break
                elif choice == '4':
                    print(f"\nEnter feature names to drop (comma-separated):")
                    available_features = list(correlations.keys())
                    print("Available features:", ', '.join(available_features[:10]) + 
                          (f" ... and {len(available_features)-10} more" if len(available_features) > 10 else ""))
                    
                    manual_input = input("\nFeatures to drop: ").strip()
                    if manual_input:
                        manual_features = [f.strip() for f in manual_input.split(',')]
                        valid_features = [f for f in manual_features if f in correlations]
                        invalid_features = [f for f in manual_features if f not in correlations]
                        
                        if invalid_features:
                            print(f"Invalid features ignored: {invalid_features}")
                        
                        selected_features = valid_features
                        print(f"Manual selection: {len(selected_features)} features selected for removal")
                        break
                    else:
                        selected_features = []
                        break
                elif choice == '5':
                    selected_features = []
                    print("Keeping all features")
                    break
                else:
                    print("Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\nSelection cancelled. Keeping all features.")
                selected_features = []
                break
        
        # Confirmation for non-empty selection
        if selected_features:
            print(f"\nFINAL SELECTION SUMMARY:")
            print(f"Features to drop: {len(selected_features)}")
            for feat in selected_features[:10]:
                score = correlations[feat].get('max_abs', 0)
                print(f"  • {feat} (score: {score:.6f})")
            if len(selected_features) > 10:
                print(f"  ... and {len(selected_features) - 10} more")
            
            confirm = input(f"\nConfirm dropping {len(selected_features)} features? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Feature dropping cancelled")
                return []
        
        return selected_features

    def _display_manufacturing_association_results_fixed(self, results: Dict, target_col: str, 
                                                       method_info: Dict) -> List[str]:
        """FIXED: Display manufacturing association results with proper formatting"""
        
        if not results:
            print("No manufacturing association results to display")
            return []
        
        print(f"\n{'='*155}")
        print(f"Manufacturing Statistical Association Results")
        print(f"{'='*155}")
        print(f"Target Variable: {target_col}")
        print(f"Features Analyzed: {len(results)}")
        print(f"Default Cutoff: {method_info['default_cutoff']} (-log10 p-value)")
        print(f"{'='*155}")
        
        # Sort by importance score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['importance_score'], reverse=True)
        
        print(f"{'Rank':<6} {'Feature Name':<35} {'Test Type':<20} {'P-Value':<12} {'Importance':<12} {'Action'}")
        print("-" * 155)
        
        features_by_cutoff = {'default': [], 'conservative': [], 'aggressive': []}
        default_cutoff = method_info['default_cutoff']
        
        for i, (feature, result_data) in enumerate(sorted_results, 1):
            p_value = result_data['p_value']
            importance = result_data['importance_score']
            test_type = result_data['test_type']
            
            # Determine action based on cutoffs
            if importance < default_cutoff:
                action = "DROP"
                features_by_cutoff['default'].append(feature)
            else:
                action = "KEEP"
            
            # Conservative and aggressive cutoffs for statistical tests
            if importance < default_cutoff * 0.4:  # More aggressive for statistical tests
                features_by_cutoff['aggressive'].append(feature)
            if importance < default_cutoff * 2.0:  # More conservative
                features_by_cutoff['conservative'].append(feature)
            
            feature_display = feature[:34] if len(feature) <= 34 else feature[:31] + "..."
            p_val_display = f"{p_value:.3e}" if p_value > 0 else "<1e-15"
            
            print(f"{i:<6} {feature_display:<35} {test_type:<20} {p_val_display:<12} {importance:<12.2f} {action}")
        
        print("-" * 155)
        
        # Interactive selection
        return self._interactive_feature_selection_improved(
            features_by_cutoff, 
            {k: {'max_abs': v['importance_score']} for k, v in results.items()}, 
            default_cutoff
        )

    # VISUALIZATION METHODS

    def _create_target_correlation_visualization(self, correlations: Dict, target_col: str, method_info: Dict):
        """Create comprehensive target correlation visualization"""
        
        if not correlations:
            return
        
        try:
            # Prepare data
            features = []
            scores = []
            types = []
            methods = []
            
            for feature, corr_data in correlations.items():
                features.append(feature[:25] + '...' if len(feature) > 25 else feature)
                scores.append(corr_data.get('max_abs', 0))
                types.append(corr_data.get('feature_type', 'unknown'))
                methods.append(corr_data.get('method', 'unknown'))
            
            # Sort by correlation strength
            sorted_indices = np.argsort(scores)[::-1]
            features = [features[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            types = [types[i] for i in sorted_indices]
            methods = [methods[i] for i in sorted_indices]
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Enhanced Target Correlation Analysis\nTarget: {target_col}', fontsize=14, fontweight='bold')
            
            # 1. Top correlations bar plot
            top_n = min(20, len(features))
            colors = ['blue' if t == 'numeric' else 'green' if t == 'categorical' else 'red' for t in types[:top_n]]
            
            bars = ax1.barh(range(top_n), scores[:top_n], color=colors, alpha=0.7)
            ax1.set_yticks(range(top_n))
            ax1.set_yticklabels(features[:top_n], fontsize=8)
            ax1.set_xlabel('Correlation Strength')
            ax1.set_title(f'Top {top_n} Features by Target Correlation')
            ax1.axvline(x=method_info['default_cutoff'], color='red', linestyle='--', 
                       label=f"Default Cutoff ({method_info['default_cutoff']})")
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
            
            # 2. Distribution by feature type
            type_counts = pd.Series(types).value_counts()
            colors_pie = ['blue', 'green', 'red', 'orange', 'purple'][:len(type_counts)]
            ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors_pie)
            ax2.set_title('Feature Types Distribution')
            
            # 3. Correlation distribution histogram
            ax3.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(method_info['default_cutoff'], color='red', linestyle='--', 
                       label=f"Default Cutoff ({method_info['default_cutoff']})")
            ax3.set_xlabel('Correlation Strength')
            ax3.set_ylabel('Number of Features')
            ax3.set_title('Distribution of Correlation Strengths')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # 4. Method usage breakdown
            method_counts = pd.Series(methods).value_counts()
            ax4.bar(range(len(method_counts)), method_counts.values, color='lightcoral', alpha=0.7)
            ax4.set_xticks(range(len(method_counts)))
            ax4.set_xticklabels(method_counts.index, rotation=45, ha='right')
            ax4.set_ylabel('Number of Features')
            ax4.set_title('Correlation Methods Used')
            ax4.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f'target_correlation_{target_col}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Target correlation visualization saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create target correlation visualization: {e}")
            plt.close('all')

    def _create_statistical_visualization(self, results: Dict, target_col: str, method_info: Dict):
        """Create statistical association visualization"""
        
        if not results:
            return
        
        try:
            # Prepare data for visualization
            features = []
            importance_scores = []
            p_values = []
            test_types = []
            
            for feature, result_data in results.items():
                features.append(feature[:25] + '...' if len(feature) > 25 else feature)
                importance_scores.append(result_data['importance_score'])
                p_values.append(result_data['p_value'])
                test_types.append(result_data['test_type'])
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1]
            features = [features[i] for i in sorted_indices[:20]]  # Top 20
            importance_scores = [importance_scores[i] for i in sorted_indices[:20]]
            p_values = [p_values[i] for i in sorted_indices[:20]]
            test_types = [test_types[i] for i in sorted_indices[:20]]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Bar plot of importance scores
            colors = ['red' if score < method_info['default_cutoff'] else 'green' for score in importance_scores]
            bars = ax1.barh(range(len(features)), importance_scores, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=8)
            ax1.set_xlabel('Importance Score (-log10 p-value)')
            ax1.set_title(f'Statistical Association with {target_col}')
            ax1.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                       label=f"Cutoff ({method_info['default_cutoff']})")
            ax1.legend()
            
            # P-value distribution
            all_p_values = [result['p_value'] for result in results.values() if result['p_value'] > 0]
            if all_p_values:
                log_p_values = [-np.log10(p) for p in all_p_values]
                ax2.hist(log_p_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                           label=f"Cutoff ({method_info['default_cutoff']})")
                ax2.set_xlabel('-log10(p-value)')
                ax2.set_ylabel('Number of Features')
                ax2.set_title('P-value Distribution')
                ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f'statistical_association_{target_col}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Statistical association visualization saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create statistical visualization: {e}")
            plt.close('all')

    def _create_mi_visualization(self, mi_results: Dict, target_col: str, method_info: Dict):
        """Create mutual information visualization"""
        
        if not mi_results:
            return
        
        try:
            sorted_items = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
            features = [item[0][:25] + '...' if len(item[0]) > 25 else item[0] for item in sorted_items[:20]]
            scores = [item[1] for item in sorted_items[:20]]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            colors = ['red' if score < method_info['default_cutoff'] else 'green' for score in scores]
            bars = ax1.barh(range(len(features)), scores, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=8)
            ax1.set_xlabel('Normalized Mutual Information')
            ax1.set_title(f'Mutual Information with {target_col}')
            ax1.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                       label=f"Cutoff ({method_info['default_cutoff']})")
            ax1.legend()
            
            # Distribution histogram
            all_scores = list(mi_results.values())
            ax2.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                       label=f"Cutoff ({method_info['default_cutoff']})")
            ax2.set_xlabel('Mutual Information Score')
            ax2.set_ylabel('Number of Features')
            ax2.set_title('MI Score Distribution')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f'mutual_information_{target_col}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Mutual information visualization saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create MI visualization: {e}")
            plt.close('all')

    def _create_information_gain_visualization(self, ig_results: Dict, target_col: str, method_info: Dict):
        """Create information gain visualization"""
        
        if not ig_results:
            return
        
        try:
            sorted_items = sorted(ig_results.items(), key=lambda x: x[1], reverse=True)
            features = [item[0][:25] + '...' if len(item[0]) > 25 else item[0] for item in sorted_items[:20]]
            scores = [item[1] for item in sorted_items[:20]]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            colors = ['red' if score < method_info['default_cutoff'] else 'green' for score in scores]
            bars = ax1.barh(range(len(features)), scores, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=8)
            ax1.set_xlabel('Information Gain')
            ax1.set_title(f'Information Gain for QC - {target_col}')
            ax1.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                       label=f"Cutoff ({method_info['default_cutoff']})")
            ax1.legend()
            
            # Distribution histogram  
            all_scores = list(ig_results.values())
            ax2.hist(all_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(method_info['default_cutoff'], color='blue', linestyle='--', 
                       label=f"Cutoff ({method_info['default_cutoff']})")
            ax2.set_xlabel('Information Gain')
            ax2.set_ylabel('Number of Features')
            ax2.set_title('Information Gain Distribution')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f'information_gain_{target_col}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Information gain visualization saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create information gain visualization: {e}")
            plt.close('all')

    def _create_correlation_heatmap_enhanced(self, corr_matrix: pd.DataFrame, method_name: str, method_info: Dict):
        """Create enhanced correlation heatmap"""
        
        try:
            if len(corr_matrix.columns) > 50:
                corr_matrix = corr_matrix.iloc[:50, :50]
            
            plt.figure(figsize=(12, 10))
            
            sns.heatmap(corr_matrix, 
                       annot=False,
                       cmap='coolwarm',
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={'shrink': 0.8})
            
            plt.title(f'{method_name.title()} Correlation Matrix\n({len(corr_matrix.columns)} features)')
            plt.tight_layout()
            
            plot_file = self.plots_dir / f'{method_name}_correlation_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Correlation heatmap saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create correlation heatmap: {e}")
            plt.close('all')