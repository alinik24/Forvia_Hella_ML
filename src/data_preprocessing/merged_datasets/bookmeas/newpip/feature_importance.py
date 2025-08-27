#!/usr/bin/env python3
"""
CLEAN Feature Importance Analyzer - Fixed Version with Proper Error Handling
Addresses dropping bugs, ratio input, visualization errors, and model performance
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
import warnings
import time
import gc
from scipy.stats import chi2_contingency, f_oneway, spearmanr
from itertools import combinations
import signal
from contextlib import contextmanager

class FixedFeatureImportanceAnalyzer:
    """Fixed feature importance analyzer with robust error handling"""
    
    def __init__(self, logger: logging.Logger, plots_dir: Path, balance_handler=None):
        self.logger = logger
        self.plots_dir = plots_dir
        self.balance_handler = balance_handler
        
        # Enhanced methods with better tuning
        self.importance_methods = {
            'permutation_robust': {
                'name': 'Permutation Importance (Production)',
                'description': 'Model-agnostic with stability assessment and timeout protection',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Balanced scoring with robust base model',
                'default_cutoff': 0.001,
                'computational_cost': 'Medium',
                'symbol': 'Target',
                'research_basis': 'Breiman (2001), Altmann et al. (2010)',
                'timeout_minutes': 5
            },
            'gradient_boosting_importance': {
                'name': 'Gradient Boosting Importance',
                'description': 'Tree boosting with built-in feature interaction detection',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Class balancing with subsample weighting',
                'default_cutoff': 0.005,
                'computational_cost': 'Medium',
                'symbol': 'Trees',
                'research_basis': 'Friedman (2001), Chen & Guestrin (2016)',
                'timeout_minutes': 3
            },
            'univariate_robust': {
                'name': 'Robust Univariate Selection',
                'description': 'Multiple statistical tests with FDR correction and effect sizes',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Distribution-free tests',
                'default_cutoff': 0.01,
                'computational_cost': 'Very Low',
                'symbol': 'Stats',
                'research_basis': 'Benjamini & Hochberg (1995), Cohen (1988)',
                'timeout_minutes': 1
            },
            'correlation_network': {
                'name': 'Correlation Network Analysis',
                'description': 'Network-based feature selection with centrality measures',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Robust correlation measures',
                'default_cutoff': 0.1,
                'computational_cost': 'Low',
                'symbol': 'Network',
                'research_basis': 'Yu & Liu (2003), Song et al. (2012)',
                'timeout_minutes': 2
            },
            'elastic_net_selection': {
                'name': 'Elastic Net Feature Selection',
                'description': 'L1+L2 regularization with cross-validation and stability',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Balanced class weights with regularization',
                'default_cutoff': 0.001,
                'computational_cost': 'Medium',
                'symbol': 'Elastic',
                'research_basis': 'Zou & Hastie (2005), Simon et al. (2013)',
                'timeout_minutes': 4
            },
            'information_gain_ratio': {
                'name': 'Information Gain Ratio',
                'description': 'Enhanced information gain with bias correction',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Entropy-based, naturally handles imbalance',
                'default_cutoff': 0.01,
                'computational_cost': 'Low',
                'symbol': 'Info',
                'research_basis': 'Quinlan (1986), Kononenko (1995)',
                'timeout_minutes': 1
            },
            'logistic_regression_optimized': {
                'name': 'Optimized Logistic Regression',
                'description': 'Fine-tuned regularized logistic regression with stability',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Advanced class weighting with SAGA solver',
                'default_cutoff': 0.001,
                'computational_cost': 'Medium',
                'symbol': 'Linear',
                'research_basis': 'Cox (1958), Hastie et al. (2009)',
                'timeout_minutes': 3
            },
            'consensus_ranking': {
                'name': 'Multi-Method Consensus',
                'description': 'Robust consensus from fastest, most reliable methods',
                'multiclass_support': 'Excellent',
                'imbalance_handling': 'Combines multiple balanced approaches',
                'default_cutoff': 0.3,
                'computational_cost': 'Medium',
                'symbol': 'Consensus',
                'research_basis': 'Saeys et al. (2007), Haury et al. (2011)',
                'timeout_minutes': 8
            }
        }
        
        # Fixed sampling strategies with percentage-based ratios
        self.sampling_strategies = {
            'enhanced_minority_boost': {
                'name': 'Enhanced Minority Boost',
                'description': 'Oversample minorities with noise injection, cap majority',
                'symbol': 'Balance',
                'best_for': 'Extreme imbalance (>1000:1)',
                'default_size': 100000
            },
            'balanced_stratified': {
                'name': 'Balanced Stratified',
                'description': 'Equal samples per class with smart oversampling',
                'symbol': 'Equal',
                'best_for': 'Moderate imbalance (<100:1)',
                'default_size': 75000
            },
            'proportional_capped': {
                'name': 'Proportional Capped',
                'description': 'Maintain ratios but cap extremes',
                'symbol': 'Proportional',
                'best_for': 'Preserve original distribution',
                'default_size': 150000
            },
            'custom_ratios': {
                'name': 'Custom Percentage Ratios',
                'description': 'User-defined percentage ratios (must sum to 100%)',
                'symbol': 'Custom',
                'best_for': 'Specific requirements',
                'default_size': 100000
            }
        }
        
        # Optimized settings for dataset characteristics
        self.max_sample_size_default = 150000  # Increased for better representation
        self.n_jobs = -1
        
    def analyze_enhanced_importance(self, df: pd.DataFrame, target_col: str, 
                                method: str) -> Tuple[pd.DataFrame, Dict]:
        """Fixed feature importance analysis with proper error handling"""
        
        self.logger.info(f"Starting CLEAN feature importance analysis: {method}")
        
        if method not in self.importance_methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.importance_methods.keys())}")
        
        method_info = self.importance_methods[method]
        
        # Display method info
        print(f"\n[{method_info['symbol']}] {method_info['name']}")
        print(f"Description: {method_info['description']}")
        print(f"Multiclass Support: {method_info['multiclass_support']}")
        print(f"Imbalance Handling: {method_info['imbalance_handling']}")
        print(f"Research Basis: {method_info['research_basis']}")
        print(f"Timeout: {method_info['timeout_minutes']} minutes")
        print(f"Default Cutoff: {method_info['default_cutoff']}")
        
        # Interactive sampling strategy selection
        if len(df) > 50000:
            df_sampled, sampling_info = self._interactive_sampling_selection(df, target_col)
        else:
            print(f"Dataset size ({len(df):,} rows) - using full dataset")
            df_sampled = df.copy()
            sampling_info = {'strategy': 'full_dataset', 'original_size': len(df)}
        
        print("Preparing data with production-grade preprocessing...")
        X, y, feature_names = self._prepare_data_robust(df_sampled, target_col)
        
        try:
            print(f"Applying {method} analysis with timeout protection...")
            timeout_seconds = method_info['timeout_minutes'] * 60
            
            with self._timeout_context(timeout_seconds):
                start_time = time.time()
                
                # Route to appropriate method with improved performance
                if method == 'permutation_robust':
                    importance_scores, method_details = self._analyze_permutation_tuned(X, y, feature_names)
                elif method == 'gradient_boosting_importance':
                    importance_scores, method_details = self._analyze_gradient_boosting_optimized(X, y, feature_names)
                elif method == 'univariate_robust':
                    importance_scores, method_details = self._analyze_univariate_enhanced(X, y, feature_names)
                elif method == 'correlation_network':
                    importance_scores, method_details = self._analyze_correlation_network_robust(X, y, feature_names)
                elif method == 'elastic_net_selection':
                    importance_scores, method_details = self._analyze_elastic_net_fixed(X, y, feature_names)
                elif method == 'information_gain_ratio':
                    importance_scores, method_details = self._analyze_information_gain_enhanced(X, y, feature_names)
                elif method == 'logistic_regression_optimized':
                    importance_scores, method_details = self._analyze_logistic_regression_tuned(X, y, feature_names)
                elif method == 'consensus_ranking':
                    importance_scores, method_details = self._analyze_consensus_ranking_robust(X, y, feature_names)
                else:
                    raise ValueError(f"Method {method} not implemented")
                
                execution_time = time.time() - start_time
                method_details['execution_time'] = execution_time
                method_details['sampling_info'] = sampling_info
                
        except TimeoutError:
            print(f"Method {method} timed out after {method_info['timeout_minutes']} minutes")
            self.logger.warning(f"Method {method} timed out")
            raise
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")
            self.logger.error(f"Method {method} failed: {str(e)}")
            raise
        
        # Fixed display and selection with proper return handling
        features_to_drop = self._display_and_select_fixed(
            importance_scores, method_details, method, target_col, method_info
        )
        
        # FIXED: Ensure features_to_drop is always a proper list
        if features_to_drop is None:
            features_to_drop = []
        elif not isinstance(features_to_drop, list):
            features_to_drop = list(features_to_drop) if features_to_drop else []
        
        # Apply selection to original dataframe with robust error handling
        original_shape = df.shape
        if features_to_drop:
            # Ensure all features to drop exist in the dataframe and are not the target
            valid_features_to_drop = [
                f for f in features_to_drop 
                if isinstance(f, str) and f in df.columns and f != target_col
            ]
            
            if valid_features_to_drop:
                try:
                    df_result = df.drop(columns=valid_features_to_drop)
                    print(f"Dropped {len(valid_features_to_drop)} features: {original_shape} -> {df_result.shape}")
                    self.logger.info(f"Dropped {len(valid_features_to_drop)} features using {method}")
                except Exception as e:
                    print(f"Error dropping features: {e}")
                    self.logger.error(f"Error dropping features: {e}")
                    df_result = df.copy()
                    valid_features_to_drop = []
            else:
                print("No valid features to drop")
                df_result = df.copy()
                valid_features_to_drop = []
        else:
            print("No features selected for dropping")
            df_result = df.copy()
            valid_features_to_drop = []
        
        # Create visualization with proper error handling
        try:
            self._create_fixed_visualization(importance_scores, method, target_col, method_info)
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
            print(f"Visualization creation failed: {e}")
        
        # FIXED: Return properly structured result dictionary
        result_dict = {
            'method': str(method),
            'removed_features': list(valid_features_to_drop),  # Ensure list
            'method_details': dict(method_details) if method_details else {},
            'importance_scores': dict(importance_scores) if importance_scores else {},
            'total_features_analyzed': len(importance_scores) if importance_scores else 0,
            'execution_time': float(execution_time),
            'sampling_strategy': dict(sampling_info) if sampling_info else {}
        }
        
        return df_result, result_dict
    
    def _interactive_sampling_selection(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict]:
        """Fixed interactive sampling with proper percentage-based ratios"""
        
        target_counts = df[target_col].value_counts().sort_index()
        majority_class = target_counts.idxmax()
        minority_classes = target_counts.drop(majority_class)
        imbalance_ratio = target_counts[majority_class] / minority_classes.max()
        
        print(f"\nDATASET SAMPLING STRATEGY SELECTION")
        print(f"=" * 80)
        print(f"Large dataset detected: {len(df):,} rows")
        print(f"Original distribution: {dict(target_counts)}")
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"Majority class: {majority_class} ({target_counts[majority_class]:,} samples)")
        for cls in minority_classes.index:
            print(f"Minority class: {cls} ({target_counts[cls]:,} samples)")
        
        # Recommend strategy
        if imbalance_ratio > 1000:
            recommended = 'enhanced_minority_boost'
            print(f"\nExtreme imbalance detected (>{imbalance_ratio:.0f}:1)")
        elif imbalance_ratio > 100:
            recommended = 'balanced_stratified' 
            print(f"\nHigh imbalance detected (>{imbalance_ratio:.0f}:1)")
        else:
            recommended = 'proportional_capped'
            print(f"\nModerate imbalance detected ({imbalance_ratio:.0f}:1)")
        
        print(f"Recommended strategy: {recommended}")
        
        print(f"\nAVAILABLE SAMPLING STRATEGIES:")
        print("-" * 80)
        
        for i, (strategy_key, strategy_info) in enumerate(self.sampling_strategies.items(), 1):
            marker = ">>> RECOMMENDED" if strategy_key == recommended else "   "
            print(f"{marker} {i}. [{strategy_info['symbol']}] {strategy_info['name']}")
            print(f"     Description: {strategy_info['description']}")
            print(f"     Best for: {strategy_info['best_for']}")
            print(f"     Default size: {strategy_info['default_size']:,}")
            print()
        
        while True:
            try:
                choice = input("Select sampling strategy (1-4): ").strip()
                
                if choice in ['1', '2', '3', '4']:
                    strategy_keys = list(self.sampling_strategies.keys())
                    selected_strategy = strategy_keys[int(choice) - 1]
                    strategy_info = self.sampling_strategies[selected_strategy]
                    
                    print(f"\nSelected: [{strategy_info['symbol']}] {strategy_info['name']}")
                    
                    # Get sample size
                    default_size = strategy_info['default_size']
                    max_reasonable = min(len(df), 200000)
                    
                    size_input = input(f"Sample size (default {default_size:,}, max {max_reasonable:,}): ").strip()
                    
                    if size_input:
                        try:
                            sample_size = int(size_input)
                            sample_size = min(max(sample_size, 10000), max_reasonable)
                        except ValueError:
                            sample_size = default_size
                    else:
                        sample_size = default_size
                    
                    print(f"Using sample size: {sample_size:,}")
                    
                    # Apply selected strategy
                    sampled_df = self._apply_sampling_strategy(
                        df, target_col, selected_strategy, sample_size
                    )
                    
                    sampling_info = {
                        'strategy': selected_strategy,
                        'strategy_name': strategy_info['name'],
                        'original_size': len(df),
                        'sampled_size': len(sampled_df),
                        'target_size': sample_size,
                        'original_distribution': dict(target_counts),
                        'final_distribution': dict(sampled_df[target_col].value_counts().sort_index()),
                        'imbalance_ratio_original': imbalance_ratio,
                        'imbalance_ratio_final': self._calculate_imbalance_ratio(sampled_df, target_col)
                    }
                    
                    return sampled_df, sampling_info
                    
                else:
                    print("Invalid choice. Please select 1-4.")
                    
            except KeyboardInterrupt:
                print("\nSelection cancelled. Using recommended strategy.")
                sampled_df = self._apply_sampling_strategy(
                    df, target_col, recommended, self.sampling_strategies[recommended]['default_size']
                )
                sampling_info = {
                    'strategy': recommended,
                    'strategy_name': self.sampling_strategies[recommended]['name'],
                    'original_size': len(df),
                    'sampled_size': len(sampled_df),
                    'cancelled': True
                }
                return sampled_df, sampling_info
    
    def _apply_sampling_strategy(self, df: pd.DataFrame, target_col: str, 
                               strategy: str, sample_size: int) -> pd.DataFrame:
        """Apply selected sampling strategy"""
        
        print(f"Applying {strategy} sampling strategy...")
        
        if strategy == 'enhanced_minority_boost':
            return self._enhanced_minority_boost_sampling(df, target_col, sample_size)
        elif strategy == 'balanced_stratified':
            return self._balanced_stratified_sampling(df, target_col, sample_size)
        elif strategy == 'proportional_capped':
            return self._proportional_capped_sampling(df, target_col, sample_size)
        elif strategy == 'custom_ratios':
            return self._custom_percentage_ratios_sampling(df, target_col, sample_size)
        else:
            return self._enhanced_minority_boost_sampling(df, target_col, sample_size)
    
    def _custom_percentage_ratios_sampling(self, df: pd.DataFrame, target_col: str, 
                                         sample_size: int) -> pd.DataFrame:
        """Fixed custom ratios with percentage input that must sum to 100"""
        
        target_counts = df[target_col].value_counts().sort_index()
        
        print(f"Custom Percentage Ratio Configuration")
        print(f"Available classes: {list(target_counts.index)}")
        print(f"Target sample size: {sample_size:,}")
        print(f"Enter percentages for each class (must sum to 100)")
        
        # Get percentages from user
        percentages = {}
        total_percentage = 0
        
        for class_val in target_counts.index:
            while True:
                try:
                    percentage_input = input(f"Percentage for class {class_val} (0-100): ").strip()
                    percentage = float(percentage_input)
                    if 0 <= percentage <= 100:
                        percentages[class_val] = percentage
                        total_percentage += percentage
                        print(f"  Class {class_val}: {percentage}% (running total: {total_percentage}%)")
                        break
                    else:
                        print("Percentage must be between 0 and 100")
                except ValueError:
                    print("Invalid number, please enter a valid percentage")
        
        # Check if percentages sum to 100
        if abs(total_percentage - 100.0) > 0.1:  # Allow small rounding errors
            print(f"Warning: Percentages sum to {total_percentage}%, not 100%")
            print("Normalizing percentages to sum to 100%...")
            # Normalize percentages
            for class_val in percentages:
                percentages[class_val] = (percentages[class_val] / total_percentage) * 100.0
                print(f"  Normalized Class {class_val}: {percentages[class_val]:.1f}%")
        
        # Calculate samples per class
        sampled_dfs = []
        actual_total = 0
        
        for class_val, percentage in percentages.items():
            target_samples = int((percentage / 100.0) * sample_size)
            target_samples = max(10, target_samples)  # Minimum 10 samples per class
            
            class_df = df[df[target_col] == class_val]
            original_count = len(class_df)
            
            if original_count < target_samples:
                # Oversample
                sampled_class = resample(class_df, n_samples=target_samples,
                                       replace=True, random_state=42)
                print(f"   Class {class_val}: {original_count:,} -> {target_samples:,} ({percentage:.1f}% - oversampled)")
            else:
                # Subsample
                sampled_class = class_df.sample(n=target_samples, random_state=42)
                print(f"   Class {class_val}: {original_count:,} -> {target_samples:,} ({percentage:.1f}%)")
            
            sampled_dfs.append(sampled_class)
            actual_total += target_samples
        
        final_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"   Final custom sample: {len(final_df):,} rows (target: {sample_size:,})")
        print(f"   Final distribution: {dict(final_df[target_col].value_counts().sort_index())}")
        
        return final_df
    
    def _enhanced_minority_boost_sampling(self, df: pd.DataFrame, target_col: str, 
                                        sample_size: int) -> pd.DataFrame:
        """Enhanced minority boost sampling"""
        
        target_counts = df[target_col].value_counts().sort_index()
        majority_class = target_counts.idxmax()
        minority_classes = target_counts.drop(majority_class)
        
        sampled_dfs = []
        n_classes = len(target_counts)
        min_samples_per_minority = max(2000, sample_size // (n_classes * 3))  # At least 2k per minority
        
        # Process minority classes first
        minority_total = 0
        for class_val in minority_classes.index:
            class_df = df[df[target_col] == class_val]
            original_count = len(class_df)
            
            target_samples = min_samples_per_minority
            
            if original_count < target_samples:
                # Oversample with bootstrap
                bootstrap_df = resample(class_df, n_samples=target_samples,
                                      replace=True, random_state=42)
                
                # Add small noise to numeric features for diversity
                numeric_cols = bootstrap_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != target_col]
                
                for col in numeric_cols:
                    if bootstrap_df[col].std() > 0:
                        noise_std = bootstrap_df[col].std() * 0.01
                        noise = np.random.normal(0, noise_std, len(bootstrap_df))
                        bootstrap_df[col] = bootstrap_df[col] + noise
                
                sampled_dfs.append(bootstrap_df)
                print(f"   Enhanced class {class_val}: {original_count:,} -> {target_samples:,}")
            else:
                sampled_class = class_df.sample(n=target_samples, random_state=42)
                sampled_dfs.append(sampled_class)
                print(f"   Sampled class {class_val}: {original_count:,} -> {target_samples:,}")
            
            minority_total += target_samples
        
        # Process majority class
        majority_budget = max(sample_size - minority_total, minority_total * 2)
        majority_df = df[df[target_col] == majority_class]
        
        if len(majority_df) > majority_budget:
            sampled_majority = majority_df.sample(n=majority_budget, random_state=42)
        else:
            sampled_majority = majority_df
        
        sampled_dfs.append(sampled_majority)
        print(f"   Majority class {majority_class}: {len(majority_df):,} -> {len(sampled_majority):,}")
        
        final_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
        final_counts = final_df[target_col].value_counts().sort_index()
        final_ratio = final_counts.max() / final_counts.min()
        
        print(f"   Final sample: {len(final_df):,} rows")
        print(f"   Final distribution: {dict(final_counts)}")
        print(f"   Final ratio: {final_ratio:.1f}:1")
        
        return final_df
    
    def _balanced_stratified_sampling(self, df: pd.DataFrame, target_col: str, 
                                    sample_size: int) -> pd.DataFrame:
        """Balanced stratified sampling"""
        
        target_counts = df[target_col].value_counts().sort_index()
        n_classes = len(target_counts)
        samples_per_class = sample_size // n_classes
        
        print(f"   Target: {samples_per_class:,} samples per class")
        
        sampled_dfs = []
        
        for class_val, count in target_counts.items():
            class_df = df[df[target_col] == class_val]
            
            if count < samples_per_class:
                sampled_class = resample(class_df, n_samples=samples_per_class,
                                       replace=True, random_state=42)
                print(f"   Oversampled class {class_val}: {count:,} -> {samples_per_class:,}")
            else:
                sampled_class = class_df.sample(n=samples_per_class, random_state=42)
                print(f"   Undersampled class {class_val}: {count:,} -> {samples_per_class:,}")
            
            sampled_dfs.append(sampled_class)
        
        final_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"   Final balanced sample: {len(final_df):,} rows")
        print(f"   Final distribution: {dict(final_df[target_col].value_counts().sort_index())}")
        
        return final_df
    
    def _proportional_capped_sampling(self, df: pd.DataFrame, target_col: str, 
                                    sample_size: int) -> pd.DataFrame:
        """Proportional sampling with caps"""
        
        target_counts = df[target_col].value_counts().sort_index()
        total_rows = len(df)
        
        sampled_dfs = []
        
        for class_val, count in target_counts.items():
            proportion = count / total_rows
            proportional_size = int(proportion * sample_size)
            
            # Apply caps: minimum 100, maximum 80% of sample
            min_cap = min(100, count)
            max_cap = int(sample_size * 0.8)
            capped_size = max(min_cap, min(proportional_size, max_cap))
            
            class_df = df[df[target_col] == class_val]
            
            if count < capped_size:
                sampled_class = resample(class_df, n_samples=capped_size,
                                       replace=True, random_state=42)
                print(f"   Class {class_val}: {count:,} -> {capped_size:,} (oversampled)")
            else:
                sampled_class = class_df.sample(n=capped_size, random_state=42)
                print(f"   Class {class_val}: {count:,} -> {capped_size:,} (prop: {proportion:.1%})")
            
            sampled_dfs.append(sampled_class)
        
        final_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
        final_counts = final_df[target_col].value_counts().sort_index()
        final_ratio = final_counts.max() / final_counts.min()
        
        print(f"   Final proportional sample: {len(final_df):,} rows")
        print(f"   Final distribution: {dict(final_counts)}")
        print(f"   Final ratio: {final_ratio:.1f}:1")
        
        return final_df
    
    def _calculate_imbalance_ratio(self, df: pd.DataFrame, target_col: str) -> float:
        """Calculate imbalance ratio"""
        target_counts = df[target_col].value_counts()
        return target_counts.max() / target_counts.min() if len(target_counts) > 1 else 1.0
    
    @contextmanager
    def _timeout_context(self, seconds):
        """Context manager for method timeouts"""
        def timeout_handler(signum, frame):
            raise TimeoutError("Method execution timeout")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _prepare_data_robust(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Robust data preparation optimized for dataset characteristics"""
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        print("Step 1: Dataset-optimized cleaning...")
        
        # Enhanced cleaning for manufacturing data
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            
            if X[col].notna().any():
                # Conservative outlier detection for manufacturing precision
                Q1 = X[col].quantile(0.005)  # Very conservative for precision data
                Q3 = X[col].quantile(0.995)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    X[col] = np.clip(X[col], lower_bound, upper_bound)
        
        print("Step 2: Smart imputation for manufacturing data...")
        
        # Manufacturing-specific imputation
        if X.isnull().any().any():
            for col in numeric_cols:
                if X[col].isnull().any():
                    # For manufacturing measurements, use median (more robust)
                    X[col] = X[col].fillna(X[col].median())
            
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                # Use most frequent for categorical manufacturing data
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_val)
        
        print("Step 3: Manufacturing-optimized encoding...")
        
        feature_names = []
        numeric_data = []
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_names.append(col)
                col_data = X[col].astype(np.float64)
                # Prevent extreme values that could affect manufacturing models
                col_data = np.clip(col_data, -1e6, 1e6)
                numeric_data.append(col_data.values)
            else:
                unique_count = X[col].nunique()
                
                if unique_count <= 2:
                    # Binary encoding for quality flags
                    feature_names.append(col)
                    col_data = pd.factorize(X[col])[0].astype(np.float64)
                    numeric_data.append(col_data)
                elif unique_count <= 20:  # Increased threshold for manufacturing codes
                    # One-hot for manufacturing categories
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    for dummy_col in dummies.columns:
                        feature_names.append(dummy_col)
                        numeric_data.append(dummies[dummy_col].astype(np.float64).values)
                else:
                    # Target encoding with stronger smoothing for high-cardinality IDs
                    feature_names.append(col)
                    global_mean = y.mean()
                    
                    encoding_map = {}
                    for cat_val in X[col].unique():
                        if pd.notna(cat_val):
                            mask = X[col] == cat_val
                            if mask.any():
                                cat_target = y[mask]
                                if len(cat_target) >= 10:  # Higher threshold for stability
                                    cat_mean = cat_target.mean()
                                    # Stronger smoothing for manufacturing IDs
                                    smoothed = (cat_mean * len(cat_target) + global_mean * 50) / (len(cat_target) + 50)
                                    encoding_map[cat_val] = smoothed
                                else:
                                    encoding_map[cat_val] = global_mean
                            else:
                                encoding_map[cat_val] = global_mean
                        else:
                            encoding_map[cat_val] = global_mean
                    
                    col_data = X[col].map(encoding_map).fillna(global_mean)
                    numeric_data.append(col_data.astype(np.float64).values)
        
        # Create robust feature matrix
        X_array = np.column_stack(numeric_data)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # Prepare target
        if y.isnull().any():
            mode_val = y.mode()
            y = y.fillna(mode_val[0] if len(mode_val) > 0 else 0)
        
        if not pd.api.types.is_numeric_dtype(y):
            le_target = LabelEncoder()
            y_array = le_target.fit_transform(y.astype(str))
        else:
            y_array = y.values.astype(int)
        
        print(f"Data prepared: {X_array.shape[0]:,} samples × {X_array.shape[1]} features")
        unique_classes, class_counts = np.unique(y_array, return_counts=True)
        class_dist = dict(zip(unique_classes, class_counts))
        print(f"Class distribution: {class_dist}")
        
        return X_array, y_array, feature_names
    
    # TUNED MODEL METHODS FOR MANUFACTURING DATA
    
    def _analyze_permutation_tuned(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Tuned permutation importance for manufacturing failure prediction"""
        
        print("Computing manufacturing-tuned permutation importance...")
        
        # Optimized RandomForest for manufacturing data
        base_model = RandomForestClassifier(
            n_estimators=200,      # More trees for stability
            max_depth=12,          # Balanced depth for manufacturing complexity
            min_samples_split=10,  # Lower threshold for manufacturing precision
            min_samples_leaf=5,    # Lower threshold for rare failure patterns
            max_features='sqrt',   # Good balance for manufacturing features
            random_state=42,
            n_jobs=self.n_jobs,
            class_weight='balanced_subsample',
            bootstrap=True,
            oob_score=True,
            criterion='gini'       # Better for binary-like manufacturing outcomes
        )
        
        base_model.fit(X, y)
        
        print(f"Base model OOB score: {base_model.oob_score_:.4f}")
        
        # Enhanced permutation with more stability for manufacturing
        perm_importance = permutation_importance(
            base_model, X, y,
            n_repeats=20,  # More repeats for manufacturing stability
            random_state=42,
            scoring='balanced_accuracy',  # Better for manufacturing failure prediction
            n_jobs=min(4, self.n_jobs if self.n_jobs > 0 else 1)
        )
        
        importance_mean = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
        
        # Manufacturing-specific stability filtering
        stable_importance = []
        for i, (mean_imp, std_imp) in enumerate(zip(importance_mean, importance_std)):
            if mean_imp > 0:
                # Less aggressive filtering for manufacturing (coefficient of variation < 3)
                cv = std_imp / (mean_imp + 1e-10)
                if cv < 3.0:
                    stable_importance.append(mean_imp)
                else:
                    # More conservative estimate for high-variance features
                    stable_importance.append(max(0, mean_imp - 0.5 * std_imp))
            else:
                stable_importance.append(0.0)
        
        importance_scores = dict(zip(feature_names, stable_importance))
        
        method_details = {
            'base_model': 'RandomForest_manufacturing_tuned',
            'n_repeats': 20,
            'scoring_metric': 'balanced_accuracy',
            'oob_score': base_model.oob_score_,
            'n_estimators': base_model.n_estimators,
            'mean_stability': np.mean(importance_std),
            'stable_features': sum(1 for imp in stable_importance if imp > 0.001)
        }
        
        print(f"Manufacturing-tuned permutation completed (OOB: {base_model.oob_score_:.4f})")
        print(f"Stable features: {method_details['stable_features']}/{len(feature_names)}")
        
        return importance_scores, method_details
    
    def _analyze_gradient_boosting_optimized(self, X: np.ndarray, y: np.ndarray,
                                           feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Optimized Gradient Boosting for manufacturing failure patterns"""
        
        print("Computing manufacturing-optimized Gradient Boosting...")
        
        # Manufacturing-optimized Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=400,      # More estimators for complex manufacturing patterns
            learning_rate=0.05,    # Careful learning for manufacturing precision
            max_depth=8,           # Deeper for manufacturing complexity
            min_samples_split=20,  # Prevent overfitting on manufacturing IDs
            min_samples_leaf=10,   # Prevent overfitting
            subsample=0.8,         # Regularization
            max_features='sqrt',   # Feature randomness
            validation_fraction=0.1,
            n_iter_no_change=20,   # More patience for manufacturing convergence
            tol=1e-5,              # Tighter tolerance
            random_state=42
        )
        
        # Advanced sample weighting for manufacturing imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y)
        # Cap weights to prevent numerical instability
        max_weight = np.percentile(sample_weights, 95)
        sample_weights = np.minimum(sample_weights, max_weight)
        
        gb_model.fit(X, y, sample_weight=sample_weights)
        
        # Get feature importance with manufacturing-specific processing
        raw_importance = gb_model.feature_importances_
        
        # Apply light smoothing to reduce noise in manufacturing features
        if len(raw_importance) > 5:
            from scipy.ndimage import gaussian_filter1d
            smoothed_importance = gaussian_filter1d(raw_importance, sigma=0.3)
        else:
            smoothed_importance = raw_importance
        
        importance_scores = dict(zip(feature_names, smoothed_importance))
        
        method_details = {
            'model_type': 'gradient_boosting_manufacturing_optimized',
            'n_estimators': gb_model.n_estimators_,
            'train_score': gb_model.train_score_[-1] if hasattr(gb_model, 'train_score_') else 0,
            'validation_score': gb_model.validation_scores_[-1] if hasattr(gb_model, 'validation_scores_') else 0,
            'feature_importance_type': 'gain_manufacturing_smoothed',
            'convergence_iteration': getattr(gb_model, 'n_estimators_', gb_model.n_estimators),
            'weight_capping_applied': True
        }
        
        train_score = method_details.get('train_score', 0)
        val_score = method_details.get('validation_score', 0)
        
        print(f"Manufacturing-optimized Gradient Boosting completed")
        print(f"Training score: {train_score:.4f}")
        print(f"Validation score: {val_score:.4f}")
        
        return importance_scores, method_details
    
    def _analyze_univariate_enhanced(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Enhanced univariate analysis for manufacturing features"""
        
        print("Computing manufacturing-enhanced univariate tests...")
        
        n_features = X.shape[1]
        importance_scores_list = []
        
        # Use multiple complementary tests for manufacturing data
        tests_used = []
        
        try:
            # Test 1: Mutual Information (best for manufacturing nonlinear relationships)
            mi_scores = mutual_info_classif(X, y, random_state=42)
            tests_used.append('mutual_information')
            print("  Mutual information test completed")
        except:
            mi_scores = np.zeros(n_features)
        
        try:
            # Test 2: ANOVA F-test (for linear relationships)
            f_scores, p_values = f_classif(X, y)
            # Convert to importance
            f_importance = np.maximum(0, f_scores)
            tests_used.append('anova_f_test')
            print("  ANOVA F-test completed")
        except:
            f_importance = np.zeros(n_features)
        
        try:
            # Test 3: Chi-square test for categorical-like features
            # Discretize continuous features for chi-square
            X_discretized = np.zeros_like(X)
            for i in range(n_features):
                if len(np.unique(X[:, i])) > 20:
                    # Discretize into 5 bins
                    X_discretized[:, i] = pd.qcut(X[:, i], q=5, labels=False, duplicates='drop')
                else:
                    X_discretized[:, i] = X[:, i]
            
            # Make values non-negative for chi-square
            X_discretized = X_discretized - X_discretized.min(axis=0) + 1
            
            chi2_scores, _ = chi2(X_discretized, y)
            chi2_importance = np.maximum(0, chi2_scores)
            tests_used.append('chi_square')
            print("  Chi-square test completed")
        except:
            chi2_importance = np.zeros(n_features)
        
        # Combine test results with manufacturing-appropriate weighting
        combined_importance = np.zeros(n_features)
        
        if 'mutual_information' in tests_used:
            # MI is most important for manufacturing (40%)
            combined_importance += 0.4 * (mi_scores / (np.max(mi_scores) + 1e-10))
        
        if 'anova_f_test' in tests_used:
            # F-test for linear relationships (35%)
            combined_importance += 0.35 * (f_importance / (np.max(f_importance) + 1e-10))
        
        if 'chi_square' in tests_used:
            # Chi-square for categorical patterns (25%)
            combined_importance += 0.25 * (chi2_importance / (np.max(chi2_importance) + 1e-10))
        
        # Apply FDR correction if we have p-values
        n_significant = 0
        if 'anova_f_test' in tests_used:
            try:
                from statsmodels.stats.multitest import fdrcorrection
                rejected, corrected_p_values = fdrcorrection(p_values, alpha=0.05)
                n_significant = sum(rejected)
            except ImportError:
                n_significant = sum(p < 0.05 for p in p_values) if 'p_values' in locals() else 0
        
        importance_scores = dict(zip(feature_names, combined_importance))
        
        method_details = {
            'test_type': 'manufacturing_multitest_combined',
            'tests_used': tests_used,
            'correction_method': 'benjamini_hochberg_fdr',
            'n_significant_features': n_significant,
            'mean_combined_score': np.mean(combined_importance),
            'normalization_method': 'max_normalization_weighted'
        }
        
        print(f"Manufacturing-enhanced univariate completed")
        print(f"Tests used: {', '.join(tests_used)}")
        print(f"Significant features: {n_significant}/{n_features}")
        
        return importance_scores, method_details
    
    def _analyze_correlation_network_robust(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Robust correlation network for manufacturing feature relationships"""
        
        print("Computing manufacturing-robust correlation network...")
        
        # Enhanced target correlations for manufacturing
        target_corrs = []
        
        if len(np.unique(y)) > 2:
            # Multi-class: prioritize mutual information for manufacturing
            try:
                mi_scores = mutual_info_classif(X, y, random_state=42)
                target_corrs = mi_scores.tolist()
            except:
                # Fallback to correlation
                y_numeric = pd.factorize(y)[0]
                for i in range(X.shape[1]):
                    try:
                        corr = np.corrcoef(X[:, i], y_numeric)[0, 1]
                        target_corrs.append(abs(corr) if not np.isnan(corr) else 0.0)
                    except:
                        target_corrs.append(0.0)
        else:
            # Binary: use correlation
            y_numeric = pd.factorize(y)[0]
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y_numeric)[0, 1]
                    target_corrs.append(abs(corr) if not np.isnan(corr) else 0.0)
                except:
                    target_corrs.append(0.0)
        
        # Manufacturing-robust correlation matrix
        corr_matrix = np.eye(X.shape[1])  # Initialize with identity
        
        # Compute correlations with manufacturing-appropriate method
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                try:
                    # Use Spearman for manufacturing robustness
                    corr, _ = spearmanr(X[:, i], X[:, j])
                    corr = 0.0 if np.isnan(corr) else abs(corr)
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
                except:
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.0
        
        # Manufacturing-specific network analysis
        centrality_scores = []
        for i in range(len(feature_names)):
            connections = corr_matrix[i, :]
            # Focus on moderate to strong connections for manufacturing
            meaningful_connections = connections[(connections > 0.2) & (connections < 0.9)]
            # Avoid highly correlated (redundant) features
            centrality = np.sum(meaningful_connections) + 0.1 * len(meaningful_connections)
            centrality_scores.append(centrality)
        
        # Manufacturing-optimized combination
        importance_values = []
        max_target_corr = max(target_corrs) if max(target_corrs) > 0 else 1.0
        max_centrality = max(centrality_scores) if max(centrality_scores) > 0 else 1.0
        
        for target_corr, centrality in zip(target_corrs, centrality_scores):
            # Manufacturing weighting: prioritize target correlation over network centrality
            if target_corr > 0.05:  # Meaningful target correlation
                weight_target = 0.9
                weight_centrality = 0.1
            else:  # Weak target correlation
                weight_target = 0.7
                weight_centrality = 0.3
            
            combined_score = (weight_target * (target_corr / max_target_corr) + 
                            weight_centrality * (centrality / max_centrality))
            importance_values.append(combined_score)
        
        importance_scores = dict(zip(feature_names, importance_values))
        
        method_details = {
            'analysis_type': 'manufacturing_spearman_network',
            'correlation_method': 'spearman_rank',
            'centrality_method': 'meaningful_connections_filtered',
            'mean_target_correlation': np.mean(target_corrs),
            'mean_centrality': np.mean(centrality_scores),
            'connection_thresholds': {'min': 0.2, 'max': 0.9}
        }
        
        print(f"Manufacturing-robust correlation network completed")
        print(f"Mean target correlation: {np.mean(target_corrs):.4f}")
        
        return importance_scores, method_details
    
    # Additional methods would follow similar pattern...
    # Implementing key ones for space
    
    def _analyze_elastic_net_fixed(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Fixed Elastic Net implementation"""
        
        print("Computing fixed Elastic Net selection...")
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.multiclass import OneVsRestClassifier
        
        # Robust scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate balanced class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        try:
            min_class_count = min(np.bincount(y))
            cv_folds = min(5, max(3, min_class_count // 30))
            
            base_estimator = LogisticRegressionCV(
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.1, 0.5, 0.9],
                Cs=np.logspace(-3, 2, 10),
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                max_iter=3000,
                tol=1e-5,
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=1
            )
            
            if len(classes) > 2:
                model = OneVsRestClassifier(base_estimator, n_jobs=1)
            else:
                model = base_estimator
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X_scaled, y)
            
            # Extract coefficients
            if hasattr(model, 'estimators_'):
                coefficients = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'coef_') and estimator.coef_ is not None:
                        coefficients.append(np.abs(estimator.coef_.flatten()))
                if coefficients:
                    mean_coef = np.mean(coefficients, axis=0)
                else:
                    mean_coef = np.zeros(len(feature_names))
            else:
                if hasattr(model, 'coef_') and model.coef_ is not None:
                    mean_coef = np.abs(model.coef_.flatten())
                else:
                    mean_coef = np.zeros(len(feature_names))
            
            importance_scores = dict(zip(feature_names, mean_coef))
            
            method_details = {
                'model_type': 'elastic_net_fixed',
                'convergence_status': 'completed',
                'cv_folds': cv_folds
            }
            
        except Exception as e:
            print(f"Elastic Net failed: {e}, using Ridge fallback")
            
            from sklearn.linear_model import RidgeClassifierCV
            ridge_model = RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10),
                cv=3,
                class_weight=class_weight_dict
            )
            ridge_model.fit(X_scaled, y)
            
            if hasattr(ridge_model, 'coef_'):
                if ridge_model.coef_.ndim > 1:
                    mean_coef = np.mean(np.abs(ridge_model.coef_), axis=0)
                else:
                    mean_coef = np.abs(ridge_model.coef_)
            else:
                mean_coef = np.zeros(len(feature_names))
                
            importance_scores = dict(zip(feature_names, mean_coef))
            
            method_details = {
                'model_type': 'ridge_fallback',
                'reason': 'elastic_net_failed'
            }
        
        print(f"Elastic Net selection completed")
        
        return importance_scores, method_details
    
    def _analyze_information_gain_enhanced(self, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Enhanced Information Gain Ratio"""
        
        print("Computing enhanced Information Gain Ratio...")
        
        def entropy(labels):
            if len(labels) == 0:
                return 0
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        def enhanced_igr(feature, target):
            target_entropy = entropy(target)
            unique_vals = len(np.unique(feature))
            
            # Enhanced discretization
            if unique_vals > 15:
                n_bins = min(8, int(np.sqrt(len(feature))))
                try:
                    feature = pd.qcut(feature, q=n_bins, duplicates='drop', labels=False)
                except:
                    feature = pd.cut(feature, bins=n_bins, labels=False, duplicates='drop')
            
            feature_values = np.unique(feature[~pd.isna(feature)])
            weighted_entropy = 0
            intrinsic_value = 0
            
            for value in feature_values:
                if pd.isna(value):
                    continue
                subset = target[feature == value]
                if len(subset) > 0:
                    weight = len(subset) / len(target)
                    weighted_entropy += weight * entropy(subset)
                    if weight > 0:
                        intrinsic_value -= weight * np.log2(weight)
            
            information_gain = target_entropy - weighted_entropy
            
            if intrinsic_value < 1e-10:
                return information_gain
            else:
                return information_gain / intrinsic_value
        
        # Compute IGR for each feature
        igr_scores = []
        for i in range(X.shape[1]):
            try:
                igr = enhanced_igr(X[:, i], y)
                igr_scores.append(max(0, igr))
            except Exception:
                igr_scores.append(0.0)
        
        importance_scores = dict(zip(feature_names, igr_scores))
        
        method_details = {
            'algorithm': 'information_gain_ratio_enhanced',
            'mean_igr': np.mean(igr_scores)
        }
        
        print(f"Enhanced Information Gain Ratio completed")
        
        return importance_scores, method_details
    
    def _analyze_logistic_regression_tuned(self, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Tuned Logistic Regression"""
        
        print("Computing tuned Logistic Regression...")
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.multiclass import OneVsRestClassifier
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        try:
            min_class_count = min(np.bincount(y))
            cv_folds = min(5, max(3, min_class_count // 20))
            
            base_estimator = LogisticRegressionCV(
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.3, 0.7],
                Cs=np.logspace(-2, 2, 8),
                cv=cv_folds,
                max_iter=2000,
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=1
            )
            
            if len(classes) > 2:
                model = OneVsRestClassifier(base_estimator, n_jobs=1)
            else:
                model = base_estimator
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X_scaled, y)
            
            if hasattr(model, 'estimators_'):
                coefficients = [np.abs(est.coef_.flatten()) for est in model.estimators_ if hasattr(est, 'coef_')]
                mean_coef = np.mean(coefficients, axis=0) if coefficients else np.zeros(len(feature_names))
            else:
                mean_coef = np.abs(model.coef_.flatten()) if hasattr(model, 'coef_') else np.zeros(len(feature_names))
            
            importance_scores = dict(zip(feature_names, mean_coef))
            
            method_details = {
                'model_type': 'logistic_regression_tuned',
                'cv_folds': cv_folds
            }
            
        except Exception as e:
            print(f"Logistic Regression failed: {e}")
            importance_scores = dict(zip(feature_names, np.zeros(len(feature_names))))
            method_details = {'model_type': 'failed', 'error': str(e)}
        
        print(f"Tuned Logistic Regression completed")
        
        return importance_scores, method_details
    
    def _analyze_consensus_ranking_robust(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Tuple[Dict[str, float], Dict]:
        """Robust consensus ranking"""
        
        print("Computing robust consensus ranking...")
        
        methods_results = {}
        methods_weights = {}
        
        # Run multiple methods
        try:
            print("  Running permutation...")
            perm_scores, _ = self._analyze_permutation_tuned(X, y, feature_names)
            methods_results['permutation'] = perm_scores
            methods_weights['permutation'] = 0.3
        except:
            pass
        
        try:
            print("  Running gradient boosting...")
            gb_scores, _ = self._analyze_gradient_boosting_optimized(X, y, feature_names)
            methods_results['gradient_boosting'] = gb_scores
            methods_weights['gradient_boosting'] = 0.3
        except:
            pass
        
        try:
            print("  Running univariate...")
            uni_scores, _ = self._analyze_univariate_enhanced(X, y, feature_names)
            methods_results['univariate'] = uni_scores
            methods_weights['univariate'] = 0.2
        except:
            pass
        
        try:
            print("  Running correlation network...")
            corr_scores, _ = self._analyze_correlation_network_robust(X, y, feature_names)
            methods_results['correlation'] = corr_scores
            methods_weights['correlation'] = 0.2
        except:
            pass
        
        # Fallback if no methods worked
        if not methods_results:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X, y)
            methods_results['rf_fallback'] = dict(zip(feature_names, rf.feature_importances_))
            methods_weights['rf_fallback'] = 1.0
        
        # Combine results
        consensus_scores = {name: 0.0 for name in feature_names}
        total_weight = sum(methods_weights.values())
        
        for method_name, scores in methods_results.items():
            score_values = np.array(list(scores.values()))
            if score_values.max() > 0:
                normalized_scores = score_values / score_values.max()
            else:
                normalized_scores = score_values
            
            weight = methods_weights[method_name] / total_weight
            for i, feature_name in enumerate(feature_names):
                consensus_scores[feature_name] += weight * normalized_scores[i]
        
        method_details = {
            'methods_used': list(methods_results.keys()),
            'n_successful_methods': len(methods_results)
        }
        
        print(f"Robust consensus completed ({len(methods_results)} methods)")
        
        return consensus_scores, method_details
    
    def _display_and_select_fixed(self, importance_scores: Dict[str, float],
                                method_details: Dict, method: str,
                                target_col: str, method_info: Dict) -> List[str]:
        """Fixed display and selection with proper return handling"""
        
        if not importance_scores:
            print("No importance scores to display")
            return []
        
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        max_importance = max(s[1] for s in sorted_features) if sorted_features else 1.0
        max_importance = max(max_importance, 1e-10)  # Prevent division by zero
        default_cutoff = method_info['default_cutoff']
        
        print(f"\n{'='*155}")
        print(f"FEATURE IMPORTANCE RESULTS - {method.upper()}")
        print(f"{'='*155}")
        print(f"Target Variable: {target_col}")
        print(f"Features Analyzed: {len(importance_scores)}")
        print(f"[{method_info['symbol']}] Method: {method_info['name']}")
        print(f"Research Basis: {method_info['research_basis']}")
        print(f"Default Cutoff: {default_cutoff}")
        if 'oob_score' in method_details:
            print(f"Model Performance: {method_details['oob_score']:.4f}")
        print(f"Execution Time: {method_details.get('execution_time', 0):.2f}s")
        print(f"{'='*120}")
        
        # Display table
        print(f"{'Rank':<6} {'Feature Name':<40} {'Raw Score':<12} {'Normalized':<12} {'Category':<15} {'Action'}")
        print("-" * 120)
        
        features_by_cutoff = {'default': [], 'conservative': [], 'aggressive': []}
        
        for i, (feature, score) in enumerate(sorted_features, 1):
            normalized_score = score / max_importance
            
            if normalized_score >= 0.7:
                category = "CRITICAL"
            elif normalized_score >= 0.4:
                category = "HIGH"
            elif normalized_score >= 0.15:
                category = "MODERATE"
            elif normalized_score >= 0.05:
                category = "LOW"
            else:
                category = "MINIMAL"
            
            action = "DROP" if score < default_cutoff else "KEEP"
            
            if score < default_cutoff:
                features_by_cutoff['default'].append(feature)
            if score < default_cutoff * 10:
                features_by_cutoff['conservative'].append(feature)
            if score < default_cutoff * 0.1:
                features_by_cutoff['aggressive'].append(feature)
            
            feature_display = feature[:39] if len(feature) <= 39 else feature[:36] + "..."
            
            print(f"{i:<6} {feature_display:<40} {score:<12.6f} {normalized_score:<12.3f} {category:<15} {action}")
        
        print("-" * 120)
        
        # Summary
        print(f"ANALYSIS SUMMARY:")
        print(f"  • Features Analyzed: {len(sorted_features)}")
        print(f"  • Critical Features (>=0.7): {sum(1 for _, score in sorted_features if score/max_importance >= 0.7)}")
        print(f"  • High Features (>=0.4): {sum(1 for _, score in sorted_features if score/max_importance >= 0.4)}")
        print(f"  • Zero/Minimal Scores: {sum(1 for _, score in sorted_features if score <= 1e-6)}")
        
        print(f"\nCUTOFF OPTIONS:")
        print(f"  1. Default ({default_cutoff:.6f}): Drop {len(features_by_cutoff['default'])} features")
        print(f"  2. Conservative ({default_cutoff * 10:.6f}): Drop {len(features_by_cutoff['conservative'])} features")
        print(f"  3. Aggressive ({default_cutoff * 0.1:.6f}): Drop {len(features_by_cutoff['aggressive'])} features")
        
        # Interactive selection
        print(f"\nFEATURE SELECTION OPTIONS:")
        print(f"1. Use default cutoff ({default_cutoff:.6f})")
        print(f"2. Use conservative cutoff ({default_cutoff * 10:.6f})")
        print(f"3. Use aggressive cutoff ({default_cutoff * 0.1:.6f})")
        print(f"4. Drop only zero-importance features")
        print(f"5. Manual selection")
        print(f"6. Keep all features")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    return list(features_by_cutoff['default'])
                elif choice == '2':
                    return list(features_by_cutoff['conservative'])
                elif choice == '3':
                    return list(features_by_cutoff['aggressive'])
                elif choice == '4':
                    return [feat for feat, score in sorted_features if score <= 1e-6]
                elif choice == '5':
                    print(f"\nTop features:")
                    for feat, score in sorted_features[:10]:
                        print(f"  {feat} ({score:.6f})")
                    manual_input = input(f"\nFeatures to drop (comma-separated): ").strip()
                    if manual_input:
                        manual_features = [f.strip() for f in manual_input.split(',')]
                        valid_features = [f for f in manual_features if f in importance_scores]
                        return list(valid_features)
                    return []
                elif choice == '6':
                    return []
                else:
                    print("Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                print("\nSelection cancelled. Keeping all features.")
                return []
    
    def _create_fixed_visualization(self, importance_scores: Dict[str, float],
                                  method: str, target_col: str, method_info: Dict):
        """Fixed visualization with proper error handling"""
        
        if not importance_scores:
            return
        
        try:
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'[{method_info["symbol"]}] {method_info["name"]}\nTarget: {target_col}', 
                        fontsize=14, fontweight='bold')
            
            # 1. Top features bar plot
            features = [f[0] for f in sorted_features[:15]]
            scores = [f[1] for f in sorted_features[:15]]
            
            bars = ax1.barh(range(len(features)), scores, color='steelblue', alpha=0.7)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in features], fontsize=9)
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top 15 Features')
            ax1.grid(axis='x', alpha=0.3)
            
            # 2. Score distribution
            all_scores = list(importance_scores.values())
            ax2.hist(all_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(method_info['default_cutoff'], color='red', linestyle='--', 
                       label=f'Cutoff ({method_info["default_cutoff"]:.6f})')
            ax2.set_xlabel('Importance Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # 3. Cumulative importance
            sorted_scores = sorted(all_scores, reverse=True)
            if max(sorted_scores) > 0:
                cumulative_scores = np.cumsum(sorted_scores)
                cumulative_pct = cumulative_scores / cumulative_scores[-1] * 100
                ax3.plot(range(1, len(sorted_scores) + 1), cumulative_pct, 'b-', linewidth=2)
                ax3.set_xlabel('Features (Ranked)')
                ax3.set_ylabel('Cumulative Percent')
                ax3.set_title('Cumulative Importance')
                ax3.grid(alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No positive importance scores', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cumulative Importance')
            
            # 4. Summary information
            ax4.axis('off')
            
            summary_lines = [
                "ANALYSIS SUMMARY",
                "",
                f"Method: {method_info['name'][:30]}",
                f"Research: {method_info['research_basis'][:40]}",
                "",
                f"Features: {len(importance_scores)}",
                f"Top Feature: {sorted_features[0][0][:25] if sorted_features else 'None'}",
                f"Top Score: {sorted_features[0][1]:.6f if sorted_features else 0}",
                "",
                f"Zero Scores: {sum(1 for s in all_scores if abs(s) < 1e-10)}",
                f"Mean: {np.mean(all_scores):.6f}",
                f"Max: {max(all_scores):.6f}",
                f"Min: {min(all_scores):.6f}"
            ]
            
            summary_text = "\n".join(summary_lines)
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save with proper filename handling
            plot_filename = f'feature_importance_{method}_{target_col}.png'
            plot_filename = ''.join(c for c in plot_filename if c.isalnum() or c in '._-')
            plot_file = self.plots_dir / plot_filename
            
            plt.savefig(str(plot_file), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Visualization saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
            print(f"Visualization failed: {e}")
            plt.close('all')