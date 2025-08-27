"""
REFACTORED Balance Handler - Critical Issues Fixed
Based on log analysis for 500/5.5M imbalanced dataset

KEY FIXES:
1. Removed redundant methods (custom_ratio, custom_balance fallback to same class_weights)
2. Fixed missing method implementations (_prompt_custom_ratios, _cost_sensitive_balancing)
3. Corrected ratio calculations (showing 3-class but claiming binary)
4. Added proper memory handling for extreme imbalance
5. Streamlined to 4 core methods with real differentiation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import time
import gc
import psutil
from datetime import datetime

# Enhanced balancing libraries with fallbacks
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

class FixedBalanceHandler:
    def __init__(self, logger: logging.Logger, plots_dir: Path):
        self.logger = logger
        self.plots_dir = plots_dir
        
        # STREAMLINED: Only 4 core methods with real differentiation
        self.balancing_methods = {
            'smote': {
                'name': 'SMOTE (Synthetic Minority Oversampling)',
                'description': 'Generate synthetic samples using k-nearest neighbors',
                'best_for': 'Numerical features, minority class ≥50 samples',
                'available': IMBLEARN_AVAILABLE,
                'memory_efficient': True,
                'suitable_for_extreme': False  # Not good for 500/5.5M ratio
            },
            'adaptive_oversample': {
                'name': 'Adaptive Random Oversampling',
                'description': 'Intelligent oversampling with progressive ratios',
                'best_for': 'Extreme imbalance (>1000:1), memory constraints',
                'available': True,
                'memory_efficient': True,
                'suitable_for_extreme': True
            },
            'stratified_undersample': {
                'name': 'Stratified Undersampling',
                'description': 'Preserve class distribution while reducing majority',
                'best_for': 'Large datasets, maintain representativeness',
                'available': True,
                'memory_efficient': True,
                'suitable_for_extreme': True
            },
            'class_weights': {
                'name': 'Balanced Class Weights',
                'description': 'Algorithm-level balancing without data modification',
                'best_for': 'Preserve all data, use with any classifier',
                'available': True,
                'memory_efficient': True,
                'suitable_for_extreme': True
            }
        }
        
        self.max_sample_size = 100000  # Reduced for extreme imbalance
        self.imbalance_thresholds = {
            'balanced': 3.0,
            'mild': 10.0, 
            'severe': 100.0,
            'extreme': 1000.0,
            'critical': float('inf')
        }

    def apply_enhanced_balancing(self, df: pd.DataFrame, target_col: str, 
                               method: str) -> Tuple[pd.DataFrame, Dict]:
        
        start_time = time.time()
        self.logger.info(f"Applying {method} balancing...")
        
        if method not in self.balancing_methods:
            available_methods = list(self.balancing_methods.keys())
            raise ValueError(f"Unknown method: {method}. Available: {available_methods}")
        
        method_info = self.balancing_methods[method]
        
        print(f"\n{method_info['name']}")
        print(f"Description: {method_info['description']}")
        print(f"Best for: {method_info['best_for']}")
        
        # Memory usage monitoring
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        # Analyze imbalance
        print("Step 1: Analyzing class imbalance...")
        imbalance_analysis = self._analyze_imbalance_comprehensive(df, target_col)
        
        # Check method suitability for extreme imbalance
        if not self._is_method_suitable(method_info, imbalance_analysis):
            print(f"Warning: {method} may not be optimal for this imbalance level")
            recommended = self._get_recommended_method(imbalance_analysis)
            print(f"Recommended: {recommended}")
            
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return self._apply_recommended_method(df, target_col, recommended, imbalance_analysis)
        
        # Memory optimization for extreme imbalance
        working_df = df
        if self._needs_memory_optimization(df, imbalance_analysis):
            print("Large dataset optimization for extreme imbalance...")
            working_df = self._optimize_for_extreme_imbalance(df, target_col, imbalance_analysis)
        
        # Apply balancing method
        print(f"Step 2: Applying {method} balancing...")
        try:
            balanced_df, balance_report = self._route_to_balancing_method(
                working_df, target_col, method, imbalance_analysis
            )
            
            # Evaluation
            print("Step 3: Evaluating results...")
            evaluation_results = self._evaluate_balancing_comprehensive(
                working_df, balanced_df, target_col, method
            )
            
            # Create visualization
            self._create_balance_visualization_fixed(working_df, balanced_df, target_col, method)
            
            # Resource usage summary
            final_memory = psutil.Process().memory_info().rss / 1024**2
            execution_time = time.time() - start_time
            
            print(f"\nBALANCING COMPLETED:")
            print(f"   Execution Time: {execution_time:.2f} seconds")
            print(f"   Memory Change: {final_memory - initial_memory:+.1f} MB")
            
            return balanced_df, {
                'method': method,
                'original_distribution': imbalance_analysis['distribution'],
                'final_distribution': balanced_df[target_col].value_counts().to_dict(),
                'balance_report': balance_report,
                'evaluation_results': evaluation_results,
                'execution_time': execution_time,
                'memory_optimized': len(df) != len(working_df)
            }
            
        except Exception as e:
            self.logger.error(f"Balancing failed: {str(e)}")
            print(f"Balancing failed: {str(e)}")
            return self._apply_fallback_class_weights(df, target_col, imbalance_analysis)

    def _analyze_imbalance_comprehensive(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Comprehensive imbalance analysis with correct calculations"""
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        target_series = df[target_col]
        value_counts = target_series.value_counts().sort_index()
        total_samples = len(target_series)
        
        if len(value_counts) < 2:
            return {
                'severity': 'single_class',
                'distribution': value_counts.to_dict(),
                'total_samples': total_samples,
                'error': 'Only one class present'
            }
        
        # FIXED: Correct calculations
        max_class_count = value_counts.max()
        min_class_count = value_counts.min()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()
        
        # Class percentages
        class_percentages = (value_counts / total_samples * 100).to_dict()
        
        severity = self._classify_imbalance_severity_fixed(imbalance_ratio)
        
        analysis = {
            'severity': severity,
            'distribution': value_counts.to_dict(),
            'class_percentages': class_percentages,
            'total_samples': total_samples,
            'n_classes': len(value_counts),
            'imbalance_ratio': imbalance_ratio,
            'majority_class': majority_class,
            'minority_class': minority_class,
            'majority_count': max_class_count,
            'minority_count': min_class_count,
            'missing_values': target_series.isnull().sum(),
            'is_extreme_imbalance': imbalance_ratio > 1000
        }
        
        self._display_imbalance_analysis_fixed(analysis)
        return analysis

    def _classify_imbalance_severity_fixed(self, imbalance_ratio: float) -> str:
        """Fixed severity classification"""
        if imbalance_ratio <= 3.0:
            return 'balanced'
        elif imbalance_ratio <= 10.0:
            return 'mild'
        elif imbalance_ratio <= 100.0:
            return 'severe'
        elif imbalance_ratio <= 1000.0:
            return 'extreme'
        else:
            return 'critical'

    def _display_imbalance_analysis_fixed(self, analysis: Dict):
        """Fixed display with correct information"""
        
        print(f"\nIMBALANCE ANALYSIS")
        print(f"="*60)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        severity = analysis['severity']
        severity_icons = {
            'balanced': 'BALANCED', 'mild': 'MILD', 'severe': 'SEVERE', 
            'extreme': 'EXTREME', 'critical': 'CRITICAL'
        }
        
        print(f"Dataset Overview:")
        print(f"   Total Samples: {analysis['total_samples']:,}")
        print(f"   Number of Classes: {analysis['n_classes']}")
        
        print(f"\nImbalance Metrics:")
        print(f"   Severity: {severity_icons.get(severity, 'UNKNOWN')} ({severity})")
        print(f"   Imbalance Ratio: {analysis['imbalance_ratio']:.2f}:1")
        
        print(f"\nClass Distribution:")
        for class_val in sorted(analysis['distribution'].keys()):
            count = analysis['distribution'][class_val]
            percentage = analysis['class_percentages'][class_val]
            
            if class_val == analysis['majority_class']:
                status = "MAJORITY"
            elif class_val == analysis['minority_class']:
                status = "MINORITY"
            else:
                status = "REGULAR"
            
            print(f"   Class {class_val}: {count:,} samples ({percentage:.2f}%) - {status}")
        
        self._provide_method_recommendations_fixed(analysis)

    def _provide_method_recommendations_fixed(self, analysis: Dict):
        """Provide specific recommendations for extreme imbalance"""
        
        severity = analysis['severity']
        minority_count = analysis['minority_count']
        imbalance_ratio = analysis['imbalance_ratio']
        
        print(f"\nRECOMMENDATIONS:")
        
        if severity == 'balanced':
            print("   Dataset is well-balanced - no balancing needed")
        elif severity in ['mild', 'severe']:
            if minority_count >= 50:
                print("   SMOTE or adaptive oversampling recommended")
            else:
                print("   Class weights or careful oversampling recommended")
        elif severity in ['extreme', 'critical']:
            print(f"   EXTREME IMBALANCE (ratio: {imbalance_ratio:.0f}:1)")
            print("   Recommended approaches:")
            print("   1. Adaptive oversampling (progressive ratios)")
            print("   2. Class weights (preserve all data)")
            print("   3. Stratified undersampling (if memory limited)")
            print("   4. Ensemble methods with cost-sensitive learning")

    def _is_method_suitable(self, method_info: Dict, analysis: Dict) -> bool:
        """Check if method is suitable for the imbalance level"""
        
        if analysis['severity'] in ['extreme', 'critical']:
            return method_info.get('suitable_for_extreme', False)
        
        if analysis['severity'] == 'mild' and 'smote' in method_info['name'].lower():
            return analysis['minority_count'] >= 50
        
        return True

    def _get_recommended_method(self, analysis: Dict) -> str:
        """Get recommended method based on analysis"""
        
        if analysis['severity'] in ['extreme', 'critical']:
            if analysis['total_samples'] > 1000000:
                return 'class_weights'  # Most efficient for very large datasets
            else:
                return 'adaptive_oversample'
        elif analysis['severity'] == 'severe':
            if analysis['minority_count'] >= 50:
                return 'smote'
            else:
                return 'adaptive_oversample'
        else:
            return 'smote'

    def _needs_memory_optimization(self, df: pd.DataFrame, analysis: Dict) -> bool:
        """Check if memory optimization is needed"""
        return (len(df) > self.max_sample_size and 
                analysis['severity'] in ['extreme', 'critical'])

    def _optimize_for_extreme_imbalance(self, df: pd.DataFrame, target_col: str, 
                                      analysis: Dict) -> pd.DataFrame:
        """Optimize dataset for extreme imbalance scenarios"""
        
        print(f"Optimizing for extreme imbalance ({analysis['imbalance_ratio']:.0f}:1)...")
        
        # Keep ALL minority samples
        minority_data = df[df[target_col] == analysis['minority_class']]
        print(f"Preserving all {len(minority_data)} minority samples")
        
        # Intelligent majority sampling
        majority_data = df[df[target_col] == analysis['majority_class']]
        
        # For extreme imbalance, use larger majority sample
        max_majority = min(
            len(majority_data),
            analysis['minority_count'] * 100,  # Max 100:1 ratio for processing
            50000  # Absolute cap
        )
        
        sampled_majority = majority_data.sample(n=max_majority, random_state=42)
        print(f"Sampled {len(sampled_majority)} from {len(majority_data)} majority samples")
        
        # Handle other classes if present
        other_classes = []
        for class_val in analysis['distribution'].keys():
            if class_val not in [analysis['minority_class'], analysis['majority_class']]:
                class_data = df[df[target_col] == class_val]
                # Keep reasonable sample of other classes
                sample_size = min(len(class_data), analysis['minority_count'] * 5)
                other_classes.append(class_data.sample(n=sample_size, random_state=42))
        
        # Combine all data
        data_parts = [minority_data, sampled_majority] + other_classes
        optimized_df = pd.concat(data_parts, axis=0).sample(frac=1, random_state=42)
        
        print(f"Optimized dataset: {len(df):,} -> {len(optimized_df):,} rows")
        return optimized_df

    def _route_to_balancing_method(self, df: pd.DataFrame, target_col: str,
                                 method: str, analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Route to appropriate balancing method"""
        
        if method == 'smote':
            return self._apply_smote_fixed(df, target_col, analysis)
        elif method == 'adaptive_oversample':
            return self._apply_adaptive_oversampling(df, target_col, analysis)
        elif method == 'stratified_undersample':
            return self._apply_stratified_undersampling(df, target_col, analysis)
        elif method == 'class_weights':
            return self._apply_class_weights_fixed(df, target_col, analysis)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def _apply_adaptive_oversampling(self, df: pd.DataFrame, target_col: str,
                                   analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """NEW: Adaptive oversampling for extreme imbalance"""
        
        print("Applying adaptive oversampling with progressive ratios...")
        
        # Progressive ratio strategy for extreme imbalance
        if analysis['imbalance_ratio'] > 1000:
            # Very conservative for extreme cases
            target_ratio = min(10.0, np.sqrt(analysis['imbalance_ratio']))
        elif analysis['imbalance_ratio'] > 100:
            target_ratio = min(20.0, analysis['imbalance_ratio'] / 10)
        else:
            target_ratio = min(5.0, analysis['imbalance_ratio'] / 2)
        
        print(f"Target ratio: {target_ratio:.1f}:1 (reduced from {analysis['imbalance_ratio']:.1f}:1)")
        
        balanced_data = []
        
        # Process each class
        for class_val in analysis['distribution'].keys():
            class_data = df[df[target_col] == class_val]
            current_count = len(class_data)
            
            if class_val == analysis['minority_class']:
                # Calculate target count for minority class
                target_count = int(analysis['minority_count'] * target_ratio)
                
                if target_count > current_count:
                    # Intelligent oversampling with noise injection
                    additional_samples = target_count - current_count
                    
                    # Add slight noise to prevent exact duplicates
                    oversampled_data = []
                    for _ in range(additional_samples):
                        base_sample = class_data.sample(n=1, random_state=None).iloc[0]
                        
                        # Add minimal noise to numerical columns
                        noisy_sample = base_sample.copy()
                        numeric_cols = class_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col != target_col:
                                noise = np.random.normal(0, abs(noisy_sample[col]) * 0.01)  # 1% noise
                                noisy_sample[col] = noisy_sample[col] + noise
                        
                        oversampled_data.append(noisy_sample)
                    
                    oversampled_df = pd.DataFrame(oversampled_data)
                    class_data = pd.concat([class_data, oversampled_df], ignore_index=True)
            
            balanced_data.append(class_data)
        
        balanced_df = pd.concat(balanced_data, axis=0).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        
        print(f"Adaptive oversampling completed: {len(df):,} -> {len(balanced_df):,} samples")
        
        return balanced_df, {
            'method': 'adaptive_oversampling',
            'target_ratio': target_ratio,
            'samples_added': len(balanced_df) - len(df),
            'noise_injection': True
        }

    def _apply_stratified_undersampling(self, df: pd.DataFrame, target_col: str,
                                      analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """NEW: Stratified undersampling preserving distribution"""
        
        print("Applying stratified undersampling...")
        
        # Target size based on minority class
        base_size = analysis['minority_count']
        
        # Different strategies based on imbalance severity
        if analysis['severity'] in ['extreme', 'critical']:
            # Very conservative undersampling
            multiplier = min(20, int(np.sqrt(analysis['imbalance_ratio'])))
        else:
            multiplier = 5
        
        target_majority_size = base_size * multiplier
        
        balanced_data = []
        
        for class_val in analysis['distribution'].keys():
            class_data = df[df[target_col] == class_val]
            
            if class_val == analysis['minority_class']:
                # Keep all minority samples
                balanced_data.append(class_data)
                print(f"Class {class_val}: kept all {len(class_data)} samples")
            else:
                # Stratified sampling for other classes
                sample_size = min(len(class_data), target_majority_size)
                
                # Stratified sampling if possible
                if len(class_data) > sample_size:
                    sampled_data = class_data.sample(n=sample_size, random_state=42)
                else:
                    sampled_data = class_data
                
                balanced_data.append(sampled_data)
                print(f"Class {class_val}: {len(class_data):,} -> {len(sampled_data):,} samples")
        
        balanced_df = pd.concat(balanced_data, axis=0).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        
        print(f"Stratified undersampling completed: {len(df):,} -> {len(balanced_df):,} samples")
        
        return balanced_df, {
            'method': 'stratified_undersampling',
            'multiplier': multiplier,
            'samples_removed': len(df) - len(balanced_df),
            'preservation_strategy': 'stratified'
        }

    def _apply_smote_fixed(self, df: pd.DataFrame, target_col: str,
                         analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED SMOTE implementation"""
        
        if not IMBLEARN_AVAILABLE:
            print("SMOTE not available. Using adaptive oversampling.")
            return self._apply_adaptive_oversampling(df, target_col, analysis)
        
        if analysis['minority_count'] < 6:
            print(f"SMOTE needs ≥6 minority samples, found {analysis['minority_count']}. Using adaptive oversampling.")
            return self._apply_adaptive_oversampling(df, target_col, analysis)
        
        print("Preparing data for SMOTE...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print("Applying SMOTE...")
        
        try:
            k_neighbors = min(5, analysis['minority_count'] - 1)
            
            # FIXED: Correct SMOTE parameters
            smote = SMOTE(
                k_neighbors=k_neighbors,
                random_state=42
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Combine back to DataFrame
            balanced_df = pd.concat([
                pd.DataFrame(X_resampled, columns=X.columns),
                pd.Series(y_resampled, name=target_col)
            ], axis=1)
            
            print(f"SMOTE completed: generated {len(balanced_df) - len(df):,} synthetic samples")
            
            return balanced_df, {
                'method': 'smote',
                'k_neighbors': k_neighbors,
                'samples_generated': len(balanced_df) - len(df),
                'categorical_encoded': len(categorical_cols)
            }
            
        except Exception as e:
            self.logger.warning(f"SMOTE failed: {e}")
            print(f"SMOTE failed: {e}. Using adaptive oversampling.")
            return self._apply_adaptive_oversampling(df, target_col, analysis)

    def _apply_class_weights_fixed(self, df: pd.DataFrame, target_col: str,
                                 analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """FIXED class weights with correct calculation"""
        
        print("Calculating balanced class weights...")
        
        distribution = analysis['distribution']
        total_samples = analysis['total_samples']
        n_classes = analysis['n_classes']
        
        # Sklearn-style balanced weights
        class_weights = {}
        for class_val, count in distribution.items():
            class_weights[class_val] = total_samples / (n_classes * count)
        
        print("Class weights calculated:")
        for class_val, weight in class_weights.items():
            print(f"   Class {class_val}: {weight:.4f}")
        
        return df, {
            'method': 'class_weights',
            'class_weights': class_weights,
            'sklearn_format': class_weights,
            'preserves_all_data': True
        }

    def _apply_fallback_class_weights(self, df: pd.DataFrame, target_col: str,
                                    analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fallback to class weights"""
        print("Applying fallback method (class weights)...")
        return self._apply_class_weights_fixed(df, target_col, analysis)

    def _apply_recommended_method(self, df: pd.DataFrame, target_col: str,
                                method: str, analysis: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Apply recommended method"""
        print(f"Applying recommended method: {method}")
        return self._route_to_balancing_method(df, target_col, method, analysis)

    def _evaluate_balancing_comprehensive(self, original_df: pd.DataFrame, 
                                        balanced_df: pd.DataFrame, target_col: str,
                                        method: str) -> Dict:
        """Comprehensive evaluation with correct metrics"""
        
        original_dist = original_df[target_col].value_counts().to_dict()
        balanced_dist = balanced_df[target_col].value_counts().to_dict()
        
        # FIXED: Correct ratio calculations
        original_ratio = max(original_dist.values()) / min(original_dist.values()) if original_dist else 1.0
        balanced_ratio = max(balanced_dist.values()) / min(balanced_dist.values()) if balanced_dist else 1.0
        
        improvement = ((original_ratio - balanced_ratio) / original_ratio * 100) if original_ratio > 0 else 0
        size_change = len(balanced_df) - len(original_df)
        size_change_pct = (size_change / len(original_df)) * 100 if len(original_df) > 0 else 0
        
        results = {
            'original_distribution': original_dist,
            'balanced_distribution': balanced_dist,
            'original_imbalance_ratio': original_ratio,
            'balanced_imbalance_ratio': balanced_ratio,
            'improvement_percentage': improvement,
            'size_change': size_change,
            'size_change_percentage': size_change_pct
        }
        
        print(f"\nBALANCING EVALUATION")
        print(f"="*50)
        print(f"Distribution Changes:")
        for class_val, count in original_dist.items():
            new_count = balanced_dist.get(class_val, 0)
            print(f"   Class {class_val}: {count:,} -> {new_count:,}")
        
        print(f"\nBalance Improvement:")
        print(f"   Original Ratio: {original_ratio:.2f}:1")
        print(f"   New Ratio: {balanced_ratio:.2f}:1")
        print(f"   Improvement: {improvement:.1f}%")
        
        return results

    def _create_balance_visualization_fixed(self, original_df: pd.DataFrame, 
                                          balanced_df: pd.DataFrame, target_col: str, method: str):
        """Create corrected visualization"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Original distribution
            orig_counts = original_df[target_col].value_counts().sort_index()
            axes[0].bar(range(len(orig_counts)), orig_counts.values, alpha=0.8, color='red')
            axes[0].set_title('Original Distribution')
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Count')
            axes[0].set_xticks(range(len(orig_counts)))
            axes[0].set_xticklabels(orig_counts.index)
            
            # Balanced distribution
            balanced_counts = balanced_df[target_col].value_counts().sort_index()
            axes[1].bar(range(len(balanced_counts)), balanced_counts.values, alpha=0.8, color='green')
            axes[1].set_title(f'After {method.replace("_", " ").title()}')
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Count')
            axes[1].set_xticks(range(len(balanced_counts)))
            axes[1].set_xticklabels(balanced_counts.index)
            
            # Log scale comparison (better for extreme imbalance)
            axes[2].bar(range(len(orig_counts)), orig_counts.values, alpha=0.6, color='red', label='Original')
            axes[2].bar(range(len(balanced_counts)), balanced_counts.values, alpha=0.6, color='green', label='Balanced')
            axes[2].set_yscale('log')
            axes[2].set_title('Log Scale Comparison')
            axes[2].set_xlabel('Class')
            axes[2].set_ylabel('Count (log scale)')
            axes[2].legend()
            
            # Ratio comparison
            orig_ratio = orig_counts.max() / orig_counts.min()
            balanced_ratio = balanced_counts.max() / balanced_counts.min()
            axes[3].bar(['Original', 'Balanced'], [orig_ratio, balanced_ratio], 
                       color=['red', 'green'], alpha=0.7)
            axes[3].set_title('Imbalance Ratio Comparison')
            axes[3].set_ylabel('Ratio (Major:Minor)')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'balance_comparison_{method}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Balance visualization saved for {method}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create balance visualization: {e}")

    def get_available_methods(self) -> List[str]:
        """Get available balancing methods"""
        return [method for method, info in self.balancing_methods.items() 
                if info.get('available', True)]

    def get_method_info(self, method: str) -> Dict:
        """Get method information"""
        return self.balancing_methods.get(method, {})

    def get_balancing_methods(self) -> Dict:
        """Get all balancing methods info for compatibility"""
        return self.balancing_methods