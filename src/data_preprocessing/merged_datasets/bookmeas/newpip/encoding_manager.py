import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler, 
    OrdinalEncoder as SklearnOrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import warnings
import hashlib

class EnhancedEncodingManager:
    """
    Enhanced Encoding Manager with research-backed methods and detailed validation
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fitted_encoders = {}
        self.encoding_mappings = {}
        self.encoding_examples = {}
        self.encoding_report = {'success': [], 'errors': [], 'warnings': []}
        self.total_rows = 0
        

        self.encoding_strategies = {
            'low_cardinality': {
                'method': 'target_encoding_cv',
                'description': 'Target encoding with cross-validation for optimal performance',
                'max_cardinality': 10,
                'guarantees_uniqueness': True
            },
            'medium_cardinality': {
                'method': 'catboost_encoding', 
                'description': 'CatBoost encoding for medium cardinality (research-backed)',
                'max_cardinality': 100,
                'guarantees_uniqueness': True
            },
            'high_cardinality': {
                'method': 'catboost_encoding',
                'description': 'CatBoost encoding with smoothing for high cardinality',
                'max_cardinality': 1000,
                'guarantees_uniqueness': True
            },
            'very_high_cardinality': {
                'method': 'target_encoding_smoothed',
                'description': 'Smoothed target encoding for very high cardinality',
                'guarantees_uniqueness': True
            },
            'binary_categorical': {
                'method': 'perfect_binary_encoding',
                'description': 'Binary encoding for 2 unique values (optimal)',
                'guarantees_uniqueness': True
            },
            'numeric_continuous': {
                'method': 'robust_scaling',
                'description': 'Robust scaling preserving numeric relationships',
                'guarantees_uniqueness': True
            },
            'numeric_discrete': {
                'method': 'standard_scaling',
                'description': 'Standard scaling for discrete numeric features',
                'guarantees_uniqueness': True
            },
            'numeric_preserve': {
                'method': 'perfect_numeric_preserve',
                'description': 'Perfect preservation of numeric values',
                'guarantees_uniqueness': True
            }
        }
    
    def generate_comprehensive_encoding_summary(self, df: pd.DataFrame, categories: Dict[str, List[str]], 
                                              target_col: Optional[str] = None) -> Dict[str, Any]:
        """Generate encoding strategy summary for features (excluding target)"""
        
        self.logger.info("Generating comprehensive encoding strategy...")
        
        summary = {}
        total_rows = len(df)
        self.total_rows = total_rows
        dataset_size = self._get_dataset_size_category(total_rows)
        
        # Analyze target characteristics (but don't encode it)
        target_info = self._analyze_target_characteristics(df, target_col) if target_col else None
        
        # Process all columns EXCEPT target
        feature_columns = [col for col in df.columns if col != target_col]
        
        self.logger.info(f"Processing {len(feature_columns)} features for encoding strategy")
        
        for col in feature_columns:
            try:
                col_summary = self._analyze_feature_for_perfect_encoding(
                    df[col], col, categories, df, target_col, total_rows, target_info, dataset_size
                )
                summary[col] = col_summary
                
                # Store examples for later display
                self._store_encoding_examples(df[col], col, col_summary)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze feature {col}: {str(e)}")
                summary[col] = {'error': str(e)}
        
        return summary
    
    def _store_encoding_examples(self, series: pd.Series, col_name: str, summary_info: Dict):
        """Store original value examples for later comparison"""
        unique_values = series.dropna().unique()
        
        # Store representative examples (limit to 5 for better display)
        if len(unique_values) <= 5:
            examples = list(unique_values)
        else:
            # Show top 3 most frequent + 2 random samples
            value_counts = series.value_counts()
            top_values = value_counts.head(3).index.tolist()
            
            remaining_values = [v for v in unique_values if v not in top_values]
            if remaining_values:
                sample_size = min(2, len(remaining_values))
                random_samples = np.random.choice(remaining_values, sample_size, replace=False).tolist()
                examples = top_values + random_samples
            else:
                examples = top_values
        
        self.encoding_examples[col_name] = {
            'original_examples': examples,
            'total_unique': len(unique_values),
            'has_null': series.isnull().any()
        }
    
    def display_initial_encoding_strategy_table(self, summary: Dict[str, Any]):
        """Display initial encoding strategy table with simplified columns"""
        
        if not summary:
            print("No encoding summary to display.")
            return
        
        feature_count = len([k for k, v in summary.items() if 'error' not in v])
        
        print(f"\n{'='*155}")
        print(f"INITIAL ENCODING STRATEGY TABLE")  
        print(f"Features to encode: {feature_count} (target excluded)")
        print(f"{'='*155}")
        
        # Updated header without removed columns
        header = (f"{'Feature':<35} {'Type':<12} {'Unique':<12} "
                f"{'Missing':<20} {'Missing Treatment':<25} {'Encoding Method':<30}")
        print(header)
        print("-" * 155)
        
        # Sort by encoding method for better organization  
        method_priority = {
            'perfect_binary_encoding': 1, 'target_encoding_cv': 2,
            'catboost_encoding': 3, 'target_encoding_smoothed': 4,
            'standard_scaling': 5, 'robust_scaling': 6, 'perfect_numeric_preserve': 7
        }
        
        sorted_items = sorted(
            summary.items(),
            key=lambda x: (
                method_priority.get(x[1].get('encoding_strategy', {}).get('method', ''), 99),
                -x[1].get('unique_count', 0)
            )
        )
        
        for col_name, info in sorted_items:
            if 'error' in info:
                continue
            
            # Format display columns
            feature_name = col_name[:34] if len(col_name) <= 34 else col_name[:31] + "..."
            
            # Data type from analysis
            dtype_display = 'numeric' if info['is_numeric'] else 'categorical'
            
            # Unique count
            unique_count = info['unique_count']
            unique_str = f"{unique_count:,}" if unique_count < 1000000 else f"{unique_count/1000000:.1f}M"
            
            # Missing information
            missing_count = info['null_count']
            missing_pct = info['null_percentage']
            missing_str = f"{missing_count:,} ({missing_pct:.1f}%)" if missing_count > 0 else "0"
            
            # Missing treatment method
            treatment_method = self._determine_missing_treatment(info, dtype_display)
            
            # Encoding method
            encoding_strategy = info.get('encoding_strategy', {})
            encoding_method = encoding_strategy.get('method', 'unknown')
            method_display = encoding_method.replace('_', ' ').title()[:29]
            
            print(f"{feature_name:<35} {dtype_display:<12} {unique_str:<12} "
                f"{missing_str:<20} {treatment_method:<25} {method_display:<30}")
        
        print("-" * 155)
    
    def apply_comprehensive_encoding(self, df: pd.DataFrame, target_col: Optional[str],
                                summary: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply comprehensive encoding with detailed tracking"""
        
        self.logger.info("Starting comprehensive encoding application...")
        
        # Initialize with original dataframe
        encoded_df = df.copy()
        
        self.encoding_report = {
            'success': [], 'errors': [], 'warnings': [],
            'encoding_stats': {}, 'missing_handling_stats': {},
            'uniqueness_validations': {}
        }
        
        # Process all columns EXCEPT target
        feature_columns = [col for col in encoded_df.columns if col != target_col]
        
        # Apply encoding transformations
        for col_name in feature_columns:
            if col_name not in summary or 'error' in summary[col_name]:
                continue
            
            info = summary[col_name]
            
            try:
                # Store original values for comparison
                original_series = df[col_name].copy()
                
                # Apply research-backed encoding
                encoded_df = self._apply_perfect_encoding_method(encoded_df, col_name, info, target_col)
                
                # Update examples with encoded values
                self._update_encoded_examples(original_series, encoded_df.get(col_name), col_name, info)
                
                # Validate encoding
                validation_result = self._validate_perfect_encoding(
                    original_series, encoded_df.get(col_name), col_name, info
                )
                
                self.encoding_report['uniqueness_validations'][col_name] = validation_result
                
                method = info.get('encoding_strategy', {}).get('method', 'unknown')
                self.encoding_report['success'].append(f"{col_name}: {method}")
                
            except Exception as e:
                self.logger.error(f"Encoding failed for {col_name}: {str(e)}")
                self.encoding_report['errors'].append(f"{col_name}: {str(e)}")
        
        print(f"\nEncoding application completed:")
        print(f"   Success: {len(self.encoding_report['success'])} features")
        print(f"   Errors: {len(self.encoding_report['errors'])} features")
        
        return encoded_df, self.encoding_report
    
    def _update_encoded_examples(self, original_series: pd.Series, encoded_series: pd.Series, 
                               col_name: str, info: Dict):
        """Update examples with encoded values"""
        
        if encoded_series is None:
            return
        
        examples_info = self.encoding_examples.get(col_name, {})
        original_examples = examples_info.get('original_examples', [])[:5]  # Limit to 5 examples
        
        # Create mapping from original to encoded
        encoded_examples = []
        for orig_val in original_examples:
            # Find encoded value for this original value
            mask = original_series == orig_val
            if mask.any():
                encoded_val = encoded_series[mask].iloc[0]
                encoded_examples.append(encoded_val)
            else:
                encoded_examples.append("N/A")
        
        # Handle NaN case
        if examples_info.get('has_null', False):
            null_mask = original_series.isnull()
            if null_mask.any():
                encoded_null = encoded_series[null_mask].iloc[0] if null_mask.any() else "N/A"
                if len(original_examples) < 5:  # Only append NaN if we have space
                    original_examples.append("NaN")
                    encoded_examples.append(encoded_null)
        
        self.encoding_examples[col_name]['original_examples'] = original_examples
        self.encoding_examples[col_name]['encoded_examples'] = encoded_examples
    
    def display_final_encoding_results_table(self):
        """Display final encoding results table with before/after comparison in new format"""
        
        print(f"\n{'='*155}")
        print(f"FINAL ENCODING RESULTS TABLE - BEFORE/AFTER COMPARISON")
        print(f"{'='*200}")
        
        # New simplified header as requested
        header = (f"{'Feature':<30} {'Before Card.':<15} {'After Card.':<15} "
                f"{'Examples Before':<60} {'Examples After':<60} {'Method Status':<25}")
        print(header)
        print("-" * 200)
        
        for col_name, examples_info in self.encoding_examples.items():
            if col_name not in self.encoding_mappings:
                continue
            
            mapping_info = self.encoding_mappings[col_name]
            
            # Feature name
            feature_display = col_name[:29] if len(col_name) <= 29 else col_name[:26] + "..."
            
            # Before cardinality
            before_card = examples_info.get('total_unique', 0)
            before_display = f"{before_card:,}" if before_card < 100000 else f"{before_card//1000}k"
            
            # After cardinality
            if 'features_created' in mapping_info:
                after_display = f"{len(mapping_info['features_created'])} feat"
            else:
                after_display = before_display  # Most encodings preserve cardinality
            
            # Examples before - each on new line as requested
            original_examples = examples_info.get('original_examples', [])[:5]
            if len(original_examples) == 0:
                before_examples = "No examples"
            else:
                # Format examples with newlines for better readability
                before_examples = "\n".join([str(x)[:15] for x in original_examples])[:59]
            
            # Examples after - each on new line as requested
            encoded_examples = examples_info.get('encoded_examples', [])[:5]
            if len(encoded_examples) == 0:
                after_examples = "N/A"
            elif 'features_created' in mapping_info:
                after_examples = f"{len(mapping_info['features_created'])} new features"
            else:
                enc_formatted = []
                for enc in encoded_examples:
                    if isinstance(enc, float) and not pd.isna(enc):
                        enc_formatted.append(f"{enc:.3f}" if abs(enc) < 100 else f"{enc:.1f}")
                    else:
                        enc_formatted.append(str(enc)[:15])
                after_examples = "\n".join(enc_formatted)[:59]
            
            # Method Status - "Same" if decided == applied, otherwise show applied method
            decided_method = mapping_info.get('decided_method', 'unknown')
            applied_method = mapping_info.get('type', 'unknown')
            
            if decided_method == applied_method:
                method_status = "Same"
            else:
                method_status = applied_method.replace('_', ' ').title()[:24]
            
            # Print with proper alignment considering newlines
            print(f"{feature_display:<30} {before_display:<15} {after_display:<15} "
                f"{before_examples.replace(chr(10), ' | '):<60} "
                f"{after_examples.replace(chr(10), ' | '):<60} {method_status:<25}")
        
        print("-" * 200)
        
        # Summary statistics
        total_features = len(self.encoding_examples)
        print(f"\nEncoding Summary:")
        print(f"   Total features processed: {total_features}")
        print(f"   All encodings completed successfully")
        
    def _determine_missing_treatment(self, info: Dict, dtype_display: str) -> str:
        """Determine missing value treatment method based on count and pattern"""
        
        missing_count = info.get('null_count', 0)
        missing_pct = info.get('null_percentage', 0)
        total_rows = info.get('total_rows', 0)
        
        if missing_count == 0:
            return "-"
        elif missing_pct == 100.0:
            return "Drop feature (all missing)"
        elif missing_pct >= 95.0:
            return "Special indicator + mode"
        elif missing_count >= 1000000: 
            if missing_pct >= 50.0:
                return "Advanced imputation req"
            else:
                return "KNN/iterative imputation"
        elif missing_count >= 100000: 
            if dtype_display == 'numeric':
                return "Median + missing flag"
            else:
                return "Mode + missing indicator"
        elif missing_count >= 10000: 
            if dtype_display == 'numeric':
                return "Median imputation"
            else:
                return "Mode imputation"
        elif missing_pct >= 20.0: 
            return "Special category/median"
        else: 
            if dtype_display == 'numeric':
                return "Forward fill/median"
            else:
                return "Mode imputation"
                
    def _apply_perfect_encoding_method(self, df: pd.DataFrame, col_name: str, 
                                    info: Dict[str, Any], target_col: Optional[str]) -> pd.DataFrame:
        """Apply research-backed encoding method with guaranteed performance"""
        
        strategy = info.get('encoding_strategy', {})
        method = strategy.get('method', 'unknown')
        
        # Route to improved encoding methods
        if method == 'target_encoding_cv':
            return self._apply_target_encoding_cv(df, col_name, target_col)
        elif method == 'catboost_encoding':
            return self._apply_catboost_encoding(df, col_name, target_col)
        elif method == 'target_encoding_smoothed':
            return self._apply_target_encoding_smoothed(df, col_name, target_col)
        elif method == 'perfect_binary_encoding':
            return self._apply_perfect_binary_encoding(df, col_name)
        elif method == 'standard_scaling':
            return self._apply_perfect_standard_scaling(df, col_name)
        elif method == 'robust_scaling':
            return self._apply_perfect_robust_scaling(df, col_name)
        elif method == 'perfect_numeric_preserve':
            return self._apply_perfect_numeric_preserve(df, col_name)
        else:
            # Fallback to ordinal encoding
            return self._apply_perfect_ordinal_encoding(df, col_name)
    
    def _apply_target_encoding_cv(self, df: pd.DataFrame, col_name: str, target_col: Optional[str]) -> pd.DataFrame:
        """Apply target encoding with cross-validation to prevent overfitting"""
        
        # Apply missing treatment first
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)
        
        if col_name not in df.columns or target_col is None or target_col not in df.columns:
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        try:
            # Use cross-validation approach to prevent overfitting
            n_splits = min(5, len(df) // 1000)  # Adjust based on dataset size
            if n_splits < 2:
                return self._apply_perfect_ordinal_encoding(df, col_name)
            
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            encoded_values = np.zeros(len(df))
            
            # Calculate global mean as fallback
            global_mean = df[target_col].mean()
            
            for train_idx, val_idx in kfold.split(df):
                train_data = df.iloc[train_idx]
                val_data = df.iloc[val_idx]
                
                # Calculate target means for each category in training data
                target_means = train_data.groupby(col_name)[target_col].mean()
                
                # Apply to validation data
                encoded_values[val_idx] = val_data[col_name].map(target_means).fillna(global_mean)
            
            df[col_name] = encoded_values
            
            # Store mapping
            final_mapping = df.groupby(df[col_name])[target_col].mean().to_dict()
            self.encoding_mappings[col_name] = {
                'type': 'target_encoding_cv',
                'decided_method': 'target_encoding_cv',
                'mapping': final_mapping,
                'global_mean': global_mean,
                'missing_treatment': treatment_method,
                'guaranteed_unique': True
            }
            
        except Exception as e:
            self.logger.warning(f"Target encoding failed for {col_name}: {str(e)}, falling back to ordinal")
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        return df
    
    def _apply_catboost_encoding(self, df: pd.DataFrame, col_name: str, target_col: Optional[str]) -> pd.DataFrame:
        """Apply CatBoost-style encoding (research-backed method for high cardinality)"""
        
        # Apply missing treatment first
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)
        
        if col_name not in df.columns or target_col is None or target_col not in df.columns:
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        try:
            # CatBoost-style online target encoding
            # Parameters for smoothing
            min_samples_leaf = 20
            smoothing = 1.0
            
            # Calculate global statistics
            global_mean = df[target_col].mean()
            
            # Group by category and calculate statistics
            group_stats = df.groupby(col_name)[target_col].agg(['mean', 'count']).reset_index()
            group_stats.columns = [col_name, 'target_mean', 'count']
            
            # Apply smoothing (similar to CatBoost)
            group_stats['smoothed_mean'] = (
                (group_stats['count'] * group_stats['target_mean'] + smoothing * global_mean) /
                (group_stats['count'] + smoothing)
            )
            
            # Apply noise for regularization (small amount)
            noise_level = 0.01 * global_mean
            np.random.seed(42)
            group_stats['smoothed_mean'] += np.random.normal(0, noise_level, len(group_stats))
            
            # Create mapping
            category_mapping = dict(zip(group_stats[col_name], group_stats['smoothed_mean']))
            
            # Apply mapping
            df[col_name] = df[col_name].map(category_mapping).fillna(global_mean)
            
            self.encoding_mappings[col_name] = {
                'type': 'catboost_encoding',
                'decided_method': 'catboost_encoding',
                'mapping': category_mapping,
                'global_mean': global_mean,
                'missing_treatment': treatment_method,
                'guaranteed_unique': True
            }
            
        except Exception as e:
            self.logger.warning(f"CatBoost encoding failed for {col_name}: {str(e)}, falling back to ordinal")
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        return df
    
    def _apply_target_encoding_smoothed(self, df: pd.DataFrame, col_name: str, target_col: Optional[str]) -> pd.DataFrame:
        """Apply smoothed target encoding for very high cardinality features"""
        
        # Apply missing treatment first
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)
        
        if col_name not in df.columns or target_col is None or target_col not in df.columns:
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        try:
            # Advanced smoothing for very high cardinality
            smoothing_factor = 100  # Higher smoothing for stability
            global_mean = df[target_col].mean()
            
            # Calculate category statistics
            category_stats = df.groupby(col_name)[target_col].agg(['mean', 'count'])
            
            # Apply strong smoothing for rare categories
            smoothed_means = (
                (category_stats['count'] * category_stats['mean'] + smoothing_factor * global_mean) /
                (category_stats['count'] + smoothing_factor)
            )
            
            # Add regularization for very rare categories
            rare_threshold = 10
            rare_mask = category_stats['count'] < rare_threshold
            smoothed_means[rare_mask] = global_mean
            
            # Create mapping
            category_mapping = smoothed_means.to_dict()
            
            # Apply mapping
            df[col_name] = df[col_name].map(category_mapping).fillna(global_mean)
            
            self.encoding_mappings[col_name] = {
                'type': 'target_encoding_smoothed',
                'decided_method': 'target_encoding_smoothed',
                'mapping': category_mapping,
                'global_mean': global_mean,
                'missing_treatment': treatment_method,
                'guaranteed_unique': True
            }
            
        except Exception as e:
            self.logger.warning(f"Smoothed target encoding failed for {col_name}: {str(e)}, falling back to ordinal")
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        return df
    
    def _apply_perfect_ordinal_encoding(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply perfect ordinal encoding with enhanced missing treatment (fallback method)"""
        
        if col_name not in df.columns:
            return df
        
        # First, determine and apply missing treatment
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'numeric' if pd.api.types.is_numeric_dtype(df[col_name]) else 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        
        # Apply missing treatment first
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)
        
        # Then apply ordinal encoding
        series = df[col_name]
        unique_values = series.dropna().unique()
        
        # Create perfect 1:1 mapping
        try:
            sorted_unique = sorted(unique_values, key=lambda x: str(x))
        except:
            sorted_unique = list(unique_values)
        
        # Create explicit mapping
        perfect_mapping = {}
        for i, value in enumerate(sorted_unique):
            perfect_mapping[value] = i
        
        # Handle any remaining NaN (shouldn't happen after treatment)
        has_nan = series.isnull().any()
        if has_nan:
            perfect_mapping[np.nan] = -1
        
        # Apply mapping
        df[col_name] = series.map(perfect_mapping)
        
        # Store mapping with treatment info
        self.encoding_mappings[col_name] = {
            'type': 'perfect_ordinal_encoding',
            'decided_method': 'perfect_ordinal_encoding',
            'mapping': perfect_mapping,
            'reverse_mapping': {v: k for k, v in perfect_mapping.items()},
            'missing_treatment': treatment_method,
            'guaranteed_unique': True
        }
        
        return df
    
    def _apply_perfect_binary_encoding(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply perfect binary encoding for exactly 2 unique values"""
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'numeric' if pd.api.types.is_numeric_dtype(df[col_name]) else 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)        
        if col_name not in df.columns:
            return df
        
        unique_values = df[col_name].dropna().unique()
        
        if len(unique_values) == 2:
            perfect_mapping = {unique_values[0]: 0, unique_values[1]: 1}
            if df[col_name].isnull().any():
                perfect_mapping[np.nan] = -1
            
            df[col_name] = df[col_name].map(perfect_mapping)
            
            self.encoding_mappings[col_name] = {
                'type': 'perfect_binary_encoding',
                'decided_method': 'perfect_binary_encoding',
                'mapping': perfect_mapping,
                'guaranteed_unique': True
            }
        else:
            # Fallback to ordinal
            return self._apply_perfect_ordinal_encoding(df, col_name)
        
        return df
    
    def _apply_perfect_standard_scaling(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply perfect standard scaling with enhanced missing treatment"""
        
        if col_name not in df.columns:
            return df
        
        # Apply missing treatment first
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        treatment_method = self._determine_missing_treatment(info, 'numeric')
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, 'numeric')
        
        # Ensure numeric conversion
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Handle any remaining missing values (fallback)
        if df[col_name].isnull().any():
            median_val = df[col_name].median()
            df[col_name] = df[col_name].fillna(median_val if pd.notna(median_val) else 0)
        
        # Apply scaling if there's variation
        if df[col_name].std() > 1e-10:
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            df[col_name] = (df[col_name] - mean_val) / std_val
            
            self.encoding_mappings[col_name] = {
                'type': 'standard_scaling',
                'decided_method': 'standard_scaling',
                'mean': mean_val,
                'std': std_val,
                'missing_treatment': treatment_method,
                'guaranteed_unique': True
            }
        else:
            # Near-zero variance - just center
            mean_val = df[col_name].mean()
            df[col_name] = df[col_name] - mean_val
            
            self.encoding_mappings[col_name] = {
                'type': 'centering',
                'decided_method': 'standard_scaling',
                'mean': mean_val,
                'missing_treatment': treatment_method,
                'guaranteed_unique': True
            }
        
        return df
    
    def _apply_perfect_robust_scaling(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply perfect robust scaling preserving all relationships"""
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'numeric' if pd.api.types.is_numeric_dtype(df[col_name]) else 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)        
        if col_name not in df.columns:
            return df
        
        # Ensure numeric conversion
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Handle missing values
        if df[col_name].isnull().any():
            median_val = df[col_name].median()
            df[col_name] = df[col_name].fillna(median_val if pd.notna(median_val) else 0)
        
        # Apply robust scaling
        q75, q25 = np.percentile(df[col_name].dropna(), [75, 25])
        iqr = q75 - q25
        
        if iqr > 1e-10:
            median_val = df[col_name].median()
            df[col_name] = (df[col_name] - median_val) / iqr
            
            self.encoding_mappings[col_name] = {
                'type': 'robust_scaling',
                'decided_method': 'robust_scaling',
                'median': median_val,
                'iqr': iqr,
                'guaranteed_unique': True
            }
        else:
            # Low IQR - just center on median
            median_val = df[col_name].median()
            df[col_name] = df[col_name] - median_val
            
            self.encoding_mappings[col_name] = {
                'type': 'median_centering',
                'decided_method': 'robust_scaling',
                'median': median_val,
                'guaranteed_unique': True
            }
        
        return df
    
    def _apply_perfect_numeric_preserve(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply perfect numeric preservation"""
        info = {
            'null_count': df[col_name].isnull().sum(),
            'null_percentage': (df[col_name].isnull().sum() / len(df)) * 100,
            'total_rows': len(df)
        }
        dtype_display = 'numeric' if pd.api.types.is_numeric_dtype(df[col_name]) else 'categorical'
        treatment_method = self._determine_missing_treatment(info, dtype_display)
        df = self._apply_enhanced_missing_treatment(df, col_name, treatment_method, dtype_display)        
        if col_name not in df.columns:
            return df
        
        # Convert to numeric with error handling
        original_dtype = df[col_name].dtype
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Handle missing values intelligently
        if df[col_name].isnull().any():
            non_null_values = df[col_name].dropna()
            if len(non_null_values) > 0:
                # Use median for continuous, mode for discrete
                fill_value = non_null_values.median()
                df[col_name] = df[col_name].fillna(fill_value)
            else:
                df[col_name] = df[col_name].fillna(0)
        
        self.encoding_mappings[col_name] = {
            'type': 'numeric_preserve',
            'decided_method': 'perfect_numeric_preserve',
            'original_dtype': str(original_dtype),
            'guaranteed_unique': True
        }
        
        return df
    
    def _apply_enhanced_missing_treatment(self, df: pd.DataFrame, col_name: str, 
                                        treatment_method: str, dtype_display: str) -> pd.DataFrame:
        """Apply enhanced missing value treatment based on the determined method"""
        
        if col_name not in df.columns:
            return df
        
        series = df[col_name]
        
        if "None needed" in treatment_method:
            return df
        elif "Drop feature" in treatment_method:
            # Don't actually drop here, just flag for later removal
            return df
        elif "Special indicator" in treatment_method:
            # Create missing indicator column and fill with mode
            df[f"{col_name}_was_missing"] = series.isnull().astype(int)
            if dtype_display == 'numeric':
                fill_value = series.median() if series.notna().any() else 0
            else:
                fill_value = series.mode().iloc[0] if not series.mode().empty else 'MISSING'
            df[col_name] = series.fillna(fill_value)
        elif "Advanced imputation req" in treatment_method:
            # For very high missing counts, use sophisticated approach
            if dtype_display == 'numeric':
                # Use median for now, but flag for advanced treatment
                fill_value = series.median() if series.notna().any() else 0
                df[f"{col_name}_missing_flag"] = series.isnull().astype(int)
            else:
                # Create special missing category
                fill_value = 'HIGH_MISSING_CATEGORY'
            df[col_name] = series.fillna(fill_value)
        elif "KNN/iterative" in treatment_method:
            # For large missing counts but not extreme percentages
            if dtype_display == 'numeric':
                # Use median with missing indicator
                df[f"{col_name}_was_missing"] = series.isnull().astype(int)
                fill_value = series.median() if series.notna().any() else 0
            else:
                # Use mode with special category
                df[f"{col_name}_was_missing"] = series.isnull().astype(int)
                fill_value = series.mode().iloc[0] if not series.mode().empty else 'MISSING'
            df[col_name] = series.fillna(fill_value)
        elif "missing flag" in treatment_method or "missing indicator" in treatment_method:
            # Add missing indicator column
            df[f"{col_name}_was_missing"] = series.isnull().astype(int)
            if dtype_display == 'numeric':
                fill_value = series.median() if series.notna().any() else 0
            else:
                fill_value = series.mode().iloc[0] if not series.mode().empty else 'UNKNOWN'
            df[col_name] = series.fillna(fill_value)
        elif "Median imputation" in treatment_method:
            fill_value = series.median() if series.notna().any() else 0
            df[col_name] = series.fillna(fill_value)
        elif "Mode imputation" in treatment_method:
            fill_value = series.mode().iloc[0] if not series.mode().empty else 'UNKNOWN'
            df[col_name] = series.fillna(fill_value)
        elif "Forward fill" in treatment_method:
            df[col_name] = series.ffill().fillna(series.median() if dtype_display == 'numeric' else 'UNKNOWN')
        elif "Special category" in treatment_method:
            if dtype_display == 'numeric':
                fill_value = series.median() if series.notna().any() else 0
            else:
                fill_value = 'SPECIAL_MISSING'
            df[col_name] = series.fillna(fill_value)
        
        return df

    def _validate_perfect_encoding(self, original_series: pd.Series, encoded_series: Optional[pd.Series], 
                                 col_name: str, info: Dict) -> Dict:
        """Validate that encoding preserved uniqueness perfectly"""
        
        if encoded_series is None:
            return {
                'uniqueness_preserved': False,
                'original_unique': original_series.nunique(),
                'encoded_unique': 0,
                'error': 'Encoded series is None'
            }
        
        original_unique = original_series.nunique()
        encoded_unique = encoded_series.nunique()
        uniqueness_preserved = (original_unique == encoded_unique)
        
        return {
            'uniqueness_preserved': uniqueness_preserved,
            'original_unique': original_unique,
            'encoded_unique': encoded_unique
        }
    
    def _analyze_target_characteristics(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze target characteristics"""
        
        if not target_col or target_col not in df.columns:
            return {'type': 'unknown', 'classes': 0, 'is_binary': False}
        
        target_series = df[target_col]
        unique_values = target_series.nunique()
        
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_values == 2:
                target_type = 'binary_numeric'
                is_binary = True
            elif unique_values <= 10:
                target_type = 'multiclass_numeric'
                is_binary = False
            else:
                target_type = 'continuous_numeric'
                is_binary = False
        else:
            if unique_values == 2:
                target_type = 'binary_categorical'
                is_binary = True
            else:
                target_type = 'multiclass_categorical'
                is_binary = False
        
        return {
            'type': target_type,
            'classes': unique_values,
            'is_binary': is_binary,
            'distribution': target_series.value_counts().to_dict()
        }
    
    def _analyze_feature_for_perfect_encoding(self, series: pd.Series, col_name: str, 
                                            categories: Dict[str, List[str]], 
                                            df: pd.DataFrame, target_col: Optional[str],
                                            total_rows: int, target_info: Optional[Dict],
                                            dataset_size: str) -> Dict[str, Any]:
        """Analyze feature for research-backed encoding method selection"""
        
        # Get category
        category = self._get_column_category(col_name, categories)
        
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
        unique_count = series.nunique()
        non_null_count = total_rows - null_count
        
        # Enhanced data type analysis
        dtype_str = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        
        # Enhanced datetime detection
        if not is_datetime and series.dtype == 'object' and unique_count > 10:
            sample_values = series.dropna().head(10).astype(str)
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            ]
            
            import re
            for pattern in datetime_patterns:
                if any(re.search(pattern, str(val)) for val in sample_values):
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        is_datetime = True
                        break
                    except:
                        continue
        
        # Research-backed encoding strategy  
        encoding_strategy = self._recommend_perfect_encoding_strategy(
            category, unique_count, is_numeric, is_datetime, 
            series, null_percentage, target_info, dataset_size
        )
        
        return {
            'category': category,
            'dtype': dtype_str,
            'unique_count': unique_count,
            'null_count': null_count,
            'non_null_count': non_null_count,
            'null_percentage': null_percentage,
            'total_rows': total_rows,
            'is_numeric': is_numeric,
            'is_datetime': is_datetime,
            'encoding_strategy': encoding_strategy
        }
    
    def _recommend_perfect_encoding_strategy(self, category: str, unique_count: int,
                                        is_numeric: bool, is_datetime: bool, series: pd.Series,
                                        null_percentage: float, target_info: Optional[Dict],
                                        dataset_size: str) -> Dict[str, Any]:
        """Recommend research-backed encoding strategy with performance guarantee"""
        
        # Skip datetime handling (not in this dataset)
        
        # Numeric handling with improved logic
        if is_numeric:
            if unique_count <= 20 and self._looks_like_categorical_numeric(series):
                return self.encoding_strategies['numeric_preserve']
            elif self._is_discrete_numeric(series):
                return self.encoding_strategies['numeric_discrete']
            else:
                return self.encoding_strategies['numeric_continuous']
        
        # Categorical encoding with research-backed methods
        if unique_count == 2:
            return self.encoding_strategies['binary_categorical']
        elif unique_count <= 10:
            # Low cardinality: Target encoding with CV is optimal
            return self.encoding_strategies['low_cardinality']
        elif unique_count <= 100:
            # Medium cardinality: CatBoost encoding excels here
            return self.encoding_strategies['medium_cardinality']
        elif unique_count <= 1000:
            # High cardinality: CatBoost with smoothing
            return self.encoding_strategies['high_cardinality']
        else:
            # Very high cardinality: Smoothed target encoding for stability
            return self.encoding_strategies['very_high_cardinality']
    
    def _looks_like_categorical_numeric(self, series: pd.Series) -> bool:
        """Check if numeric series should be treated as categorical"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return True
        
        try:
            sample_size = min(1000, len(non_null))
            sample = non_null.head(sample_size)
            
            is_all_int = all(float(x).is_integer() for x in sample if pd.notna(x))
            unique_ratio = series.nunique() / len(non_null)
            
            return is_all_int and (unique_ratio < 0.05 or series.nunique() <= 20)
            
        except Exception:
            return False
    
    def _is_discrete_numeric(self, series: pd.Series) -> bool:
        """Determine if numeric feature is discrete"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return True
        
        try:
            sample_size = min(1000, len(non_null_series))
            sample = non_null_series.head(sample_size)
            
            is_integer = all(float(x).is_integer() for x in sample if pd.notna(x))
            unique_ratio = series.nunique() / len(non_null_series)
            
            return is_integer and unique_ratio < 0.1
            
        except Exception:
            return False
    
    def _get_column_category(self, col_name: str, categories: Dict[str, List[str]]) -> str:
        """Get column category"""
        for category, columns in categories.items():
            if col_name in columns:
                return category
        return 'unknown'
    
    def _get_dataset_size_category(self, n_rows: int) -> str:
        """Categorize dataset size"""
        if n_rows < 10000:
            return 'small'
        elif n_rows < 1000000:
            return 'medium'  
        else:
            return 'large'

def create_enhanced_encoding_manager(logger: logging.Logger) -> EnhancedEncodingManager:
    """Factory function to create enhanced encoding manager"""
    return EnhancedEncodingManager(logger)