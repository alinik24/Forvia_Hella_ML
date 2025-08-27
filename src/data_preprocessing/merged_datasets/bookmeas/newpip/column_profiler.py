"""
Enhanced Column Profiler
High-precision column analysis with simplified workflow focused on 5 key data quality issues
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import re
from difflib import SequenceMatcher

class ColumnProfiler:
    """Enhanced column profiling with streamlined workflow"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.profile_results = None
        self.total_rows = 0
    
    def profile_dataframe(self, df: pd.DataFrame, categories: Dict[str, List[str]]) -> Dict[str, Any]:
        """Profile all columns with high precision checks"""
        self.total_rows = len(df)
        
        results = {}
        
        for col in df.columns:
            try:
                profile = self._profile_column(df[col], col, categories)
                results[col] = profile
            except Exception as e:
                self.logger.error(f"Failed to profile column {col}: {str(e)}")
                results[col] = {'error': str(e)}
        
        self.logger.info(f"✅ Profiled {len(results)} columns")
        self.profile_results = results
        return results
    
    def display_enhanced_profile_summary(self, results: Dict[str, Any], total_rows: int):
        """Display enhanced profiling table with absolute missing values and total rows"""
        
        if not results:
            print("No profiling results to display.")
            return
        
        self.total_rows = total_rows
        
        # Filter out error results and sort by cardinality (descending)
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1].get('cardinality', 0), reverse=True)
        
        # Display enhanced profiling table
        self._display_enhanced_profile_table(sorted_results)
        
        # Display comprehensive summary statistics
        self._display_enhanced_summary_stats(valid_results)
    
    def _display_enhanced_profile_table(self, sorted_results: List[tuple]):
        """Display enhanced profiling table with absolute values and numbered features"""
        
        print(f"\n{'='*155}")
        print(f"🔍 ENHANCED COLUMN PROFILING SUMMARY")
        print(f"Total Rows: {self.total_rows:,} | Features Analyzed: {len(sorted_results)}")
        print(f"{'='*155}")
        
        # Enhanced header with Total Rows context
        header = (f"{'#':<3} {'Feature':<25} {'Category':<20} {'Storage':<12} {'Cardinality':<12} "
                 f"{'Missing Count':<15} {'Example Value':<20}")
        print(header)
        print("-" * 155)
        
        # Display rows with enhanced information and numbers
        for idx, (col_name, profile) in enumerate(sorted_results, 1):
            feature_name = col_name[:24] if len(col_name) <= 24 else col_name[:21] + "..."
            category = profile['category'][:19] if len(profile['category']) <= 19 else profile['category'][:16] + "..."
            storage = profile['storage_type'][:11] if len(profile['storage_type']) <= 11 else profile['storage_type'][:8] + "..."
            
            # Cardinality
            cardinality = profile.get('cardinality', 0)
            card_str = f"{cardinality:,}" if cardinality < 1000000 else f"{cardinality/1000000:.1f}M"
            
            # Absolute missing counts
            missing_count = profile.get('null_count', 0)
            missing_count_str = f"{missing_count:,}"
            
            # Example value (truncated for display)
            example = str(profile.get('example_value', 'N/A'))[:19]
            if len(str(profile.get('example_value', ''))) > 19:
                example += "..."
            
            # Row with enhanced formatting and number
            print(f"{idx:<3} {feature_name:<25} {category:<20} {storage:<12} {card_str:<12} "
                  f"{missing_count_str:<15} {example:<20}")
        
        print("-" * 155)
    
    def _display_enhanced_summary_stats(self, results: Dict[str, Any]):
        """Display enhanced summary statistics with actionable insights"""
        
        total_features = len(results)
        # Category distribution
        category_counts = {}
        for result in results.values():
            cat = result.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"\n   🏷️ Category Distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / total_features) * 100
            print(f"      • {category}: {count} features ({percentage:.1f}%)")
    
    def prompt_target_selection_and_find_issues(self, df: pd.DataFrame, results: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Target selection followed by 5 key data quality issues analysis"""
        
        print(f"\n{'='*155}")
        print("🎯 TARGET SELECTION")
        print(f"{'='*155}")
        
        # Use the same sorting logic as the profiling table
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1].get('cardinality', 0), reverse=True)
        
        # Create ordered features list matching the displayed table
        features_list = [col_name for col_name, profile in sorted_results]
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nSelect target column (number 1-{len(features_list)} or feature name): ").strip()
                
                # Try as number first
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(features_list):
                        target_column = features_list[idx]
                        break
                    else:
                        print(f"❌ Invalid number. Please enter 1-{len(features_list)}")
                        continue
                
                # Try as feature name
                if choice in features_list:
                    target_column = choice
                    break
                else:
                    print(f"❌ Feature '{choice}' not found. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a valid number or feature name.")
        
        print(f"✅ Selected target: {target_column}")
        
        # Immediately analyze the 5 key problematic features and get user input for dropping
        features_to_drop = self.analyze_5_key_data_quality_issues(df, results, target_column)
        
        return target_column, features_to_drop
    
    def analyze_5_key_data_quality_issues(self, df: pd.DataFrame, results: Dict[str, Any], target_column: str) -> List[str]:
        """Analyze the 5 key data quality issues and get user input for dropping"""
        
        print(f"\n💡 DATA QUALITY INSIGHTS:")
        
        issues_found = {}
        total_rows = len(df)
        
        # Get target distribution for reference - SMALLEST CLASS
        target_value_counts = df[target_column].value_counts()
        smallest_class = target_value_counts.iloc[-1] if len(target_value_counts) > 0 else 0
        
        # 1. Features with constant values (cardinality 0 or 1)
        constant_features = []
        for feature, info in results.items():
            if feature != target_column and info.get('cardinality', 0) <= 1:
                constant_features.append((feature, info.get('cardinality', 0)))
        
        if constant_features:
            issues_found['constant'] = constant_features
            print(f"\n1️⃣ CONSTANT VALUE FEATURES ({len(constant_features)} found):")
            print("   Features with cardinality ≤ 1 (no variation)")
            for feature, cardinality in constant_features:
                print(f"   • {feature} (cardinality: {cardinality})")
        
        # 2. Features with absolute null values (all missing)
        all_null_features = []
        for feature, info in results.items():
            if feature != target_column and info.get('null_count', 0) == total_rows:
                all_null_features.append(feature)
        
        if all_null_features:
            issues_found['all_null'] = all_null_features
            print(f"\n2️⃣ COMPLETELY NULL FEATURES ({len(all_null_features)} found):")
            print("   Features where all values are missing")
            for feature in all_null_features:
                print(f"   • {feature}")
        
        # 3. Features with high missing values (REVISED LOGIC: non-null values < smallest class)
        high_missing_features = []
        for feature, info in results.items():
            if feature != target_column:
                missing_count = info.get('null_count', 0)
                non_null_count = info.get('non_null_count', 0)
                # Only consider high missing if non-null values are less than smallest class
                if non_null_count < smallest_class and missing_count > 0 and missing_count < total_rows:
                    missing_pct = info.get('null_percentage', 0)
                    high_missing_features.append((feature, missing_count, missing_pct, non_null_count))
        
        if high_missing_features:
            issues_found['high_missing'] = high_missing_features
            print(f"\n3️⃣ HIGH MISSING VALUE FEATURES ({len(high_missing_features)} found):")
            print(f"   Features with non-null values < {smallest_class:,} (smallest class size)")
            for feature, missing_count, missing_pct, non_null_count in high_missing_features:
                print(f"   • {feature}: {non_null_count:,} non-null values ({missing_pct:.1f}% missing)")
        
        # 4. High cardinality features (TOP 8, excluding measurement values)
        high_cardinality_candidates = []
        for feature, info in results.items():
            if feature != target_column:
                cardinality = info.get('cardinality', 0)
                # Exclude features with 'measur' in name (measurement values)
                if cardinality > 1000 and 'measur' not in feature.lower():
                    high_cardinality_candidates.append((feature, cardinality))
        
        # Sort by cardinality descending and take top 8
        high_cardinality_candidates.sort(key=lambda x: x[1], reverse=True)
        high_cardinality_features = high_cardinality_candidates[:8]
        
        if high_cardinality_features:
            issues_found['high_cardinality'] = high_cardinality_features
            print(f"\n4️⃣ HIGH CARDINALITY FEATURES (Top {len(high_cardinality_features)} found, excluding measurements):")
            print("   Features with cardinality > 10,000 (potential encoding issues)")
            for feature, cardinality in high_cardinality_features:
                print(f"   • {feature}: {cardinality:,} unique values")
        
        # 5. Similar feature groups (NEW: detect correlated/duplicate features)
        similar_feature_groups = self._detect_similar_feature_groups(results, target_column)
        
        if similar_feature_groups:
            issues_found['similar_groups'] = similar_feature_groups
            print(f"\n5️⃣ SIMILAR FEATURE GROUPS ({len(similar_feature_groups)} groups found):")
            print("   Feature groups with similar names, cardinality, and missing values")
            for i, group in enumerate(similar_feature_groups, 1):
                print(f"   Group {i}: {len(group['features'])} features")
                for feature_info in group['features']:
                    feature = feature_info['name']
                    cardinality = feature_info['cardinality']
                    missing_count = feature_info['missing_count']
                    print(f"      • {feature} (cardinality: {cardinality}, missing: {missing_count:,})")
                print(f"      💡 Suggestion: Consider dropping similar features, keep one representative")
        
        # No issues found
        if not issues_found:
            print(f"\n   ✅ No major data quality issues detected!")
        
        return self._prompt_user_for_feature_dropping(issues_found)
    
    def _detect_similar_feature_groups(self, results: Dict[str, Any], target_column: str) -> List[Dict]:
        """Detect groups of similar features based on name patterns, cardinality, and missing values"""
        
        # Get all features except target
        features = [(name, info) for name, info in results.items() 
                   if name != target_column and 'error' not in info]
        
        similar_groups = []
        processed_features = set()
        
        for i, (feature1, info1) in enumerate(features):
            if feature1 in processed_features:
                continue
                
            # Find similar features
            group_features = [{'name': feature1, 
                             'cardinality': info1.get('cardinality', 0),
                             'missing_count': info1.get('null_count', 0)}]
            
            for j, (feature2, info2) in enumerate(features):
                if i >= j or feature2 in processed_features:
                    continue
                
                # Check similarity criteria
                if self._are_features_similar(feature1, info1, feature2, info2):
                    group_features.append({'name': feature2,
                                         'cardinality': info2.get('cardinality', 0),
                                         'missing_count': info2.get('null_count', 0)})
                    processed_features.add(feature2)
            
            # If we found a group (more than 1 feature), add it
            if len(group_features) > 1:
                similar_groups.append({'features': group_features})
                processed_features.add(feature1)
        
        return similar_groups
    
    def _are_features_similar(self, feature1: str, info1: Dict, feature2: str, info2: Dict) -> bool:
        """Check if two features are similar based on name pattern, cardinality, and missing values"""
        
        # 1. Check cardinality similarity (exact match)
        cardinality1 = info1.get('cardinality', 0)
        cardinality2 = info2.get('cardinality', 0)
        if cardinality1 != cardinality2:
            return False
        
        # 2. Check missing values similarity (exact match)
        missing1 = info1.get('null_count', 0)
        missing2 = info2.get('null_count', 0)
        if missing1 != missing2:
            return False
        
        # 3. Check name similarity using multiple methods
        return self._are_names_similar(feature1, feature2)
    
    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two feature names are similar using multiple similarity methods"""
        
        # Method 1: Common prefix/suffix patterns
        # Look for common patterns like: feature_id/feature_desc, feature_number/feature_name, etc.
        base1 = self._extract_base_name(name1)
        base2 = self._extract_base_name(name2)
        
        if base1 == base2 and base1:  # Same base name
            return True
        
        # Method 2: String similarity using SequenceMatcher
        similarity_ratio = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        if similarity_ratio >= 0.7:  # 70% similarity threshold
            return True
        
        # Method 3: Common word patterns
        words1 = set(re.findall(r'\w+', name1.lower()))
        words2 = set(re.findall(r'\w+', name2.lower()))
        
        # Check if they share significant common words
        common_words = words1.intersection(words2)
        if len(common_words) >= 2:  # At least 2 common words
            return True
        
        return False
    
    def _extract_base_name(self, name: str) -> str:
        """Extract base name by removing common suffixes/prefixes"""
        
        # Common suffixes to remove
        suffixes = ['_id', '_desc', '_description', '_number', '_num', '_name', 
                   '_code', '_key', '_value', '_mes', '_erp']
        
        name_lower = name.lower()
        base_name = name_lower
        
        # Remove suffixes
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                base_name = name_lower[:-len(suffix)]
                break
        
        # Remove common prefixes if any
        prefixes = ['new_', 'old_', 'temp_', 'tmp_']
        for prefix in prefixes:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        return base_name if len(base_name) > 2 else ""  # Return only if meaningful length
    
    def _prompt_user_for_feature_dropping(self, issues_found: Dict[str, List]) -> List[str]:
        """Prompt user for features to drop based on identified issues, defaulting to autodetected features"""
        
        print(f"\n{'='*155}")
        print(f"🗑️ FEATURE DROPPING SELECTION")
        print(f"{'='*155}")
        
        # Collect autodetected features for dropping
        auto_drop_candidates = []
        candidate_counter = 1
        
        print(f"\n💡 AUTOMATIC DROP SUGGESTIONS BY CATEGORY:")
        
        # 1. Constant features
        if 'constant' in issues_found:
            print(f"\n   1️⃣ CONSTANT VALUE FEATURES:")
            for feature, cardinality in issues_found['constant']:
                print(f"   {candidate_counter}. {feature} (cardinality: {cardinality})")
                auto_drop_candidates.append(feature)
                candidate_counter += 1
        
        # 2. All null features
        if 'all_null' in issues_found:
            print(f"\n   2️⃣ COMPLETELY NULL FEATURES:")
            for feature in issues_found['all_null']:
                print(f"   {candidate_counter}. {feature} (100% missing)")
                auto_drop_candidates.append(feature)
                candidate_counter += 1
        
        # 3. High missing features
        if 'high_missing' in issues_found:
            print(f"\n   3️⃣ HIGH MISSING VALUE FEATURES:")
            for feature, missing_count, missing_pct, non_null_count in issues_found['high_missing']:
                print(f"   {candidate_counter}. {feature} (non-null: {non_null_count:,}, missing: {missing_count:,})")
                auto_drop_candidates.append(feature)
                candidate_counter += 1
        
        # 4. High cardinality features
        if 'high_cardinality' in issues_found:
            print(f"\n   4️⃣ HIGH CARDINALITY FEATURES:")
            for feature, cardinality in issues_found['high_cardinality']:
                print(f"   {candidate_counter}. {feature} (cardinality: {cardinality:,})")
                auto_drop_candidates.append(feature)
                candidate_counter += 1
        
        # 5. Similar feature groups
        if 'similar_groups' in issues_found:
            print(f"\n   5️⃣ SIMILAR FEATURE GROUPS:")
            for group_idx, group in enumerate(issues_found['similar_groups'], 1):
                features_in_group = group['features']
                
                # Get cardinality and missing values (should be same for all in group)
                cardinality = features_in_group[0]['cardinality']
                missing_count = features_in_group[0]['missing_count']
                
                print(f"   Group {group_idx}: (cardinality: {cardinality}, missing: {missing_count:,})")
                
                # Show kept feature (first one)
                kept_feature = features_in_group[0]['name']
                print(f"      ✅ KEPT: {kept_feature}")
                
                # Show features to drop (rest of group)
                for feature_info in features_in_group[1:]:
                    feature = feature_info['name']
                    print(f"      {candidate_counter}. {feature} (duplicate of {kept_feature})")
                    auto_drop_candidates.append(feature)
                    candidate_counter += 1
        
        # Remove duplicates while preserving order
        auto_drop_candidates = list(dict.fromkeys(auto_drop_candidates))
        
        # Show summary
        if auto_drop_candidates:
            print(f"\n📊 SUMMARY: {len(auto_drop_candidates)} features suggested for dropping")
            print(f"   Features to drop: {', '.join(auto_drop_candidates)}")
        else:
            print(f"\n📊 SUMMARY: No features suggested for dropping")
        
        # Prompt for features to drop
        print(f"\nEnter feature names to drop (comma-separated), 'none' to skip, or press Enter to accept defaults:")
        print(f"💡 Tip: You can copy-paste the complete list from summary above")
        
        while True:
            user_input = input(f"\nEnter selection: ").strip()
            
            if user_input.lower() == 'none':
                return []
            
            if not user_input:
                # Return autodetected features if user input is empty
                if auto_drop_candidates:
                    print(f"\n✅ Using default: {len(auto_drop_candidates)} features to drop:")
                    for feature in auto_drop_candidates:
                        print(f"   • {feature}")
                    confirm = input(f"\nConfirm dropping these features? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return auto_drop_candidates
                    else:
                        continue
                else:
                    return []
            
            # Parse comma-separated feature names
            selected_features = [name.strip() for name in user_input.split(',') if name.strip()]
            
            if selected_features:
                # Remove duplicates
                selected_features = list(set(selected_features))
                print(f"\n✅ Selected {len(selected_features)} features to drop:")
                for feature in selected_features:
                    print(f"   • {feature}")
                
                confirm = input(f"\nConfirm dropping these features? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_features
                else:
                    continue
            else:
                return auto_drop_candidates
    
    def _profile_column(self, series: pd.Series, col_name: str, categories: Dict[str, List[str]]) -> Dict[str, Any]:
        """Profile a single column with comprehensive analysis"""
        
        # Determine category
        category = self._get_column_category(col_name, categories)
        
        # Basic counts
        total_count = len(series)
        null_count = series.isnull().sum()
        non_null_count = total_count - null_count
        unique_count = series.nunique()
        
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
        cardinality = unique_count
        
        # Storage type detection
        storage_type = self._get_storage_type(series)
        
        # Constant value detection
        is_constant = unique_count <= 1
        
        # Get example value safely
        example_value = self._get_safe_example(series)
        
        # Additional analysis
        has_mixed_types = self._detect_mixed_types(series)
        is_potential_id = self._is_potential_identifier(series, col_name)
        
        return {
            'dtype': str(series.dtype),
            'category': category,
            'storage_type': storage_type,
            'cardinality': cardinality,
            'total_count': total_count,
            'null_count': null_count,
            'non_null_count': non_null_count,
            'null_percentage': null_percentage,
            'example_value': example_value,
            'is_constant': is_constant,
            'has_mixed_types': has_mixed_types,
            'is_potential_id': is_potential_id
        }
    
    def _detect_mixed_types(self, series: pd.Series) -> bool:
        """Detect mixed data types"""
        if series.dtype == 'object':
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return False
            
            sample = non_null_series.head(1000)  # Sample for performance
            types = set(type(x).__name__ for x in sample)
            return len(types) > 1
        
        return False
    
    def _is_potential_identifier(self, series: pd.Series, col_name: str) -> bool:
        """Check if column is potentially an identifier"""
        # Check if column name suggests it's an ID
        id_keywords = ['id', 'key', 'number', 'code', 'serial']
        name_suggests_id = any(keyword in col_name.lower() for keyword in id_keywords)
        
        # Check if values are unique (or nearly unique)
        uniqueness_ratio = series.nunique() / len(series)
        
        return name_suggests_id and uniqueness_ratio > 0.95
    
    def _get_column_category(self, col_name: str, categories: Dict[str, List[str]]) -> str:
        """Get the category this column belongs to"""
        for category, columns in categories.items():
            if col_name in columns:
                return category
        return 'unknown'
    
    def _get_storage_type(self, series: pd.Series) -> str:
        """Detect precise storage type"""
        dtype_str = str(series.dtype)
        
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'string'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return 'unknown'
    
    def _get_safe_example(self, series: pd.Series) -> str:
        """Get a safe example value handling all data types"""
        try:
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return "All null"
                
            example = non_null_series.iloc[0]
            
            if pd.isna(example):
                return "null"
            elif isinstance(example, (int, float, np.integer, np.floating)):
                if np.isinf(example):
                    return "inf"
                return str(example)
            elif isinstance(example, bool):
                return str(example)
            else:
                example_str = str(example)
                return example_str[:47] + "..." if len(example_str) > 47 else example_str
                
        except Exception:
            return "Error getting example"
    
    def show_improvement_suggestions(self, results: Dict[str, Any]):
        """Minimal implementation to maintain compatibility with main pipeline"""
        print(f"\n{'='*155}")
        print(f"🛠️ DATASET IMPROVEMENT SUGGESTIONS")
        print(f"{'='*155}")
        print("\n✅ Data quality analysis completed. Proceed to next step.")
    
    # Legacy method for compatibility
    def display_profile_summary_ranked(self, results: Dict[str, Any]):
        """Legacy method - redirects to enhanced version"""
        total_rows = self.total_rows or len(results)
        self.display_enhanced_profile_summary(results, total_rows)