#!/usr/bin/env python3
"""
MAIN PIPELINE INTEGRATION FIXES
Addresses the integration issues between the main pipeline and the fixed feature importance analyzer.
Fixes NoneType division errors and improves workflow integration.

Key Fixes:
- Fixed feature importance workflow integration 
- Removed orphaned dataset saving that's not needed
- Fixed NoneType division errors in main pipeline
- Improved error handling in feature importance workflow
- Optimized for manufacturing datasets with 117:5.5 ratio imbalance
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import time
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Import all components (assuming they exist)
from data_categorizer import ManufacturingDataCategorizer
from column_profiler import ColumnProfiler
from encoding_manager import EnhancedEncodingManager
from correlation_analyzer import RevisedCorrelationAnalyzer
from balance_handler import FixedBalanceHandler
from pipeline_utils import EnhancedPipelineUtils

# Import the FIXED feature importance analyzer
from feature_importance import FixedFeatureImportanceAnalyzer


class EnhancedIntegratedManufacturingPipeline:
    """Fixed main pipeline with proper feature importance integration"""

    def __init__(self, data_path: str = None, output_dir: str = None):
        if not data_path:
            data_path = self._prompt_for_input_path()
        if not output_dir:
            output_dir = self._prompt_for_output_directory()
            
        self.data_path = Path(data_path)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = Path(output_dir) if output_dir else Path('./enhanced_integrated_results')
        self.output_dir = base_output / f"enhanced_integrated_analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup comprehensive logging
        self.logger = EnhancedPipelineUtils.setup_logging(self.output_dir)
        self.plots_dir = EnhancedPipelineUtils.create_plots_dir(self.output_dir)
        
        # Initialize all components with FIXED feature importance analyzer
        self.categorizer = ManufacturingDataCategorizer(logger=self.logger)
        self.profiler = ColumnProfiler(self.logger)
        self.encoder = EnhancedEncodingManager(self.logger)
        self.balance_handler = FixedBalanceHandler(self.logger, self.plots_dir)
        self.correlation_analyzer = RevisedCorrelationAnalyzer(self.logger, self.plots_dir)
        
        # FIXED: Use the new fixed feature importance analyzer
        self.feature_analyzer = FixedFeatureImportanceAnalyzer(
            self.logger, self.plots_dir, self.balance_handler
        )
        
        # Get methods from components safely
        self.correlation_methods = getattr(self.correlation_analyzer, 'correlation_methods', {})
        self.feature_importance_methods = getattr(self.feature_analyzer, 'importance_methods', {})
        self.balancing_methods = getattr(self.balance_handler, 'balancing_methods', {})
        
        # Pipeline state
        self.current_dataset = None
        self.target_column = None
        self.data_categories = None
        self.saved_datasets = {}
        self.analysis_history = []
        
        # Safe optimization state
        self.optimization_applied = False
        self.optimization_level = None
    
    def run_enhanced_integrated_pipeline(self):
        """Run the comprehensive enhanced integrated pipeline with fixes"""
        
        try:
            # Step 1: Load and categorize data
            self._load_and_categorize_data()
            
            # Step 2: Enhanced column profiling 
            self._enhanced_column_profiling()
            
            # Step 3: Target selection and quality improvements
            self._target_selection_and_quality_improvements()
            
            # Step 4: Enhanced encoding
            self._apply_enhanced_encoding_with_debugging(self.current_dataset)

            # Step 5: Binary classification creation (if needed)
            self._create_binary_classification_if_needed()

            # Step 6: Interactive analysis loop
            self._interactive_analysis_loop()
            
            # Step 7: Final summary
            self._generate_final_summary()
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Pipeline interrupted by user")
            self._save_current_state()
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _load_and_categorize_data(self):
        """Step 1: Load and categorize data"""
        
        print(f"\n🔧 Step 1: Loading and Categorizing Data")
        with EnhancedPipelineUtils.performance_monitor("Data Loading & Categorization", self.logger):
            
            # Load data
            if self.data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(self.data_path)
            elif self.data_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.data_path, low_memory=False)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                
            # Categorize data (optimized - no heavy computation)
            self.data_categories = self.categorizer.categorize_data(df)
            self.current_dataset = df
    
    def _enhanced_column_profiling(self):
        """Step 2: Enhanced column profiling"""
        
        print(f"\n🔧 Step 2: Enhanced Column Profiling")
        with EnhancedPipelineUtils.performance_monitor("Column Profiling", self.logger):
            
            # Profile columns
            profile_results = self.profiler.profile_dataframe(self.current_dataset, self.data_categories)
            self.profiler.display_enhanced_profile_summary(profile_results, len(self.current_dataset))
            
            # Store for later use
            self.profile_results = profile_results
    
    def _target_selection_and_quality_improvements(self):
        """Step 3: Target selection and quality improvements"""
        
        print(f"\n🔧 Step 3: Target Selection & Quality Improvements")
        with EnhancedPipelineUtils.performance_monitor("Target Selection & Quality", self.logger):
            
            # Target selection
            self.target_column, features_to_drop = self.profiler.prompt_target_selection_and_find_issues(
                self.current_dataset, self.profile_results
            )
            
            # Apply quality improvements
            if features_to_drop:
                original_shape = self.current_dataset.shape
                self.current_dataset = self.current_dataset.drop(columns=features_to_drop)
                print(f"✅ Quality improvement: {original_shape} → {self.current_dataset.shape}")
            
            # ESSENTIAL DATASETS ONLY - Save key intermediate results
            self._save_essential_dataset(
                self.current_dataset,
                "02_quality_filtered",
                {
                    'target_column': self.target_column,
                    'features_dropped_count': len(features_to_drop) if features_to_drop else 0,
                    'processing_stage': 'quality_filtering'
                }
            )
    
    def _apply_enhanced_encoding_with_debugging(self, df: pd.DataFrame):
        """Step 4: Apply encoding with essential tracking only"""
        
        print(f"\n🔧 Step 4: Encoding")
        
        # Generate and apply encoding
        encoding_summary = self.encoder.generate_comprehensive_encoding_summary(
            df, self.data_categories, self.target_column
        )
        
        print(f"\n📊 ENCODING STRATEGY:")
        self.encoder.display_initial_encoding_strategy_table(encoding_summary)
        
        # Apply encoding transformations
        encoded_df, encoding_report = self.encoder.apply_comprehensive_encoding(
            df, self.target_column, encoding_summary
        )
        
        print(f"\n📈 ENCODING RESULTS:")
        self.encoder.display_final_encoding_results_table()
        
        # Store the encoded dataset
        self.current_dataset = encoded_df
        
        # Summary statistics
        original_feature_count = len([col for col in df.columns if col != self.target_column])
        encoded_feature_count = len([col for col in encoded_df.columns if col != self.target_column])
        
        print(f"\n✅ Encoding Process Completed:")
        print(f"   📊 Original shape: {df.shape}")
        print(f"   📊 Encoded shape: {encoded_df.shape}")
        print(f"   📈 Feature count: {original_feature_count} → {encoded_feature_count}")
        
        # ESSENTIAL DATASETS ONLY - Save encoded dataset
        self._save_essential_dataset(
            self.current_dataset,
            "03_encoded", 
            {
                'encoding_success_rate': len(encoding_report.get('success', [])) / original_feature_count * 100 if original_feature_count > 0 else 0,
                'processing_stage': 'encoding_complete',
                'feature_expansion': encoded_feature_count - original_feature_count
            }
        )
        
        return encoded_df, encoding_report

    def _create_binary_classification_if_needed(self):
        """Step 5: Create binary classification target if needed"""
        
        if self.target_column and self.target_column in self.current_dataset.columns:
            target_series = self.current_dataset[self.target_column]
            unique_values = sorted(target_series.unique())
            
            print(f"\n🎯 Target '{self.target_column}' analysis:")
            print(f"   Unique values: {unique_values}")
            print(f"   Value counts: {target_series.value_counts().to_dict()}")
            
            if len(unique_values) > 2:
                create_binary = input("\nCreate binary classification target? (y/n): ").strip().lower()
                if create_binary == 'y':
                    print("Creating binary target: 1,2 = faulty (1), 0 = pass (0)")
                    binary_target = target_series.apply(lambda x: 1 if x in [1, 2] else 0)
                    self.current_dataset[self.target_column] = binary_target
                    
                    value_counts = binary_target.value_counts().to_dict()
                    self.logger.info(f"Binary target created - Distribution: {value_counts}")
                    
                    # ESSENTIAL DATASETS ONLY
                    self._save_essential_dataset(
                        self.current_dataset,
                        "04_binary_target",
                        {
                            'binary_classification': True,
                            'final_distribution': value_counts,
                            'processing_stage': 'binary_conversion'
                        }
                    )   

    def _interactive_analysis_loop(self):
        """Step 6: Main interactive analysis loop"""
        
        while True:
            choice = self._display_main_menu()
            
            if choice == '1':
                self._correlation_analysis_workflow()
            elif choice == '2':
                self._fixed_feature_importance_workflow()  # FIXED METHOD
            elif choice == '3':
                self._balancing_workflow()
            elif choice == '4':
                self._view_saved_datasets()
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _display_main_menu(self) -> str:
        """Display main analysis menu"""
        
        print(f"\n{'='*155}")
        print(f"🔬 INTEGRATED PREPROCESSING OPTIONS")
        print(f"{'='*155}")
        print(f"📊 Current Dataset: {self.current_dataset.shape if self.current_dataset is not None else 'None'}")
        print(f"🎯 Target Column: {self.target_column}")
        print(f"📁 Essential Datasets: {len(self.saved_datasets)} available")
        print(f"")
        print(f"Available Analysis Options:")
        print(f"1. 📊 Correlation Analysis")
        print(f"2. 🎯 Feature Importance Analysis (FIXED)")
        print(f"3. ⚖️  Data Balancing")
        print(f"4. 📁 View Saved Datasets")
        print(f"5. ✅ Finish Analysis")
        
        return input(f"\nSelect option (1-5): ").strip()

    def _fixed_feature_importance_workflow(self):
        """FIXED feature importance analysis workflow with comprehensive error handling"""
        
        print(f"\n{'='*155}")
        print(f"🎯 FEATURE IMPORTANCE ANALYSIS WORKFLOW (FIXED)")
        print(f"{'='*155}")
        
        if not self.feature_importance_methods:
            print("❌ Feature importance methods not available")
            return
        
        # Display methods safely
        print(f"\n📚 Manufacturing-Optimized Feature Importance Methods:")
        methods_list = list(self.feature_importance_methods.keys())
        
        for i, (method_key, method_info) in enumerate(self.feature_importance_methods.items(), 1):
            req_symbol = "⚖️" if method_info.get('requires_balanced', False) else "✅"
            print(f"{i}. {req_symbol} [{method_info.get('symbol', 'N/A')}] {method_info.get('name', method_key)}")
            print(f"   📖 {method_info.get('description', 'No description')}")
            print(f"   🎯 Cost: {method_info.get('computational_cost', 'Unknown')}")
            
        print(f"\n⚖️ = May work better with balanced data")
        print(f"✅ = Works well with imbalanced data")
        
        # Safe method selection
        try:
            choice = int(input(f"\nSelect method (1-{len(methods_list)}): "))
            if 1 <= choice <= len(methods_list):
                method_key = methods_list[choice - 1]
                method_info = self.feature_importance_methods[method_key]
            else:
                print("❌ Invalid selection")
                return
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
        
        # Safe dataset validation
        if self.current_dataset is None:
            print("❌ No current dataset available")
            return
        
        if self.target_column is None or self.target_column not in self.current_dataset.columns:
            print("❌ No valid target column available")
            return
        
        # Dataset selection with validation
        try:
            selected_df = self._safe_dataset_selection("feature_importance")
            if selected_df is None:
                print("❌ Dataset selection failed")
                return
        except Exception as e:
            print(f"❌ Dataset selection error: {e}")
            return
        
        # Apply feature importance analysis with comprehensive error handling
        try:
            print(f"\n🚀 Starting {method_info.get('name', method_key)} analysis...")
            
            with EnhancedPipelineUtils.performance_monitor(f"Feature Importance ({method_key})", self.logger):
                
                # FIXED: Call the corrected analyze method
                processed_df, analysis_results = self.feature_analyzer.analyze_enhanced_importance(
                    selected_df, self.target_column, method_key
                )
                
                if processed_df is None or analysis_results is None:
                    raise ValueError("Analysis returned None results")
                
                # Create analysis timestamp
                analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"feature_importance_{method_key}_{analysis_timestamp}"
                
                # FIXED: Safe results saving
                try:
                    self._save_feature_importance_results(
                        processed_df, analysis_results, dataset_name, method_key
                    )
                except Exception as save_error:
                    print(f"⚠️ Could not save results: {save_error}")
                    self.logger.warning(f"Feature importance results save failed: {save_error}")
                
                # Safe dataset update option
                try:
                    update_current = input(f"\nUpdate current dataset with results? (y/n): ").strip().lower()
                    if update_current == 'y':
                        self.current_dataset = processed_df
                        print("✅ Current dataset updated")
                except Exception as e:
                    print(f"⚠️ Could not update current dataset: {e}")
                
                # Record in history safely
                try:
                    self.analysis_history.append({
                        'type': 'feature_importance',
                        'method': method_key,
                        'timestamp': analysis_timestamp,
                        'features_removed': len(analysis_results.get('removed_features', [])),
                        'success': True
                    })
                except Exception as e:
                    self.logger.warning(f"Could not record analysis history: {e}")
                
                print(f"\n✅ Feature importance analysis completed successfully!")
                print(f"📁 Results processed for method: {method_key}")
                
        except Exception as e:
            error_msg = f"Feature importance analysis failed: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            
            # Record failed attempt in history
            try:
                self.analysis_history.append({
                    'type': 'feature_importance',
                    'method': method_key,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'error': str(e),
                    'success': False
                })
            except:
                pass  # Don't let history recording break error handling

    def _safe_dataset_selection(self, method_type: str) -> Optional[pd.DataFrame]:
        """Safe dataset selection with validation"""
        
        print(f"\n📊 DATASET SELECTION FOR {method_type.upper()}")
        
        # Always use current dataset for simplicity and to avoid orphaned datasets
        if self.current_dataset is not None:
            print(f"Using current dataset: {self.current_dataset.shape}")
            return self.current_dataset.copy()
        
        print("❌ No current dataset available")
        return None

    def _save_feature_importance_results(self, df: pd.DataFrame, results: Dict, 
                                       name: str, method: str):
        """Save feature importance results with essential information only"""
        
        try:
            # Save essential dataset
            self._save_essential_dataset(df, name, {
                'analysis_type': 'feature_importance',
                'method': method,
                'features_removed_count': len(results.get('removed_features', [])),
                'total_features_analyzed': results.get('total_features_analyzed', 0),
                'execution_time': results.get('execution_time', 0),
                'processing_stage': 'feature_importance_complete'
            })
            
            # Save essential JSON summary only
            summary = {
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'features_analyzed': results.get('total_features_analyzed', 0),
                'features_removed': len(results.get('removed_features', [])),
                'execution_time': results.get('execution_time', 0),
                'success': True
            }
            
            summary_file = self.output_dir / f"{name}_summary.json"
            try:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"📄 Summary saved: {summary_file.name}")
            except Exception as e:
                self.logger.warning(f"Could not save summary: {e}")
                
        except Exception as e:
            error_msg = f"Could not save feature importance results: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.warning(error_msg)

    def _correlation_analysis_workflow(self):
        """Correlation analysis workflow (unchanged)"""
        
        print(f"\n{'='*155}")
        print(f"📊 CORRELATION ANALYSIS WORKFLOW")
        print(f"{'='*155}")
        
        if not self.correlation_methods:
            print("❌ Correlation methods not available")
            return
        
        # Display available methods
        print(f"\n📚 Available Correlation Methods:")
        methods_list = list(self.correlation_methods.keys())
        
        for i, (method_key, method_info) in enumerate(self.correlation_methods.items(), 1):
            available_status = "✅" if method_info.get('implemented', True) else "❌"
            print(f"{i}. {available_status} {method_info.get('name', method_key)}")
        
        # Method selection and application (existing logic)
        try:
            choice = int(input(f"\nSelect method (1-{len(methods_list)}): "))
            if 1 <= choice <= len(methods_list):
                method_key = methods_list[choice - 1]
                method_info = self.correlation_methods[method_key]
                
                if not method_info.get('implemented', True):
                    print("❌ Selected method is not available")
                    return
                
                # Apply correlation analysis
                selected_df = self._safe_dataset_selection("correlation")
                if selected_df is not None:
                    processed_df, analysis_results = self.correlation_analyzer.analyze_enhanced_correlation(
                        selected_df, method_key, self.target_column
                    )
                    
                    analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dataset_name = f"correlation_{method_key}_{analysis_timestamp}"
                    
                    self._save_essential_dataset(processed_df, dataset_name, {
                        'analysis_type': 'correlation',
                        'method': method_key,
                        'processing_stage': 'correlation_complete'
                    })
                    
                    print(f"✅ Correlation analysis completed: {dataset_name}")
                    
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            self.logger.error(f"Correlation analysis failed: {str(e)}")
    
    def _balancing_workflow(self):
        """Data balancing workflow (unchanged)"""
        
        print(f"\n{'='*155}")
        print(f"⚖️  DATA BALANCING WORKFLOW")
        print(f"{'='*155}")
        
        if not self.balancing_methods:
            print("❌ Balancing methods not available")
            return
        
        # Display current balance
        if self.target_column and self.current_dataset is not None:
            current_balance = self._analyze_current_balance()
            self._display_balance_analysis(current_balance)
        
        # Display available methods
        print(f"\n📚 Available Balancing Methods:")
        methods_list = list(self.balancing_methods.keys())
        
        for i, (method_key, method_info) in enumerate(self.balancing_methods.items(), 1):
            available_status = "✅" if method_info.get('available', True) else "❌"
            print(f"{i}. {available_status} {method_info.get('name', method_key)}")
        
        # Method selection and application (existing logic)
        try:
            choice = int(input(f"\nSelect method (1-{len(methods_list)}): "))
            if 1 <= choice <= len(methods_list):
                method_key = methods_list[choice - 1]
                
                selected_df = self._safe_dataset_selection("balancing")
                if selected_df is not None:
                    balanced_df, balancing_results = self.balance_handler.apply_enhanced_balancing(
                        selected_df, self.target_column, method_key
                    )
                    
                    analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dataset_name = f"balanced_{method_key}_{analysis_timestamp}"
                    
                    self._save_essential_dataset(balanced_df, dataset_name, {
                        'analysis_type': 'balancing',
                        'method': method_key,
                        'processing_stage': 'balancing_complete'
                    })
                    
                    print(f"✅ Balancing completed: {dataset_name}")
                    
        except Exception as e:
            print(f"❌ Balancing failed: {e}")
            self.logger.error(f"Data balancing failed: {str(e)}")

    def _analyze_current_balance(self) -> Dict:
        """Analyze current dataset balance with safe error handling"""
        
        try:
            if self.target_column is None or self.current_dataset is None:
                return {'error': 'No target column or dataset available'}
            
            if self.target_column not in self.current_dataset.columns:
                return {'error': f'Target column {self.target_column} not found'}
            
            value_counts = self.current_dataset[self.target_column].value_counts()
            
            if len(value_counts) == 0:
                return {'error': 'No valid target values found'}
            
            return {
                'distribution': value_counts.to_dict(),
                'majority_count': int(value_counts.max()),
                'minority_count': int(value_counts.min()),
                'imbalance_ratio': float(value_counts.max() / value_counts.min()) if value_counts.min() > 0 else float('inf'),
                'total_samples': len(self.current_dataset)
            }
            
        except Exception as e:
            return {'error': f'Balance analysis failed: {str(e)}'}

    def _display_balance_analysis(self, balance_info: Dict):
        """Display current balance analysis safely"""
        
        if 'error' in balance_info:
            print(f"⚠️ Balance analysis error: {balance_info['error']}")
            return
        
        try:
            print(f"\n📊 Current Dataset Balance:")
            print(f"   Total samples: {balance_info['total_samples']:,}")
            print(f"   Imbalance ratio: {balance_info['imbalance_ratio']:.2f}:1")
            
            for class_val, count in balance_info['distribution'].items():
                percentage = (count / balance_info['total_samples']) * 100
                status = "👑 MAJORITY" if count == balance_info['majority_count'] else "📌 MINORITY"
                print(f"   Class {class_val}: {count:,} samples ({percentage:.1f}%) - {status}")
                
        except Exception as e:
            print(f"⚠️ Could not display balance analysis: {e}")
    
    def _view_saved_datasets(self):
        """View and manage saved datasets"""
        
        print(f"\n{'='*155}")
        print(f"📁 ESSENTIAL DATASETS MANAGER")
        print(f"{'='*155}")
        
        if not self.saved_datasets:
            print("📂 No essential datasets saved yet")
            return
        
        print(f"Available essential datasets ({len(self.saved_datasets)}):")
        dataset_list = list(self.saved_datasets.keys())
        
        for i, dataset_name in enumerate(dataset_list, 1):
            dataset_info = self.saved_datasets[dataset_name]
            shape = dataset_info.get('shape', 'Unknown')
            stage = dataset_info.get('processing_stage', 'Unknown')
            analysis_type = dataset_info.get('analysis_type', 'preprocessing')
            print(f"{i}. 📊 {dataset_name}")
            print(f"   Shape: {shape}")
            print(f"   Stage: {stage}")
            print(f"   Type: {analysis_type}")
            print()
        
        # Dataset loading option
        try:
            choice = input(f"Load dataset (1-{len(dataset_list)}) or press Enter to continue: ").strip()
            
            if choice and choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(dataset_list):
                    selected_dataset = dataset_list[choice_num - 1]
                    self.current_dataset = self.saved_datasets[selected_dataset]['dataframe']
                    print(f"✅ Loaded dataset: {selected_dataset}")
                else:
                    print("❌ Invalid selection")
                    
        except Exception as e:
            print(f"⚠️ Dataset loading error: {e}")
    
    def _save_essential_dataset(self, df: pd.DataFrame, name: str, metadata: Dict = None):
        """Save only essential datasets to avoid orphaned data"""
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file efficiently
            file_path = self.output_dir / f"{name}.parquet"
            success = EnhancedPipelineUtils.save_dataset(df, file_path, 'parquet')
            
            if success:
                # Store essential information only
                self.saved_datasets[name] = {
                    'dataframe': df.copy(),
                    'shape': df.shape,
                    'file_path': str(file_path),
                    'timestamp': datetime.now().isoformat(),
                    **(metadata or {})
                }
                
                print(f"💾 Saved essential dataset: {name} ({df.shape[0]:,} × {df.shape[1]})")
                self.logger.info(f"Essential dataset saved: {name}")
                
            else:
                print(f"⚠️ Could not save dataset: {name}")
                
        except Exception as e:
            error_msg = f"Failed to save essential dataset {name}: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.warning(error_msg)
    
    def _generate_final_summary(self):
        """Generate comprehensive final summary"""
        
        print(f"\n{'='*155}")
        print(f"📋 ENHANCED FINAL ANALYSIS SUMMARY")
        print(f"{'='*155}")
        
        # Analysis history
        if self.analysis_history:
            print(f"\nAnalysis History ({len(self.analysis_history)} operations):")
            for i, analysis in enumerate(self.analysis_history, 1):
                success_indicator = "✅" if analysis.get('success', True) else "❌"
                print(f"{i}. {success_indicator} {analysis['type'].upper()}: {analysis.get('method', 'unknown')} ({analysis.get('timestamp', 'unknown')})")
                if 'features_removed' in analysis:
                    print(f"   Features removed: {analysis['features_removed']}")
                if 'error' in analysis:
                    print(f"   Error: {analysis['error']}")
        
        # Essential datasets created
        print(f"\nEssential Datasets Created ({len(self.saved_datasets)}):")
        for name, info in self.saved_datasets.items():
            stage = info.get('processing_stage', 'Unknown')
            print(f"   {name}: {info.get('shape', 'Unknown shape')} - {stage}")
        
        # Final dataset status
        if self.current_dataset is not None:
            print(f"\nFinal Dataset Status:")
            print(f"   Shape: {self.current_dataset.shape}")
            print(f"   Target: {self.target_column}")
            print(f"   Memory: {self.current_dataset.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Save comprehensive report
        self._save_comprehensive_report()
        
        print(f"\nENHANCED INTEGRATED ANALYSIS COMPLETED!")
        print(f"All results saved in: {self.output_dir}")
    
    def _save_comprehensive_report(self):
        """Save comprehensive analysis report"""
        
        report = {
            'pipeline_info': {
                'version': 'enhanced_integrated_v3.0_fixed',
                'timestamp': datetime.now().isoformat(),
                'target_column': self.target_column,
                'feature_importance_fixed': True
            },
            'analysis_history': self.analysis_history,
            'essential_datasets': {
                name: {
                    'shape': info['shape'],
                    'processing_stage': info.get('processing_stage', 'Unknown'),
                    'file_path': info['file_path']
                }
                for name, info in self.saved_datasets.items()
            },
            'final_statistics': {
                'total_analyses': len(self.analysis_history),
                'essential_datasets_generated': len(self.saved_datasets),
                'final_shape': list(self.current_dataset.shape) if self.current_dataset is not None else None,
                'successful_analyses': len([a for a in self.analysis_history if a.get('success', True)])
            }
        }
        
        try:
            report_path = self.output_dir / 'enhanced_comprehensive_analysis_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📋 Comprehensive report saved: {report_path.name}")
        except Exception as e:
            self.logger.warning(f"Could not save comprehensive report: {e}")
    
    def _save_current_state(self):
        """Save current pipeline state for recovery"""
        
        state = {
            'current_dataset_shape': list(self.current_dataset.shape) if self.current_dataset is not None else None,
            'target_column': self.target_column,
            'analysis_history': self.analysis_history,
            'essential_datasets_count': len(self.saved_datasets),
            'interrupted_at': datetime.now().isoformat()
        }
        
        try:
            state_path = self.output_dir / 'pipeline_state.json'
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            print(f"💾 Pipeline state saved: {state_path.name}")
        except Exception as e:
            self.logger.warning(f"Could not save pipeline state: {e}")
    
    def _prompt_for_input_path(self) -> str:
        """Prompt for input data path"""
        
        print(f"\n📂 INPUT DATA FILE SELECTION")
        
        while True:
            file_path = input("Enter path to input data file (.parquet or .csv): ").strip()
            
            if not file_path:
                print("Please enter a valid file path.")
                continue
                
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                print(f"File does not exist: {file_path}")
                continue
                
            if not file_path.lower().endswith(('.parquet', '.csv')):
                print("Please provide a .parquet or .csv file.")
                continue
            
            try:
                file_size = path_obj.stat().st_size / (1024**2)  # MB
                print(f"Selected: {file_path}")
                print(f"File size: {file_size:.1f} MB")
                return file_path
            except Exception as e:
                print(f"Cannot access file: {e}")
                continue
    
    def _prompt_for_output_directory(self) -> str:
        """Prompt for output directory"""
        
        print(f"\n📂 OUTPUT DIRECTORY SELECTION")
        
        while True:
            output_dir = input("Enter output directory (press Enter for './enhanced_integrated_results'): ").strip()
            
            if not output_dir:
                output_dir = './enhanced_integrated_results'
                
            path_obj = Path(output_dir)
            
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                print(f"Output directory: {output_dir}")
                return output_dir
            except Exception as e:
                print(f"Cannot create directory {output_dir}: {e}")
                continue


def main():
    """Main entry point for enhanced integrated pipeline"""
    
    print(f"""
🏭 INTEGRATED MANUFACTURING DATA ANALYSIS PIPELINE
    """)
    
    try:
        # Initialize enhanced integrated pipeline
        pipeline = EnhancedIntegratedManufacturingPipeline()
        
        # Execute comprehensive analysis
        start_time = time.time()
        pipeline.run_enhanced_integrated_pipeline()
        total_time = time.time() - start_time
        
        print(f"\nSUCCESS! Enhanced integrated analysis completed in {total_time:.2f} seconds")
        print(f"All results saved in: {pipeline.output_dir}")
        if pipeline.optimization_applied:
            print(f"Safe optimization applied: {pipeline.optimization_level.upper()} level")
        print(f"Research-validated methods applied successfully")
        
    except KeyboardInterrupt:
        print(f"\nAnalysis interrupted by user")
        
    except Exception as e:
        print(f"\nAnalysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()