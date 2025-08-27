"""
Scientific Exploratory Data Analysis (EDA) Pipeline - Main Module
Advanced Manufacturing Data Analysis Framework

PHASE 1 REVISION: Core Infrastructure Fixed
- Improved error handling
- Better dependency management
- Memory optimization
- Robust logging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import gc
import psutil

# Core required libraries
from scipy import stats

# Safe optional imports with availability tracking
OPTIONAL_LIBS = {}

def check_optional_import(lib_name: str, import_statement: str, globals_dict=None):
    """Safely check and import optional libraries"""
    try:
        if globals_dict is None:
            globals_dict = globals()
        exec(f"import {import_statement}", globals_dict)
        OPTIONAL_LIBS[lib_name] = True
        return True
    except ImportError as e:
        OPTIONAL_LIBS[lib_name] = False
        print(f"⚠️ {lib_name} not available: {e}")
        return False

# Initialize optional dependencies
print("🔍 Checking dependencies...")
check_optional_import("plotly_express", "plotly.express as px", globals())
check_optional_import("plotly_graph", "plotly.graph_objects as go", globals())
check_optional_import("plotly_subplots", "plotly.subplots", globals())
check_optional_import("sklearn_decomp", "sklearn.decomposition", globals())
check_optional_import("sklearn_manifold", "sklearn.manifold", globals())
check_optional_import("sklearn_preprocessing", "sklearn.preprocessing", globals())
check_optional_import("sklearn_cluster", "sklearn.cluster", globals())
check_optional_import("sklearn_ensemble", "sklearn.ensemble", globals())
check_optional_import("sklearn_metrics", "sklearn.metrics", globals())
check_optional_import("pyarrow", "pyarrow as pa", globals())
check_optional_import("pyarrow_parquet", "pyarrow.parquet as pq", globals())
check_optional_import("statsmodels", "statsmodels.tsa.seasonal", globals())

# Import utility classes with better error handling
try:
    from eda_utils import (
        EDAConfig, DataProfiler, DataLoader, TimeSeriesDetector,
        StatisticalAnalyzer, VisualizationEngine, DimensionalityAnalyzer,
        ManufacturingDomainAnalyzer, ReportGenerator
    )
    UTILS_AVAILABLE = True
    print("✅ eda_utils imported successfully")
except ImportError as e:
    print(f"⚠️ eda_utils import failed: {e}")
    print("🔄 Will use fallback functionality")
    UTILS_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Unexpected error importing eda_utils: {e}")
    UTILS_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class ScientificEDAPipeline:
    """
    Main pipeline orchestrator for scientific EDA with improved error handling
    """
    
    def __init__(self, config: Optional['EDAConfig'] = None):
        """Initialize pipeline with proper error handling"""
        try:
            # Use local config if utils not available
            if UTILS_AVAILABLE and config is not None:
                self.config = config
            elif UTILS_AVAILABLE:
                from eda_utils import EDAConfig
                self.config = EDAConfig()
            else:
                # Fallback local config
                self.config = self._create_fallback_config()
            
            # Initialize components if utils available
            if UTILS_AVAILABLE:
                self._initialize_components()
            else:
                self._initialize_fallback_components()
                
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            raise
    
    def _create_fallback_config(self):
        """Create fallback config when eda_utils not available"""
        @dataclass
        class FallbackConfig:
            output_dir: str = "./eda_results"
            max_memory_usage: float = 0.8
            chunk_size: int = 10000
            missing_threshold: float = 0.5
            correlation_threshold: float = 0.9
            outlier_threshold: float = 3.0
            random_state: int = 42
        
        return FallbackConfig()
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        try:
            self.profiler = DataProfiler(self.config)
            self.loader = DataLoader(self.config, self.profiler.logger)
            self.time_detector = TimeSeriesDetector(self.config, self.profiler.logger)
            self.stats_analyzer = StatisticalAnalyzer(self.config, self.profiler.logger)
            self.viz_engine = VisualizationEngine(self.config, self.profiler.logger)
            self.dim_analyzer = DimensionalityAnalyzer(self.config, self.profiler.logger)
            self.mfg_analyzer = ManufacturingDomainAnalyzer(self.config, self.profiler.logger)
            self.report_generator = ReportGenerator(self.config, self.profiler.logger)
            
            self.logger = self.profiler.logger
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            raise
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when utils not available"""
        # Setup basic logging
        Path(self.config.output_dir).mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'eda_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("⚠️ Running with fallback components")
    
    def run_comprehensive_eda(self, file_path: str) -> Dict[str, Any]:
        """
        Run comprehensive EDA pipeline with enhanced error handling
        """
        self.logger.info("🚀 Starting Scientific EDA Pipeline v2.0")
        start_time = time.time()
        
        try:
            # Validate input
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not UTILS_AVAILABLE:
                return self._run_basic_eda(file_path, start_time)
            
            # Step 1: File metadata profiling
            self.logger.info("📋 Step 1: File Metadata Profiling")
            metadata = self.profiler.profile_file_metadata(file_path)
            
            # Step 2: Smart data loading
            self.logger.info("📂 Step 2: Smart Data Loading")
            df = self.loader.load_data_smart(file_path, metadata)
            
            # Step 3: Basic data validation
            self.logger.info("🔍 Step 3: Data Validation")
            validation_results = self._validate_dataframe(df)
            
            if not validation_results['valid']:
                raise ValueError(f"Data validation failed: {validation_results['errors']}")
            
            # Step 4: Data quality assessment
            self.logger.info("📊 Step 4: Data Quality Assessment")
            quality_metrics = self._assess_data_quality(df)
            
            # Step 5: Temporal analysis
            self.logger.info("🕐 Step 5: Temporal Pattern Detection")
            temporal_analysis = self.time_detector.detect_temporal_features(df)
            
            # Step 6: Statistical analysis
            self.logger.info("📈 Step 6: Statistical Analysis")
            statistical_analysis = self.stats_analyzer.comprehensive_statistical_analysis(df)
            
            # Step 7: Feature engineering analysis
            self.logger.info("🔧 Step 7: Feature Engineering Analysis")
            feature_engineering = self._analyze_feature_engineering_opportunities(df, statistical_analysis)
            
            # Step 8: Dimensionality analysis
            self.logger.info("📐 Step 8: Dimensionality Analysis")
            dimensionality_analysis = self.dim_analyzer.analyze_dimensionality(df)
            
            # Step 9: Manufacturing domain analysis
            self.logger.info("🏭 Step 9: Manufacturing Domain Analysis")
            manufacturing_analysis = self.mfg_analyzer.analyze_manufacturing_patterns(df)
            
            # Step 10: ML readiness assessment
            self.logger.info("🤖 Step 10: ML Pipeline Readiness Assessment")
            ml_readiness = self._assess_ml_readiness(df, statistical_analysis, temporal_analysis)
            
            # Step 11: Visualization generation
            self.logger.info("📊 Step 11: Creating Visualizations")
            all_results = {
                'file_metadata': metadata,
                'df_shape': df.shape,
                'data_validation': validation_results,
                'data_quality': quality_metrics,
                'temporal_analysis': temporal_analysis,
                'statistical_analysis': statistical_analysis,
                'feature_engineering': feature_engineering,
                'dimensionality_analysis': dimensionality_analysis,
                'manufacturing_analysis': manufacturing_analysis,
                'ml_readiness': ml_readiness
            }
            
            viz_paths = self.viz_engine.create_comprehensive_visualizations(df, all_results)
            
            # Step 12: Report generation
            self.logger.info("📄 Step 12: Generating Report")
            report_path = self.report_generator.generate_comprehensive_report(
                df, all_results, viz_paths
            )
            
            # Final results
            final_results = {
                **all_results,
                'visualizations': viz_paths,
                'report_path': report_path,
                'processing_time_seconds': time.time() - start_time,
                'config_used': self.config.__dict__,
                'pipeline_recommendations': self._generate_pipeline_recommendations(all_results)
            }
            
            # Cleanup
            del df
            gc.collect()
            
            self.logger.info(f"✅ EDA Pipeline completed in {time.time() - start_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ EDA Pipeline failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _run_basic_eda(self, file_path: str, start_time: float) -> Dict[str, Any]:
        """Run basic EDA when full utils not available"""
        self.logger.info("🔄 Running basic EDA (utils not available)")
        
        try:
            # Basic data loading
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path_obj.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            self.logger.info(f"📊 Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Basic analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            missing_data = df.isnull().sum()
            
            # Basic statistics
            basic_stats = {}
            if len(numeric_cols) > 0:
                basic_stats['numeric'] = df[numeric_cols].describe().to_dict()
            
            if len(categorical_cols) > 0:
                basic_stats['categorical'] = {}
                for col in categorical_cols[:10]:  # Limit for performance
                    basic_stats['categorical'][col] = df[col].value_counts().head().to_dict()
            
            # Data quality assessment
            total_cells = len(df) * len(df.columns)
            missing_cells = missing_data.sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100
            
            basic_results = {
                'df_shape': df.shape,
                'data_types': df.dtypes.value_counts().to_dict(),
                'missing_data': missing_data.to_dict(),
                'basic_stats': basic_stats,
                'data_quality': {
                    'completeness_score': completeness_score,
                    'missing_data_percentage': (missing_cells / total_cells) * 100,
                    'duplicate_rows': df.duplicated().sum(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
                },
                'processing_time_seconds': time.time() - start_time,
                'status': 'basic_analysis_only',
                'message': 'Basic analysis completed. For full analysis, ensure eda_utils.py is available.'
            }
            
            # Create basic visualizations if matplotlib available
            try:
                self._create_basic_matplotlib_plots(df)
                basic_results['visualizations_created'] = True
            except Exception as e:
                self.logger.warning(f"⚠️ Basic visualizations failed: {e}")
                basic_results['visualizations_created'] = False
            
            self.logger.info(f"✅ Basic EDA completed in {time.time() - start_time:.2f} seconds")
            return basic_results
            
        except Exception as e:
            self.logger.error(f"❌ Basic EDA failed: {e}")
            raise
    
    def _create_basic_matplotlib_plots(self, df: pd.DataFrame):
        """Create basic plots using matplotlib when utils not available"""
        try:
            output_dir = Path(self.config.output_dir)
            
            # 1. Data overview
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Data types pie chart
            dtype_counts = df.dtypes.value_counts()
            axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Data Types Distribution')
            
            # Missing values bar chart
            missing_counts = df.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            if len(missing_cols) > 0:
                missing_cols.head(10).plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Top 10 Columns with Missing Values')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
                axes[0, 1].set_title('Missing Values')
            
            # Dataset size info
            axes[1, 0].text(0.1, 0.8, f'Rows: {df.shape[0]:,}', fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].text(0.1, 0.6, f'Columns: {df.shape[1]}', fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].text(0.1, 0.4, f'Memory: {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB', fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Dataset Overview')
            axes[1, 0].axis('off')
            
            # Feature types
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
            
            feature_types = ['Numeric', 'Categorical']
            feature_counts = [numeric_count, categorical_count]
            axes[1, 1].bar(feature_types, feature_counts)
            axes[1, 1].set_title('Feature Types')
            axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'basic_overview.png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # 2. Numeric features distribution (first 6 numeric columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                
                if len(numeric_cols) == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for idx, col in enumerate(numeric_cols):
                    row = idx // n_cols
                    col_pos = idx % n_cols
                    
                    ax = axes[row, col_pos] if n_rows > 1 else axes[col_pos]
                    
                    data = df[col].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                        ax.set_title(f'{col}')
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'basic_distributions.png', dpi=100, bbox_inches='tight')
                plt.close()
            
            # 3. Basic correlation heatmap (if enough numeric columns)
            if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                numeric_df = df.select_dtypes(include=[np.number])
                
                # Limit to first 15 columns for visualization
                if len(numeric_df.columns) > 15:
                    numeric_df = numeric_df.iloc[:, :15]
                
                corr_matrix = numeric_df.corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                           square=True, fmt='.2f', cbar_kws={"shrink": .8})
                plt.title('Correlation Matrix (First 15 Numeric Features)')
                plt.tight_layout()
                plt.savefig(output_dir / 'basic_correlation.png', dpi=100, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"📊 Basic visualizations saved to: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Basic matplotlib plots creation failed: {e}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe structure and content"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if dataframe is empty
            if df.empty:
                validation['valid'] = False
                validation['errors'].append("Dataframe is empty")
                return validation
            
            # Check minimum size
            if df.shape[0] < 10:
                validation['warnings'].append("Very small dataset (< 10 rows)")
            
            if df.shape[1] < 2:
                validation['warnings'].append("Very few features (< 2 columns)")
            
            # Check for all missing columns
            all_missing_cols = df.columns[df.isnull().all()].tolist()
            if all_missing_cols:
                validation['warnings'].append(f"Columns with all missing values: {all_missing_cols}")
            
            # Check for constant columns
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                validation['warnings'].append(f"Constant columns: {constant_cols}")
            
            # Check data types
            if len(df.select_dtypes(include=[np.number]).columns) == 0:
                validation['warnings'].append("No numeric columns found")
            
            self.logger.info(f"📋 Data validation: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings")
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Validation failed: {str(e)}")
        
        return validation
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced data quality assessment"""
        quality_metrics = {}
        
        try:
            # Basic quality metrics
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            
            quality_metrics.update({
                'completeness_score': ((total_cells - missing_cells) / total_cells) * 100,
                'missing_data_percentage': (missing_cells / total_cells) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                'data_types': df.dtypes.value_counts().to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
            })
            
            # Quality issues identification
            quality_issues = []
            if quality_metrics['missing_data_percentage'] > 20:
                quality_issues.append("High missing data percentage (>20%)")
            
            if quality_metrics['duplicate_percentage'] > 5:
                quality_issues.append("High duplicate percentage (>5%)")
            
            # Constant columns
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                quality_issues.append(f"Constant columns found: {len(constant_cols)}")
            
            quality_metrics['quality_issues'] = quality_issues
            quality_metrics['constant_columns'] = constant_cols
            
            self.logger.info(f"📊 Quality score: {quality_metrics['completeness_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            quality_metrics['error'] = str(e)
        
        return quality_metrics
    
    def _analyze_feature_engineering_opportunities(self, df: pd.DataFrame, 
                                                  stats_results: Dict) -> Dict[str, Any]:
        """Identify feature engineering opportunities"""
        opportunities = {
            'transformation_recommendations': [],
            'interaction_opportunities': [],
            'aggregation_opportunities': [],
            'encoding_recommendations': []
        }
        
        try:
            # Transformation recommendations based on distribution analysis
            if 'distribution_analysis' in stats_results:
                for col, analysis in stats_results['distribution_analysis'].items():
                    params = analysis.get('distribution_params', {})
                    skewness = params.get('skewness', 0)
                    
                    if abs(skewness) > 2:  # Highly skewed
                        if skewness > 0:
                            opportunities['transformation_recommendations'].append({
                                'column': col,
                                'issue': 'right_skewed',
                                'recommendations': ['log_transform', 'sqrt_transform', 'box_cox']
                            })
                        else:
                            opportunities['transformation_recommendations'].append({
                                'column': col,
                                'issue': 'left_skewed',
                                'recommendations': ['reflect_and_log', 'yeo_johnson']
                            })
            
            # High correlation pairs for interaction features
            if 'correlation_analysis' in stats_results:
                high_corr = stats_results['correlation_analysis'].get('high_correlation_pairs', [])
                for pair in high_corr[:5]:  # Top 5 pairs
                    opportunities['interaction_opportunities'].append({
                        'features': [pair['feature1'], pair['feature2']],
                        'correlation': pair['correlation'],
                        'interactions': ['product', 'ratio', 'difference']
                    })
            
            # Categorical encoding recommendations
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                cardinality = df[col].nunique()
                if cardinality > 10:
                    opportunities['encoding_recommendations'].append({
                        'column': col,
                        'cardinality': cardinality,
                        'recommendation': 'target_encoding' if cardinality > 50 else 'one_hot_encoding'
                    })
                elif cardinality <= 10:
                    opportunities['encoding_recommendations'].append({
                        'column': col,
                        'cardinality': cardinality,
                        'recommendation': 'one_hot_encoding'
                    })
            
            self.logger.info(f"🔧 Feature engineering: {len(opportunities['transformation_recommendations'])} transformations, "
                           f"{len(opportunities['interaction_opportunities'])} interactions recommended")
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering analysis failed: {e}")
            opportunities['error'] = str(e)
        
        return opportunities
    
    def _assess_ml_readiness(self, df: pd.DataFrame, stats_results: Dict, 
                           temporal_results: Dict) -> Dict[str, Any]:
        """Assess ML readiness with detailed recommendations"""
        readiness = {
            'overall_score': 0,
            'readiness_factors': {},
            'blocking_issues': [],
            'recommended_models': [],
            'preprocessing_steps': [],
            'validation_strategy': {}
        }
        
        try:
            # Calculate readiness factors
            factors = {}
            
            # Data size factor
            n_rows, n_cols = df.shape
            if n_rows >= 10000:
                factors['data_size'] = 100
            elif n_rows >= 1000:
                factors['data_size'] = 80
            else:
                factors['data_size'] = 60
            
            # Missing data factor
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct < 5:
                factors['missing_data'] = 100
            elif missing_pct < 15:
                factors['missing_data'] = 80
            elif missing_pct < 30:
                factors['missing_data'] = 60
            else:
                factors['missing_data'] = 30
                readiness['blocking_issues'].append(f"High missing data: {missing_pct:.1f}%")
            
            # Feature diversity factor
            numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
            if 0.3 <= numeric_ratio <= 0.8:
                factors['feature_diversity'] = 100
            else:
                factors['feature_diversity'] = 70
            
            # Multicollinearity factor
            high_corr_count = len(stats_results.get('correlation_analysis', {}).get('high_correlation_pairs', []))
            if high_corr_count == 0:
                factors['multicollinearity'] = 100
            elif high_corr_count <= 5:
                factors['multicollinearity'] = 80
            else:
                factors['multicollinearity'] = 60
                readiness['blocking_issues'].append(f"High multicollinearity: {high_corr_count} pairs")
            
            readiness['readiness_factors'] = factors
            readiness['overall_score'] = int(np.mean(list(factors.values())))
            
            # Model recommendations
            if temporal_results.get('time_series_detected', False):
                readiness['recommended_models'] = [
                    'LSTM', 'GRU', 'Prophet', 'ARIMA', 'XGBoost with lag features'
                ]
                readiness['validation_strategy'] = {
                    'type': 'time_series_split',
                    'method': 'walk_forward_validation',
                    'test_size': 0.2
                }
            else:
                if n_rows > 100000:
                    readiness['recommended_models'] = [
                        'XGBoost', 'LightGBM', 'CatBoost', 'Random Forest'
                    ]
                elif n_rows > 10000:
                    readiness['recommended_models'] = [
                        'Random Forest', 'XGBoost', 'SVM', 'Logistic Regression'
                    ]
                else:
                    readiness['recommended_models'] = [
                        'Logistic Regression', 'Random Forest', 'SVM'
                    ]
                
                readiness['validation_strategy'] = {
                    'type': 'cross_validation',
                    'method': 'stratified_kfold',
                    'folds': 5,
                    'test_size': 0.2
                }
            
            # Preprocessing steps
            preprocessing_steps = []
            
            if missing_pct > 5:
                preprocessing_steps.append("Handle missing values (imputation/removal)")
            
            if high_corr_count > 0:
                preprocessing_steps.append("Address multicollinearity (feature selection/PCA)")
            
            # Check for scaling needs
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scales = df[numeric_cols].std()
                if scales.max() / scales.min() > 100:
                    preprocessing_steps.append("Feature scaling (StandardScaler/RobustScaler)")
            
            preprocessing_steps.append("Encode categorical variables")
            preprocessing_steps.append("Split data (train/validation/test)")
            
            readiness['preprocessing_steps'] = preprocessing_steps
            
            self.logger.info(f"🤖 ML Readiness: {readiness['overall_score']}/100, "
                           f"{len(readiness['blocking_issues'])} blocking issues")
            
        except Exception as e:
            self.logger.error(f"❌ ML readiness assessment failed: {e}")
            readiness['error'] = str(e)
        
        return readiness
    
    def _generate_pipeline_recommendations(self, results: Dict) -> Dict[str, Any]:
        """Generate comprehensive ML pipeline recommendations"""
        recommendations = {
            'data_preparation': [],
            'feature_engineering': [],
            'model_selection': [],
            'evaluation_metrics': [],
            'deployment_considerations': []
        }
        
        try:
            ml_readiness = results['ml_readiness']
            
            # Data preparation recommendations
            recommendations['data_preparation'].extend(ml_readiness['preprocessing_steps'])
            
            # Feature engineering recommendations
            feature_eng = results.get('feature_engineering', {})
            if feature_eng.get('transformation_recommendations'):
                recommendations['feature_engineering'].append("Apply distribution transformations")
            
            if feature_eng.get('interaction_opportunities'):
                recommendations['feature_engineering'].append("Create interaction features")
            
            # Model selection recommendations
            recommendations['model_selection'] = ml_readiness['recommended_models']
            
            # Evaluation metrics
            temporal_detected = results.get('temporal_analysis', {}).get('time_series_detected', False)
            manufacturing_detected = bool(results.get('manufacturing_analysis', {}).get('quality_patterns'))
            
            if temporal_detected:
                recommendations['evaluation_metrics'] = [
                    'MAE', 'RMSE', 'MAPE', 'Directional Accuracy'
                ]
            elif manufacturing_detected:
                recommendations['evaluation_metrics'] = [
                    'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Confusion Matrix'
                ]
            else:
                recommendations['evaluation_metrics'] = [
                    'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cross-validation Score'
                ]
            
            # Deployment considerations
            df_shape = results['df_shape']
            if df_shape[0] > 1000000:
                recommendations['deployment_considerations'] = [
                    'Consider batch processing for large datasets',
                    'Implement feature caching for real-time predictions',
                    'Use model compression techniques',
                    'Plan for distributed inference'
                ]
            else:
                recommendations['deployment_considerations'] = [
                    'Standard deployment patterns suitable',
                    'Consider containerization (Docker)',
                    'Implement model versioning',
                    'Set up monitoring and alerting'
                ]
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline recommendations failed: {e}")
            recommendations['error'] = str(e)
        
        return recommendations


def quick_eda(file_path: str, output_dir: str = None) -> str:
    """Quick EDA function for programmatic usage"""
    try:
        if output_dir is None:
            output_dir = f"./quick_eda_{int(time.time())}"
        
        if UTILS_AVAILABLE:
            config = EDAConfig(output_dir=output_dir)
        else:
            # Fallback config
            @dataclass
            class QuickConfig:
                output_dir: str = output_dir
                random_state: int = 42
            config = QuickConfig()
        
        pipeline = ScientificEDAPipeline(config)
        results = pipeline.run_comprehensive_eda(file_path)
        
        return results.get('report_path', 'Basic analysis completed')
        
    except Exception as e:
        print(f"❌ Quick EDA failed: {e}")
        raise


def main():
    """Enhanced main function with better error handling"""
    print("🔬 Scientific EDA Pipeline v2.0 for Manufacturing Data")
    print("=" * 70)
    print("Enhanced with 2024 ML Best Practices & Manufacturing Domain Expertise")
    print("=" * 70)
    
    try:
        # Get user inputs with validation
        while True:
            file_path = input("📂 Enter parquet/csv file path: ").strip().strip('"')
            if Path(file_path).exists():
                break
            print(f"❌ File not found: {file_path}")
        
        # Configuration options
        print("\n⚙️ Configuration Options:")
        print("1. Default settings (recommended)")
        print("2. Custom settings")
        print("3. Manufacturing-optimized settings")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "2":
            # Custom configuration
            output_dir = input("📁 Output directory (default: ./eda_results): ").strip() or "./eda_results"
            
            if UTILS_AVAILABLE:
                missing_threshold = float(input("📊 Missing data threshold (default: 0.5): ") or "0.5")
                correlation_threshold = float(input("🔗 Correlation threshold (default: 0.9): ") or "0.9")
                
                config = EDAConfig(
                    output_dir=output_dir,
                    missing_threshold=missing_threshold,
                    correlation_threshold=correlation_threshold
                )
            else:
                config = None
                
        elif choice == "3":
            # Manufacturing-optimized settings
            timestamp = int(time.time())
            if UTILS_AVAILABLE:
                config = EDAConfig(
                    output_dir=f"./manufacturing_eda_{timestamp}",
                    missing_threshold=0.3,
                    correlation_threshold=0.85,
                    outlier_threshold=2.5,
                    manufacturing_focused=True
                )
            else:
                config = None
        else:
            # Default configuration
            timestamp = int(time.time())
            if UTILS_AVAILABLE:
                config = EDAConfig(output_dir=f"./eda_results_{timestamp}")
            else:
                config = None
        
        output_dir = config.output_dir if config else f"./eda_results_{int(time.time())}"
        print(f"\n🚀 Starting analysis with output directory: {output_dir}")
        
        # Run the pipeline
        pipeline = ScientificEDAPipeline(config)
        results = pipeline.run_comprehensive_eda(file_path)
        
        # Display results
        print("\n" + "=" * 70)
        print("🎉 EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Quick summary
        df_shape = results['df_shape']
        processing_time = results['processing_time_seconds']
        
        print(f"📊 Dataset: {df_shape[0]:,} rows × {df_shape[1]} columns")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        if 'ml_readiness' in results:
            ml_readiness = results['ml_readiness']['overall_score']
            print(f"🤖 ML Readiness Score: {ml_readiness}/100")
        
        if 'report_path' in results:
            print(f"📄 Main report: {results['report_path']}")
        
        print(f"📁 Results directory: {output_dir}")
        
        # Next steps
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review the generated report and visualizations")
        print(f"   2. Follow preprocessing recommendations")
        print(f"   3. Implement suggested feature engineering")
        print(f"   4. Design ML pipeline with recommended models")
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()