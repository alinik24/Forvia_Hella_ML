"""
Scientific Exploratory Data Analysis (EDA) Pipeline - Main Module
Advanced Manufacturing Data Analysis Framework

Based on Latest Research & Best Practices (2024):
- NIST Statistical Handbook for Outlier Detection
- MLOps Continuous Delivery Patterns (Google Cloud, 2024)
- Manufacturing Innovation Trends (AWS, 2024)
- Statistical Outlier Detection Methods (2024)

References:
- Montgomery, D.C. (2013). Statistical Quality Control
- Little, R.J. & Rubin, D.B. (2019). Statistical Analysis with Missing Data
- Wickham, H. (2014). Tidy Data Principles
- NIST Engineering Statistics Handbook
- Google Cloud MLOps Architecture Center (2024)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Statistical and ML libraries
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, jarque_bera
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score

# Import utilities
from eda_utils import (
    EDAConfig, DataProfiler, DataLoader, TimeSeriesDetector,
    StatisticalAnalyzer, VisualizationEngine, DimensionalityAnalyzer,
    ManufacturingDomainAnalyzer, ReportGenerator
)

# Advanced visualization and missing data analysis
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class ScientificEDAPipeline:
    """
    Main pipeline orchestrator for scientific EDA
    
    Implements comprehensive data analysis pipeline following
    MLOps best practices and manufacturing domain expertise.
    """
    
    def __init__(self, config: EDAConfig = None):
        self.config = config or EDAConfig()
        
        # Initialize all analysis modules
        self.profiler = DataProfiler(self.config)
        self.loader = DataLoader(self.config, self.profiler.logger)
        self.time_detector = TimeSeriesDetector(self.config, self.profiler.logger)
        self.stats_analyzer = StatisticalAnalyzer(self.config, self.profiler.logger)
        self.viz_engine = VisualizationEngine(self.config, self.profiler.logger)
        self.dim_analyzer = DimensionalityAnalyzer(self.config, self.profiler.logger)
        self.mfg_analyzer = ManufacturingDomainAnalyzer(self.config, self.profiler.logger)
        self.report_generator = ReportGenerator(self.config, self.profiler.logger)
        
        self.logger = self.profiler.logger
    
    def run_comprehensive_eda(self, file_path: str) -> Dict[str, Any]:
        """
        Run comprehensive EDA pipeline following MLOps best practices
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            Dictionary containing all analysis results and ML pipeline recommendations
        """
        self.logger.info("🚀 Starting Scientific EDA Pipeline v2.0")
        start_time = time.time()
        
        try:
            # Step 1: File metadata profiling (Memory-aware loading)
            self.logger.info("📋 Step 1: Advanced File Metadata Profiling")
            metadata = self.profiler.profile_file_metadata(file_path)
            
            # Step 2: Smart data loading with memory optimization
            self.logger.info("📂 Step 2: Memory-Optimized Data Loading")
            df = self.loader.load_data_smart(file_path, metadata)
            
            # Step 3: Data quality assessment (Early validation)
            self.logger.info("🔍 Step 3: Data Quality Pre-Assessment")
            quality_metrics = self._assess_data_quality(df)
            
            # Step 4: Temporal analysis (Time series detection)
            self.logger.info("🕐 Step 4: Advanced Temporal Pattern Detection")
            temporal_analysis = self.time_detector.detect_temporal_features(df)
            
            # Step 5: Comprehensive statistical analysis
            self.logger.info("📊 Step 5: Advanced Statistical Analysis")
            statistical_analysis = self.stats_analyzer.comprehensive_statistical_analysis(df)
            
            # Step 6: Feature engineering recommendations
            self.logger.info("🔧 Step 6: Feature Engineering Analysis")
            feature_engineering = self._analyze_feature_engineering_opportunities(df, statistical_analysis)
            
            # Step 7: Dimensionality analysis
            self.logger.info("🔍 Step 7: Dimensionality & Clustering Analysis")
            dimensionality_analysis = self.dim_analyzer.analyze_dimensionality(df)
            
            # Step 8: Manufacturing domain analysis
            self.logger.info("🏭 Step 8: Manufacturing Domain Analysis")
            manufacturing_analysis = self.mfg_analyzer.analyze_manufacturing_patterns(df)
            
            # Step 9: ML readiness assessment
            self.logger.info("🤖 Step 9: ML Pipeline Readiness Assessment")
            ml_readiness = self._assess_ml_readiness(df, statistical_analysis, temporal_analysis)
            
            # Step 10: Comprehensive visualizations
            self.logger.info("📊 Step 10: Creating Advanced Visualizations")
            all_results = {
                'file_metadata': metadata,
                'df_shape': df.shape,
                'data_quality': quality_metrics,
                'temporal_analysis': temporal_analysis,
                'statistical_analysis': statistical_analysis,
                'feature_engineering': feature_engineering,
                'dimensionality_analysis': dimensionality_analysis,
                'manufacturing_analysis': manufacturing_analysis,
                'ml_readiness': ml_readiness
            }
            
            viz_paths = self.viz_engine.create_comprehensive_visualizations(df, all_results)
            
            # Step 11: Generate comprehensive report with ML recommendations
            self.logger.info("📄 Step 11: Generating Advanced Report & ML Recommendations")
            report_path = self.report_generator.generate_comprehensive_report(
                df, all_results, viz_paths
            )
            
            # Final results with ML pipeline guidance
            final_results = {
                **all_results,
                'visualizations': viz_paths,
                'report_path': report_path,
                'processing_time_seconds': time.time() - start_time,
                'config_used': self.config.__dict__,
                'pipeline_recommendations': self._generate_pipeline_recommendations(all_results)
            }
            
            # Memory cleanup
            del df
            gc.collect()
            
            self.logger.info(f"✅ EDA Pipeline completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"📄 Report generated: {report_path}")
            self.logger.info(f"📁 Results directory: {self.config.output_dir}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ EDA Pipeline failed: {str(e)}")
            raise
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Early data quality assessment based on best practices
        """
        quality_metrics = {}
        
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
        
        # Identify potential issues
        quality_issues = []
        if quality_metrics['missing_data_percentage'] > 20:
            quality_issues.append("High missing data percentage (>20%)")
        
        if quality_metrics['duplicate_percentage'] > 5:
            quality_issues.append("High duplicate percentage (>5%)")
        
        # Check for single-value columns (no variance)
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_issues.append(f"Constant columns found: {len(constant_cols)}")
        
        quality_metrics['quality_issues'] = quality_issues
        quality_metrics['constant_columns'] = constant_cols
        
        return quality_metrics
    
    def _analyze_feature_engineering_opportunities(self, df: pd.DataFrame, 
                                                  stats_results: Dict) -> Dict[str, Any]:
        """
        Identify feature engineering opportunities based on data characteristics
        """
        opportunities = {
            'transformation_recommendations': [],
            'interaction_opportunities': [],
            'aggregation_opportunities': [],
            'encoding_recommendations': []
        }
        
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
        
        return opportunities
    
    def _assess_ml_readiness(self, df: pd.DataFrame, stats_results: Dict, 
                           temporal_results: Dict) -> Dict[str, Any]:
        """
        Assess ML readiness and provide detailed recommendations
        """
        readiness = {
            'overall_score': 0,
            'readiness_factors': {},
            'blocking_issues': [],
            'recommended_models': [],
            'preprocessing_steps': [],
            'validation_strategy': {}
        }
        
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
        
        # Model recommendations based on characteristics
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
        
        # Preprocessing recommendations
        preprocessing_steps = []
        
        if missing_pct > 5:
            preprocessing_steps.append("Handle missing values (imputation/removal)")
        
        if high_corr_count > 0:
            preprocessing_steps.append("Address multicollinearity (feature selection/PCA)")
        
        # Check for scaling needs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scales = df[numeric_cols].std()
            if scales.max() / scales.min() > 100:  # Large scale differences
                preprocessing_steps.append("Feature scaling (StandardScaler/RobustScaler)")
        
        preprocessing_steps.append("Encode categorical variables")
        preprocessing_steps.append("Split data (train/validation/test)")
        
        readiness['preprocessing_steps'] = preprocessing_steps
        
        return readiness
    
    def _generate_pipeline_recommendations(self, results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive ML pipeline recommendations
        """
        recommendations = {
            'data_preparation': [],
            'feature_engineering': [],
            'model_selection': [],
            'evaluation_metrics': [],
            'deployment_considerations': []
        }
        
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
        
        # Evaluation metrics based on problem type
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
        if df_shape[0] > 1000000:  # Large dataset
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
        
        return recommendations


# Utility functions for standalone usage
class EDAUtils:
    """Utility functions for working with EDA results"""
    
    @staticmethod
    def load_eda_results(results_dir: str) -> Dict:
        """Load EDA results from directory"""
        results_path = Path(results_dir) / "eda_summary.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No EDA results found in {results_dir}")
    
    @staticmethod
    def get_preprocessing_recommendations(results_dir: str) -> List[str]:
        """Get preprocessing recommendations from EDA results"""
        results = EDAUtils.load_eda_results(results_dir)
        
        recommendations = []
        
        # Missing data recommendations
        missing_pct = results.get('data_quality', {}).get('missing_data_percentage', 0)
        if missing_pct > 10:
            recommendations.append("Handle missing data with imputation or removal")
        
        # Correlation recommendations
        high_corr_pairs = results.get('quality_metrics', {}).get('high_correlation_pairs', 0)
        if high_corr_pairs > 0:
            recommendations.append(f"Address {high_corr_pairs} highly correlated feature pairs")
        
        # ML readiness recommendations
        ml_recommendations = results.get('pipeline_recommendations', {})
        if ml_recommendations:
            recommendations.extend(ml_recommendations.get('data_preparation', []))
        
        return recommendations

    @staticmethod
    def get_model_recommendations(results_dir: str) -> Dict[str, Any]:
        """Get model recommendations from EDA results"""
        results = EDAUtils.load_eda_results(results_dir)
        
        return {
            'recommended_models': results.get('ml_readiness', {}).get('recommended_models', []),
            'validation_strategy': results.get('ml_readiness', {}).get('validation_strategy', {}),
            'evaluation_metrics': results.get('pipeline_recommendations', {}).get('evaluation_metrics', [])
        }


def quick_eda(file_path: str, output_dir: str = None) -> str:
    """Quick EDA function for programmatic usage"""
    if output_dir is None:
        output_dir = f"./quick_eda_{int(time.time())}"
    
    config = EDAConfig(output_dir=output_dir)
    pipeline = ScientificEDAPipeline(config)
    
    results = pipeline.run_comprehensive_eda(file_path)
    return results['report_path']


def main():
    """Main function for running the EDA pipeline"""
    print("🔬 Scientific EDA Pipeline v2.0 for Manufacturing Data")
    print("=" * 70)
    print("Enhanced with 2024 ML Best Practices & Manufacturing Domain Expertise")
    print("=" * 70)
    
    # Get user inputs
    file_path = input("📂 Enter parquet file path: ").strip().strip('"')
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Configuration options
    print("\n⚙️ Configuration Options:")
    print("1. Default settings (recommended)")
    print("2. Custom settings")
    print("3. Manufacturing-optimized settings")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "2":
        # Custom configuration
        output_dir = input("📁 Output directory (default: ./eda_results): ").strip() or "./eda_results"
        missing_threshold = float(input("🔍 Missing data threshold (default: 0.5): ") or "0.5")
        correlation_threshold = float(input("🔗 Correlation threshold (default: 0.9): ") or "0.9")
        
        config = EDAConfig(
            output_dir=output_dir,
            missing_threshold=missing_threshold,
            correlation_threshold=correlation_threshold
        )
    elif choice == "3":
        # Manufacturing-optimized settings
        timestamp = int(time.time())
        config = EDAConfig(
            output_dir=f"./manufacturing_eda_{timestamp}",
            missing_threshold=0.3,  # Stricter for manufacturing
            correlation_threshold=0.85,  # More sensitive to correlation
            outlier_threshold=2.5,  # More conservative outlier detection
            manufacturing_focused=True
        )
    else:
        # Default configuration
        timestamp = int(time.time())
        config = EDAConfig(output_dir=f"./eda_results_{timestamp}")
    
    print(f"\n🚀 Starting analysis with output directory: {config.output_dir}")
    
    try:
        # Run the pipeline
        pipeline = ScientificEDAPipeline(config)
        results = pipeline.run_comprehensive_eda(file_path)
        
        # Display comprehensive summary
        print("\n" + "=" * 70)
        print("🎉 EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Quick summary
        df_shape = results['df_shape']
        processing_time = results['processing_time_seconds']
        ml_readiness = results['ml_readiness']['overall_score']
        
        print(f"📊 Dataset: {df_shape[0]:,} rows × {df_shape[1]} columns")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"🤖 ML Readiness Score: {ml_readiness}/100")
        print(f"📁 Results directory: {config.output_dir}")
        print(f"📄 Main report: {results['report_path']}")
        
        # ML Pipeline Recommendations
        pipeline_recs = results.get('pipeline_recommendations', {})
        if pipeline_recs:
            print(f"\n🚀 ML Pipeline Recommendations:")
            recommended_models = pipeline_recs.get('model_selection', [])
            if recommended_models:
                print(f"   📈 Recommended Models: {', '.join(recommended_models[:3])}")
            
            validation_strategy = results['ml_readiness'].get('validation_strategy', {})
            if validation_strategy:
                print(f"   ✅ Validation Strategy: {validation_strategy.get('method', 'cross_validation')}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review the comprehensive report and visualizations")
        print(f"   2. Follow preprocessing recommendations")
        print(f"   3. Implement suggested feature engineering")
        print(f"   4. Design ML pipeline with recommended models")
        print(f"   5. Set up proper validation strategy")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        print(f"Check the log file for detailed error information")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()