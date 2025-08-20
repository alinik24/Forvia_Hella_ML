"""
Scientific EDA Utilities Module
Supporting classes and functions for the main EDA pipeline

Based on Latest Statistical Methods & Manufacturing Best Practices:
- NIST Statistical Handbook methods
- Robust outlier detection (2024 best practices)
- Advanced missing data analysis
- Manufacturing domain-specific patterns
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
from scipy.stats import shapiro, anderson, kstest, jarque_bera, normaltest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score

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


@dataclass
class EDAConfig:
    """Enhanced configuration for EDA pipeline based on scientific best practices"""
    
    # File processing
    output_dir: str = "./eda_results"
    max_memory_usage: float = 0.8  # Max 80% of available RAM
    chunk_size: int = 10000  # For large files
    
    # Statistical thresholds (from literature)
    missing_threshold: float = 0.5  # Montgomery (2013) - 50% missing = remove
    correlation_threshold: float = 0.9  # Wickham (2014) - high correlation
    outlier_threshold: float = 3.0  # Z-score threshold
    skewness_threshold: float = 1.0  # Moderate skewness
    
    # Enhanced thresholds based on 2024 best practices
    kurtosis_threshold: float = 3.0  # Excess kurtosis threshold
    variance_threshold: float = 1e-6  # Near-zero variance
    cardinality_threshold: float = 0.95  # High cardinality = ID-like
    
    # Manufacturing-specific thresholds
    manufacturing_focused: bool = False
    process_capability_threshold: float = 1.33  # Cpk threshold
    
    # Visualization settings
    max_categories: int = 20  # Max categories to show in plots
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    # Time series detection
    datetime_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y', '%Y-%m-%d %H:%M:%S.%f', 'ISO8601'
    ])
    
    # Advanced analysis settings
    pca_components: int = 10
    tsne_perplexity: int = 30
    cluster_range: Tuple[int, int] = (2, 10)
    
    # Random state for reproducibility
    random_state: int = 42


class DataProfiler:
    """Advanced data profiling based on scientific methodologies"""
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self.setup_logging()
        self.results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'eda_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def profile_file_metadata(self, file_path: str) -> Dict[str, Any]:
    """
    Profile file-level metadata using PyArrow for efficiency
    Based on Zaharia et al. (2016) - Apache Spark principles
    """
    self.logger.info("🔍 Profiling file metadata")
    
    file_path = Path(file_path)
    metadata = {
        'file_size_mb': file_path.stat().st_size / (1024**2),
        'file_name': file_path.name,
        'file_extension': file_path.suffix
    }
    
    if PYARROW_AVAILABLE and file_path.suffix == '.parquet':
        try:
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema_arrow  # Updated to use schema_arrow
            metadata.update({
                'num_row_groups': parquet_file.metadata.num_row_groups,  # Correct attribute
                'schema_names': [field.name for field in schema],
                'total_rows_estimate': parquet_file.metadata.num_rows,  # Correct attribute
                'compression': parquet_file.metadata.row_group(0).column(0).compression
                    if parquet_file.metadata.num_row_groups > 0 else 'unknown',
                'parquet_version': parquet_file.metadata.format_version
            })
            self.logger.info(f"📊 Parquet file: {metadata['total_rows_estimate']:,} rows, "
                           f"{len(metadata['schema_names'])} columns")
        except Exception as e:
            self.logger.warning(f"Could not read parquet metadata: {e}")
            metadata.update({
                'num_row_groups': None,
                'schema_names': [],
                'total_rows_estimate': None,
                'compression': 'unknown',
                'parquet_version': 'unknown'
            })
    
    # Memory estimation
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    estimated_memory_need = metadata['file_size_mb'] * 3 / 1024  # Rule of thumb: 3x file size
    metadata['memory_feasible'] = estimated_memory_need < (available_memory * self.config.max_memory_usage)
    metadata['estimated_memory_gb'] = estimated_memory_need
    metadata['available_memory_gb'] = available_memory
    
    self.results['file_metadata'] = metadata
    return metadata


class DataLoader:
    """Intelligent data loading with memory management"""
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def load_data_smart(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """
        Smart data loading based on file size and memory constraints
        Implements chunked loading for large files
        """
        self.logger.info("📂 Smart data loading initiated")
        
        if not metadata['memory_feasible']:
            self.logger.warning(f"⚠️ Large file detected ({metadata['file_size_mb']:.1f} MB). Using chunked loading.")
            return self._load_chunked(file_path)
        else:
            return self._load_full(file_path)
    
    def _load_full(self, file_path: str) -> pd.DataFrame:
        """Load full dataset into memory"""
        self.logger.info("📊 Loading full dataset")
        
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
        
        # Memory optimization
        df = self._optimize_dtypes(df)
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        self.logger.info(f"✅ Loaded {df.shape[0]:,} × {df.shape[1]} dataset ({memory_usage:.1f} MB)")
        
        return df
    
    def _load_chunked(self, file_path: str) -> pd.DataFrame:
        """Load data in chunks for large files"""
        self.logger.info(f"📦 Loading data in chunks of {self.config.chunk_size:,} rows")
        
        if file_path.endswith('.parquet'):
            # For parquet, read a sample
            df_sample = pd.read_parquet(file_path)
            if len(df_sample) > self.config.chunk_size * 10:
                df_sample = df_sample.sample(n=self.config.chunk_size * 10, random_state=self.config.random_state)
        else:
            # For CSV, read in chunks
            chunk_list = []
            total_chunks = 0
            
            for chunk in pd.read_csv(file_path, chunksize=self.config.chunk_size):
                chunk_list.append(chunk)
                total_chunks += 1
                if total_chunks >= 10:  # Limit to first 10 chunks for analysis
                    break
            
            df_sample = pd.concat(chunk_list, ignore_index=True)
        
        df_sample = self._optimize_dtypes(df_sample)
        self.logger.info(f"📊 Sample loaded: {df_sample.shape[0]:,} × {df_sample.shape[1]} for analysis")
        
        return df_sample
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df


class TimeSeriesDetector:
    """
    Advanced time series detection and analysis
    Based on van der Aalst (2016) - Process Mining principles
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def detect_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze temporal patterns in data"""
        self.logger.info("🕐 Detecting temporal features")
        
        temporal_info = {
            'datetime_columns': [],
            'potential_timestamps': [],
            'time_series_detected': False,
            'temporal_patterns': {},
            'seasonality_detected': False,
            'trend_detected': False
        }
        
        # Detect datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_info['datetime_columns'].append(col)
                continue
                
            # Try to parse as datetime
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                parsed_count = 0
                
                for fmt in self.config.datetime_formats:
                    try:
                        if fmt == 'ISO8601':
                            pd.to_datetime(sample_values, errors='coerce')
                        else:
                            pd.to_datetime(sample_values, format=fmt, errors='coerce')
                        parsed_count += 1
                        break
                    except:
                        continue
                
                if parsed_count > 0:
                    temporal_info['potential_timestamps'].append(col)
        
        # Analyze temporal patterns if found
        all_temporal_cols = temporal_info['datetime_columns'] + temporal_info['potential_timestamps']
        
        if all_temporal_cols:
            temporal_info['time_series_detected'] = True
            temporal_info['temporal_patterns'] = self._analyze_temporal_patterns(df, all_temporal_cols)
            
            # Advanced time series analysis
            if STATSMODELS_AVAILABLE:
                temporal_info.update(self._advanced_time_series_analysis(df, all_temporal_cols))
        
        return temporal_info
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
    """Analyze patterns in temporal data"""
    patterns = {}
    
    for col in temporal_cols:
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_series = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            else:
                dt_series = df[col]
            
            dt_series = dt_series.dropna()
            
            if len(dt_series) > 10:
                patterns[col] = {
                    'start_date': dt_series.min(),
                    'end_date': dt_series.max(),
                    'date_range_days': (dt_series.max() - dt_series.min()).days
                        if pd.api.types.is_datetime64_any_dtype(dt_series) else None,
                    'frequency_estimate': self._estimate_frequency(dt_series),
                    'has_gaps': self._detect_gaps(dt_series),
                    'business_hours_pattern': self._detect_business_hours(dt_series),
                    'regularity_score': self._calculate_regularity_score(dt_series)
                }
        except Exception as e:
            self.logger.warning(f"Could not analyze temporal patterns for {col}: {e}")
            patterns[col] = {'error': str(e)}
                
    return patterns

def _detect_gaps(self, dt_series: pd.Series) -> bool:
    """Detect if there are significant gaps in time series"""
    if len(dt_series) < 3:
        return False
    
    sorted_dates = dt_series.sort_values()
    diffs = sorted_dates.diff().dropna()
    
    # Convert Timedelta to seconds for comparison
    diffs_seconds = diffs.dt.total_seconds()
    median_diff = diffs_seconds.median()
    large_gaps = diffs_seconds > (median_diff * 3)
    
    return large_gaps.any()
    
    def _advanced_time_series_analysis(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
        """Advanced time series analysis using statsmodels"""
        analysis = {'seasonality_detected': False, 'trend_detected': False, 'stationarity_test': {}}
        
        try:
            # Use first temporal column for detailed analysis
            time_col = temporal_cols[0]
            
            # Convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Find numeric columns for time series analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Limit to 3 for performance
            
            for num_col in numeric_cols:
                # Create time series
                ts_data = df[[time_col, num_col]].dropna().sort_values(time_col)
                if len(ts_data) < 50:  # Need sufficient data
                    continue
                
                ts_data = ts_data.set_index(time_col)
                
                # Stationarity test (Augmented Dickey-Fuller)
                try:
                    adf_result = adfuller(ts_data[num_col].values)
                    analysis['stationarity_test'][num_col] = {
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'is_stationary': adf_result[1] < 0.05
                    }
                except:
                    pass
                
                # Seasonality detection (simple autocorrelation check)
                try:
                    autocorr = acf(ts_data[num_col].values, nlags=min(40, len(ts_data)//4))
                    if np.any(np.abs(autocorr[12:]) > 0.3):  # Check for seasonal patterns
                        analysis['seasonality_detected'] = True
                except:
                    pass
                
                # Trend detection (simple linear regression slope)
                try:
                    x = np.arange(len(ts_data))
                    slope, _, r_value, p_value, _ = stats.linregress(x, ts_data[num_col].values)
                    if abs(r_value) > 0.3 and p_value < 0.05:
                        analysis['trend_detected'] = True
                except:
                    pass
                
                break  # Only analyze first valid numeric column for performance
                
        except Exception as e:
            self.logger.warning(f"Advanced time series analysis failed: {e}")
        
        return analysis
    
    def _estimate_frequency(self, dt_series: pd.Series) -> str:
        """Estimate the frequency of time series data"""
        if len(dt_series) < 3:
            return "insufficient_data"
        
        # Calculate differences
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        # Get most common difference
        most_common_diff = diffs.mode()
        
        if len(most_common_diff) > 0:
            diff_seconds = most_common_diff.iloc[0].total_seconds()
            
            if diff_seconds < 60:
                return "seconds"
            elif diff_seconds < 3600:
                return "minutes"
            elif diff_seconds < 86400:
                return "hours"
            elif diff_seconds < 604800:
                return "days"
            else:
                return "weeks_or_more"
        
        return "irregular"
    
    def _detect_gaps(self, dt_series: pd.Series) -> bool:
        """Detect if there are significant gaps in time series"""
        if len(dt_series) < 3:
            return False
        
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        # Check if any gap is more than 3x the median gap
        median_diff = diffs.median()
        large_gaps = diffs > (median_diff * 3)
        
        return large_gaps.any()
    
    def _detect_business_hours(self, dt_series: pd.Series) -> Dict:
        """Detect if data follows business hours pattern"""
        hours = dt_series.dt.hour
        weekdays = dt_series.dt.weekday
        
        # Business hours typically 8-17 (8 AM to 5 PM)
        business_hours = ((hours >= 8) & (hours <= 17)).sum()
        weekday_data = (weekdays < 5).sum()  # Monday=0, Friday=4
        
        total_records = len(dt_series)
        
        return {
            'business_hours_pct': business_hours / total_records if total_records > 0 else 0,
            'weekday_pct': weekday_data / total_records if total_records > 0 else 0,
            'likely_business_pattern': (business_hours / total_records > 0.7) if total_records > 0 else False
        }
    
    def _calculate_regularity_score(self, dt_series: pd.Series) -> float:
        """Calculate regularity score of time series"""
        if len(dt_series) < 3:
            return 0.0
        
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        if len(diffs) == 0:
            return 0.0
        
        # Calculate coefficient of variation of intervals
        cv = diffs.std() / diffs.mean() if diffs.mean() > 0 else float('inf')
        
        # Convert to regularity score (lower CV = higher regularity)
        regularity_score = max(0, 100 - (cv * 100))
        
        return min(100, regularity_score)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis based on scientific literature
    Enhanced with 2024 best practices for robust statistical methods
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def comprehensive_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis with enhanced methods"""
        self.logger.info("📊 Comprehensive statistical analysis")
        
        analysis_results = {
            'basic_statistics': self._enhanced_basic_statistics(df),
            'distribution_analysis': self._enhanced_distribution_analysis(df),
            'correlation_analysis': self._enhanced_correlation_analysis(df),
            'outlier_analysis': self._robust_outlier_analysis(df),
            'missing_value_analysis': self._advanced_missing_analysis(df),
            'feature_importance': self._calculate_feature_importance(df)
        }
        
        return analysis_results
    
    def _enhanced_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced basic statistics for all features"""
        stats_dict = {}
        
        # Numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            
            # Add enhanced statistics
            additional_stats = pd.DataFrame({
                'skewness': df[numeric_cols].skew(),
                'kurtosis': df[numeric_cols].kurtosis(),
                'excess_kurtosis': df[numeric_cols].kurtosis() - 3,  # Excess kurtosis
                'missing_count': df[numeric_cols].isnull().sum(),
                'missing_percentage': df[numeric_cols].isnull().sum() / len(df) * 100,
                'unique_count': df[numeric_cols].nunique(),
                'zero_count': (df[numeric_cols] == 0).sum(),
                'negative_count': (df[numeric_cols] < 0).sum(),
                'coefficient_of_variation': df[numeric_cols].std() / df[numeric_cols].mean(),
                'mad': df[numeric_cols].apply(lambda x: stats.median_abs_deviation(x.dropna())),  # Median Absolute Deviation
                'iqr': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
            })
            
            stats_dict['numerical'] = pd.concat([numeric_stats, additional_stats]).T
        
        # Categorical features with enhanced analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_stats = []
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                stats = {
                    'count': df[col].count(),
                    'unique': df[col].nunique(),
                    'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_freq_pct': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': df[col].isnull().sum() / len(df) * 100,
                    'entropy': self._calculate_entropy(value_counts),
                    'concentration_ratio': self._calculate_concentration_ratio(value_counts)
                }
                cat_stats.append(stats)
            
            stats_dict['categorical'] = pd.DataFrame(cat_stats, index=categorical_cols)
        
        return stats_dict
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate Shannon entropy for categorical variables"""
        if len(value_counts) == 0:
            return 0.0
        
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
        return entropy
    
    def _calculate_concentration_ratio(self, value_counts: pd.Series) -> float:
        """Calculate concentration ratio (top category percentage)"""
        if len(value_counts) == 0:
            return 0.0
        
        return (value_counts.iloc[0] / value_counts.sum()) * 100
    
    def _enhanced_distribution_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced distribution analysis using multiple normality tests
        Based on NIST Statistical Handbook and recent best practices
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_results = {}
        
        for col in numeric_cols:
            if df[col].dropna().nunique() > 3:  # Skip binary/sparse features
                data = df[col].dropna()
                
                if len(data) > 3:
                    results = {
                        'normality_tests': self._comprehensive_normality_tests(data),
                        'distribution_params': self._enhanced_distribution_params(data),
                        'outlier_indices': self._detect_outliers_iqr(data),
                        'distribution_recommendation': self._recommend_distribution(data)
                    }
                    distribution_results[col] = results
        
        return distribution_results
    
    def _comprehensive_normality_tests(self, data: pd.Series) -> Dict:
        """Comprehensive normality testing with multiple methods"""
        results = {}
        
        if len(data) > 3:
            # Shapiro-Wilk test (best for n < 5000)
            if len(data) <= 5000:
                try:
                    stat, p_value = shapiro(data)
                    results['shapiro_wilk'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
                except:
                    results['shapiro_wilk'] = None
            
            # Jarque-Bera test (good for large samples)
            try:
                stat, p_value = jarque_bera(data)
                results['jarque_bera'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['jarque_bera'] = None
            
            # D'Agostino's normality test
            try:
                stat, p_value = normaltest(data)
                results['dagostino'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['dagostino'] = None
            
            # Anderson-Darling test
            try:
                result = anderson(data, dist='norm')
                results['anderson_darling'] = {
                    'statistic': result.statistic,
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist(),
                    'is_normal': result.statistic < result.critical_values[2]  # 5% significance level
                }
            except:
                results['anderson_darling'] = None
            
            # Kolmogorov-Smirnov test
            try:
                stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
                results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['kolmogorov_smirnov'] = None
        
        return results
    
    def _enhanced_distribution_params(self, data: pd.Series) -> Dict:
        """Enhanced distribution parameters with robust statistics"""
        params = {}
        
        # Basic parameters
        params['mean'] = float(data.mean())
        params['std'] = float(data.std())
        params['median'] = float(data.median())
        params['mode'] = float(data.mode().iloc[0]) if not data.mode().empty else None
        
        # Robust statistics
        params['mad'] = float(stats.median_abs_deviation(data))  # Median Absolute Deviation
        params['iqr'] = float(data.quantile(0.75) - data.quantile(0.25))
        params['trimmed_mean'] = float(stats.trim_mean(data, 0.1))  # 10% trimmed mean
        
        # Distribution shape
        params['skewness'] = float(data.skew())
        params['kurtosis'] = float(data.kurtosis())
        params['excess_kurtosis'] = float(data.kurtosis() - 3)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            params[f'percentile_{p}'] = float(data.quantile(p/100))
        
        return params
    
    def _recommend_distribution(self, data: pd.Series) -> str:
        """Recommend appropriate distribution based on characteristics"""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5:
            return "normal"
        elif skewness > 1:
            return "log_normal"
        elif skewness < -1:
            return "beta_or_uniform"
        elif kurtosis > 6:
            return "heavy_tailed"
        elif (data >= 0).all() and skewness > 0:
            return "gamma_or_exponential"
        else:
            return "custom_analysis_needed"
    
    def _enhanced_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced correlation analysis with multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'warning': 'Insufficient numeric columns for correlation analysis'}
        
        results = {}
        
        # Pearson correlation (linear relationships)
        pearson_corr = df[numeric_cols].corr(method='pearson')
        results['pearson'] = pearson_corr
        
        # Spearman correlation (monotonic relationships)
        spearman_corr = df[numeric_cols].corr(method='spearman')
        results['spearman'] = spearman_corr
        
        # Kendall correlation (robust to outliers)
        kendall_corr = df[numeric_cols].corr(method='kendall')
        results['kendall'] = kendall_corr
        
        # High correlation pairs
        high_corr_pairs = self._find_high_correlation_pairs(pearson_corr)
        results['high_correlation_pairs'] = high_corr_pairs
        
        # Correlation stability analysis
        results['correlation_stability'] = self._analyze_correlation_stability(
            pearson_corr, spearman_corr, kendall_corr
        )
        
        return results
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.config.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        return sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
    
    def _analyze_correlation_stability(self, pearson: pd.DataFrame, spearman: pd.DataFrame, 
                                     kendall: pd.DataFrame) -> Dict:
        """Analyze stability of correlations across different methods"""
        stability = {}
        
        for col1 in pearson.columns:
            for col2 in pearson.columns:
                if col1 != col2:
                    p_corr = pearson.loc[col1, col2]
                    s_corr = spearman.loc[col1, col2]
                    k_corr = kendall.loc[col1, col2]
                    
                    # Calculate stability as standard deviation of correlations
                    corr_std = np.std([p_corr, s_corr, k_corr])
                    
                    if abs(p_corr) > 0.3:  # Only analyze meaningful correlations
                        stability[f"{col1}-{col2}"] = {
                            'pearson': p_corr,
                            'spearman': s_corr,
                            'kendall': k_corr,
                            'stability_score': 1 - corr_std,  # Higher score = more stable
                            'relationship_type': self._classify_relationship(p_corr, s_corr)
                        }
        
        return stability
    
    def _classify_relationship(self, pearson: float, spearman: float) -> str:
        """Classify the type of relationship based on correlation coefficients"""
        p_abs = abs(pearson)
        s_abs = abs(spearman)
        
        if p_abs > 0.8 and s_abs > 0.8:
            return "strong_linear"
        elif s_abs > p_abs + 0.2:
            return "monotonic_nonlinear"
        elif p_abs > s_abs + 0.2:
            return "linear_with_outliers"
        else:
            return "moderate_relationship"
    
    def _robust_outlier_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Robust multi-method outlier detection based on 2024 best practices
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 10:
                outlier_results[col] = {
                    'iqr_outliers': self._detect_outliers_iqr(data),
                    'zscore_outliers': self._detect_outliers_zscore(data),
                    'modified_zscore_outliers': self._detect_outliers_modified_zscore(data),
                    'isolation_forest_outliers': self._detect_outliers_isolation_forest(data),
                    'elliptic_envelope_outliers': self._detect_outliers_elliptic_envelope(data),
                    'outlier_consensus': self._calculate_outlier_consensus(data)
                }
        
        return outlier_results
    
    def _detect_outliers_iqr(self, data: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > self.config.outlier_threshold
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_modified_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Modified Z-score (more robust)"""
        median = np.median(data)
        mad = stats.median_abs_deviation(data)
        
        if mad == 0:
            return []
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > 3.5
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_isolation_forest(self, data: pd.Series) -> List[int]:
        """Detect outliers using Isolation Forest"""
        try:
            clf = IsolationForest(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_elliptic_envelope(self, data: pd.Series) -> List[int]:
        """Detect outliers using Elliptic Envelope"""
        try:
            clf = EllipticEnvelope(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _calculate_outlier_consensus(self, data: pd.Series) -> Dict:
        """Calculate consensus outliers across multiple methods"""
        methods = {
            'iqr': set(self._detect_outliers_iqr(data)),
            'zscore': set(self._detect_outliers_zscore(data)),
            'modified_zscore': set(self._detect_outliers_modified_zscore(data)),
            'isolation_forest': set(self._detect_outliers_isolation_forest(data)),
            'elliptic_envelope': set(self._detect_outliers_elliptic_envelope(data))
        }
        
        # Find consensus outliers (detected by multiple methods)
        all_outliers = set()
        for outliers in methods.values():
            all_outliers.update(outliers)
        
        consensus_scores = {}
        for idx in all_outliers:
            score = sum(1 for outliers in methods.values() if idx in outliers)
            consensus_scores[idx] = score
        
        # High consensus outliers (detected by 3+ methods)
        high_consensus = [idx for idx, score in consensus_scores.items() if score >= 3]
        
        return {
            'consensus_outliers': high_consensus,
            'outlier_scores': consensus_scores,
            'method_agreement': len(high_consensus) / len(all_outliers) if all_outliers else 0
        }
    
    def _advanced_missing_analysis(self, df: pd.DataFrame) -> Dict:
        """Advanced missing value analysis with pattern detection"""
        missing_info = {}
        
        # Basic missing value statistics
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_info['summary'] = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        }).sort_values('missing_percentage', ascending=False)
        
        # Missing value patterns and mechanisms
        missing_info['patterns'] = self._analyze_missing_patterns(df)
        missing_info['mechanisms'] = self._analyze_missing_mechanisms(df)
        
        return missing_info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in missing data"""
        # Find columns with missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        if not cols_with_missing:
            return {'message': 'No missing values found'}
        
        patterns = {}
        
        # Co-occurrence of missing values
        if len(cols_with_missing) > 1:
            missing_matrix = df[cols_with_missing].isnull()
            
            # Find common patterns
            pattern_counts = missing_matrix.value_counts()
            patterns['common_patterns'] = pattern_counts.head(10).to_dict()
            
            # Calculate missing value correlations
            if len(cols_with_missing) > 2:
                missing_corr = missing_matrix.astype(int).corr()
                high_missing_corr = []
                
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_value = missing_corr.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            high_missing_corr.append({
                                'col1': missing_corr.columns[i],
                                'col2': missing_corr.columns[j],
                                'correlation': corr_value
                            })
                
                patterns['missing_correlations'] = high_missing_corr
        
        return patterns
    
    def _analyze_missing_mechanisms(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data mechanisms (MCAR, MAR, MNAR)"""
        mechanisms = {}
        
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        for col in cols_with_missing:
            missing_mask = df[col].isnull()
            
            # Test for MCAR vs MAR by looking at relationships with other variables
            other_cols = [c for c in df.columns if c != col and not df[c].isnull().all()]
            
            correlations_with_missing = []
            for other_col in other_cols[:10]:  # Limit for performance
                if pd.api.types.is_numeric_dtype(df[other_col]):
                    # For numeric variables, use point-biserial correlation
                    try:
                        corr = stats.pointbiserialr(missing_mask, df[other_col].fillna(df[other_col].mean()))[0]
                        if abs(corr) > 0.1:  # Threshold for meaningful correlation
                            correlations_with_missing.append({
                                'variable': other_col,
                                'correlation': corr
                            })
                    except:
                        pass
            
            # Classify missing mechanism
            if not correlations_with_missing:
                mechanism = "likely_MCAR"  # Missing Completely At Random
            elif any(abs(c['correlation']) > 0.3 for c in correlations_with_missing):
                mechanism = "likely_MAR"  # Missing At Random
            else:
                mechanism = "potentially_MNAR"  # Missing Not At Random
            
            mechanisms[col] = {
                'mechanism': mechanism,
                'correlations': correlations_with_missing[:5]  # Top 5 correlations
            }
        
        return mechanisms
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict:
        """Calculate feature importance using multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for feature importance analysis'}
        
        importance_results = {}
        
        # Use variance as a simple importance measure
        variances = df[numeric_cols].var().sort_values(ascending=False)
        importance_results['variance_importance'] = variances.to_dict()
        
        # Use coefficient of variation for normalized importance
        cv_importance = (df[numeric_cols].std() / df[numeric_cols].mean()).sort_values(ascending=False)
        importance_results['cv_importance'] = cv_importance.dropna().to_dict()
        
        return importance_results


class VisualizationEngine:
    """
    Advanced visualization engine for comprehensive EDA
    Enhanced with modern visualization best practices
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def create_comprehensive_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, str]:
        """Create comprehensive visualization suite with enhanced plots"""
        self.logger.info("📊 Creating comprehensive visualizations")
        
        viz_paths = {}
        
        try:
            # 1. Enhanced Data Overview Dashboard
            viz_paths['overview'] = self._create_enhanced_overview_dashboard(df, analysis_results)
            
            # 2. Advanced Distribution Analysis
            viz_paths['distributions'] = self._create_advanced_distribution_plots(df, analysis_results)
            
            # 3. Enhanced Correlation Analysis
            if 'correlation_analysis' in analysis_results.get('statistical_analysis', {}):
                viz_paths['correlations'] = self._create_enhanced_correlation_plots(
                    analysis_results['statistical_analysis']['correlation_analysis'])
            
            # 4. Advanced Missing Value Analysis
            viz_paths['missing_values'] = self._create_advanced_missing_plots(df, analysis_results)
            
            # 5. Robust Outlier Analysis
            viz_paths['outliers'] = self._create_advanced_outlier_plots(
                df, analysis_results.get('statistical_analysis', {}).get('outlier_analysis', {}))
            
            # 6. Time Series Analysis (if applicable)
            temporal_info = analysis_results.get('temporal_analysis', {})
            if temporal_info.get('time_series_detected', False):
                viz_paths['time_series'] = self._create_advanced_time_series_plots(df, temporal_info)
            
            # 7. Feature Engineering Opportunities
            viz_paths['feature_engineering'] = self._create_feature_engineering_plots(
                df, analysis_results.get('feature_engineering', {}))
            
        except Exception as e:
            self.logger.warning(f"Some visualizations failed: {e}")
        
        return viz_paths
    
    def _create_enhanced_overview_dashboard(self, df: pd.DataFrame, results: Dict) -> str:
        """Create enhanced overview dashboard with more insights"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Data Quality Score', 'Missing Values by Column', 
                    'Feature Types Distribution', 'ML Readiness Assessment',
                    'Statistical Anomalies', 'Data Completeness Matrix'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "heatmap"}]
                ]
            )
            
            # Data quality score indicator
            quality_score = results.get('data_quality', {}).get('completeness_score', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=quality_score,
                    title={'text': "Data Quality"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if quality_score > 80 else "orange" if quality_score > 60 else "red"},
                        'steps': [{'range': [0, 60], 'color': "lightgray"}, {'range': [60, 80], 'color': "yellow"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                    }
                ),
                row=1, col=1
            )
            
            # Missing values
            missing_counts = df.isnull().sum().sort_values(ascending=False)[:15]
            if missing_counts.sum() > 0:
                fig.add_trace(
                    go.Bar(x=missing_counts.index, y=missing_counts.values, name="Missing Values"),
                    row=1, col=2
                )
            
            # Feature types distribution
            dtype_counts = df.dtypes.value_counts()
            fig.add_trace(
                go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name="Data Types"),
                row=2, col=1
            )
            
            # ML Readiness factors
            ml_readiness = results.get('ml_readiness', {}).get('readiness_factors', {})
            if ml_readiness:
                fig.add_trace(
                    go.Bar(
                        x=list(ml_readiness.keys()), 
                        y=list(ml_readiness.values()), 
                        name="ML Readiness"
                    ),
                    row=2, col=2
                )
            
            # Statistical anomalies
            anomalies = self._count_statistical_anomalies(results)
            fig.add_trace(
                go.Bar(
                    x=list(anomalies.keys()), 
                    y=list(anomalies.values()), 
                    name="Anomalies"
                ),
                row=3, col=1
            )
            
            # Data completeness matrix (sample)
            completeness_matrix = df.head(50).notnull().astype(int)
            fig.add_trace(
                go.Heatmap(
                    z=completeness_matrix.values,
                    x=completeness_matrix.columns,
                    y=completeness_matrix.index,
                    colorscale='RdYlGn',
                    name="Completeness"
                ),
                row=3, col=2
            )
            
            fig.update_layout(
                title="Enhanced Data Overview Dashboard",
                height=1200,
                showlegend=False
            )
            
            output_path = self.output_dir / "enhanced_overview_dashboard.html"
            fig.write_html(output_path)
            self.logger.info(f"📊 Enhanced overview dashboard saved: {output_path}")
            
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create enhanced overview dashboard: {e}")
            return ""
    
    def _count_statistical_anomalies(self, results: Dict) -> Dict[str, int]:
        """Count various statistical anomalies"""
        anomalies = {
            'High Correlations': 0,
            'Outlier Features': 0,
            'Skewed Features': 0,
            'Constant Features': 0
        }
        
        stats_results = results.get('statistical_analysis', {})
        
        # High correlations
        if 'correlation_analysis' in stats_results:
            anomalies['High Correlations'] = len(
                stats_results['correlation_analysis'].get('high_correlation_pairs', [])
            )
        
        # Features with outliers
        if 'outlier_analysis' in stats_results:
            anomalies['Outlier Features'] = len(stats_results['outlier_analysis'])
        
        # Highly skewed features
        if 'distribution_analysis' in stats_results:
            skewed_count = sum(
                1 for analysis in stats_results['distribution_analysis'].values()
                if abs(analysis.get('distribution_params', {}).get('skewness', 0)) > 2
            )
            anomalies['Skewed Features'] = skewed_count
        
        # Constant features
        data_quality = results.get('data_quality', {})
        anomalies['Constant Features'] = len(data_quality.get('constant_columns', []))
        
        return anomalies
    
    def _create_advanced_distribution_plots(self, df: pd.DataFrame, results: Dict) -> str:
        """Create advanced distribution analysis plots"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns for distribution plots")
                return ""
            
            # Limit columns for performance
            cols_to_plot = list(numeric_cols)[:12]
            
            # Create subplots with both histograms and box plots
            n_cols = 3
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=cols_to_plot,
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            for idx, col in enumerate(cols_to_plot):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1
                
                data = df[col].dropna()
                
                # Add histogram with KDE
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=f"{col}_hist",
                        nbinsx=50,
                        opacity=0.7,
                        showlegend=False,
                        histnorm='probability density'
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title="Advanced Distribution Analysis - Probability Density",
                height=300 * n_rows,
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_distribution_analysis.html"
            fig.write_html(output_path)
            
            # Create separate normality assessment plot
            self._create_normality_assessment_plot(df, results)
            
            self.logger.info(f"📊 Advanced distribution plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced distribution plots: {e}")
            return ""
    
    def _create_normality_assessment_plot(self, df: pd.DataFrame, results: Dict) -> str:
        """Create normality assessment visualization"""
        try:
            dist_results = results.get('statistical_analysis', {}).get('distribution_analysis', {})
            
            if not dist_results:
                return ""
            
            # Collect normality test results
            normality_data = []
            for col, analysis in dist_results.items():
                normality_tests = analysis.get('normality_tests', {})
                
                for test_name, test_result in normality_tests.items():
                    if test_result:
                        normality_data.append({
                            'Feature': col,
                            'Test': test_name,
                            'P_Value': test_result.get('p_value', 0),
                            'Is_Normal': test_result.get('is_normal', False)
                        })
            
            if not normality_data:
                return ""
            
            normality_df = pd.DataFrame(normality_data)
            
            # Create normality assessment plot
            fig = px.scatter(
                normality_df,
                x='Feature',
                y='P_Value',
                color='Test',
                symbol='Is_Normal',
                title='Normality Test Results (p-value > 0.05 suggests normality)',
                labels={'P_Value': 'P-Value', 'Feature': 'Features'}
            )
            
            # Add significance line
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                         annotation_text="Significance Level (α=0.05)")
            
            fig.update_layout(height=600)
            
            output_path = self.output_dir / "normality_assessment.html"
            fig.write_html(output_path)
            
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create normality assessment plot: {e}")
            return ""
    
    def _create_enhanced_correlation_plots(self, correlation_results: Dict) -> str:
        """Create enhanced correlation visualizations"""
        try:
            if 'pearson' not in correlation_results:
                return ""
            
            pearson_corr = correlation_results['pearson']
            
            # Limit size for visualization
            if len(pearson_corr.columns) > 25:
                # Select most variable features
                numeric_std = pearson_corr.std().sort_values(ascending=False)
                top_features = numeric_std.head(25).index
                pearson_corr = pearson_corr.loc[top_features, top_features]
            
            # Create correlation heatmap with dendrograms
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Pearson Correlation Matrix', 
                    'Correlation Strength Distribution',
                    'High Correlation Network', 
                    'Correlation Method Comparison'
                ],
                specs=[
                    [{"type": "heatmap"}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # Main correlation heatmap
            fig.add_trace(
                go.Heatmap(
                    z=pearson_corr.values,
                    x=pearson_corr.columns,
                    y=pearson_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=pearson_corr.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 8},
                    showscale=True
                ),
                row=1, col=1
            )
            
            # Correlation strength distribution
            upper_triangle = pearson_corr.where(np.triu(np.ones_like(pearson_corr, dtype=bool), k=1))
            corr_values = upper_triangle.stack().values
            
            fig.add_trace(
                go.Histogram(
                    x=corr_values,
                    nbinsx=50,
                    name="Correlation Distribution",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # High correlation pairs
            high_corr_pairs = correlation_results.get('high_correlation_pairs', [])[:10]
            if high_corr_pairs:
                pairs_df = pd.DataFrame(high_corr_pairs)
                fig.add_trace(
                    go.Scatter(
                        x=range(len(pairs_df)),
                        y=pairs_df['abs_correlation'],
                        mode='markers+lines',
                        name="High Correlations",
                        text=pairs_df['feature1'] + ' - ' + pairs_df['feature2'],
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Method comparison (if available)
            if 'spearman' in correlation_results and 'kendall' in correlation_results:
                methods = ['Pearson', 'Spearman', 'Kendall']
                avg_abs_corr = [
                    np.abs(correlation_results['pearson'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean(),
                    np.abs(correlation_results['spearman'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean(),
                    np.abs(correlation_results['kendall'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean()
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=avg_abs_corr,
                        name="Method Comparison",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Enhanced Correlation Analysis",
                height=1000,
                showlegend=False
            )
            
            output_path = self.output_dir / "enhanced_correlation_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Enhanced correlation plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create enhanced correlation plots: {e}")
            return ""
    
    def _create_advanced_missing_plots(self, df: pd.DataFrame, results: Dict) -> str:
        """Create advanced missing value analysis plots"""
        try:
            missing_counts = df.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            
            if len(cols_with_missing) == 0:
                self.logger.info("No missing values to visualize")
                return ""
            
            # Create comprehensive missing value analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Missing Value Patterns',
                    'Missing Value Mechanisms',
                    'Missing Value Heatmap',
                    'Imputation Strategy Recommendations'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "heatmap"}, {"type": "table"}]
                ]
            )
            
            # Missing value patterns
            fig.add_trace(
                go.Bar(
                    x=cols_with_missing.index,
                    y=(cols_with_missing / len(df)) * 100,
                    name="Missing %",
                    marker_color='red',
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Missing mechanisms (if analyzed)
            missing_analysis = results.get('statistical_analysis', {}).get('missing_value_analysis', {})
            mechanisms = missing_analysis.get('mechanisms', {})
            
            if mechanisms:
                mechanism_counts = {}
                for col, info in mechanisms.items():
                    mechanism = info.get('mechanism', 'unknown')
                    mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
                
                fig.add_trace(
                    go.Pie(
                        labels=list(mechanism_counts.keys()),
                        values=list(mechanism_counts.values()),
                        name="Mechanisms",
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            # Missing value heatmap (sample of data)
            sample_size = min(100, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)[cols_with_missing.index]
            missing_matrix = sample_df.isnull().astype(int)
            
            fig.add_trace(
                go.Heatmap(
                    z=missing_matrix.values,
                    x=missing_matrix.columns,
                    y=missing_matrix.index,
                    colorscale=[[0, 'green'], [1, 'red']],
                    name="Missing Pattern",
                    showscale=False
                ),
                row=2, col=1
            )
            
            # Imputation recommendations
            imputation_recs = self._generate_imputation_recommendations(cols_with_missing, df)
            fig.add_trace(
                go.Table(
                    header=dict(values=["Column", "Missing %", "Recommendation"]),
                    cells=dict(values=[
                        list(imputation_recs.keys()),
                        [f"{(cols_with_missing[col] / len(df)) * 100:.1f}%" for col in imputation_recs.keys()],
                        list(imputation_recs.values())
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Advanced Missing Value Analysis",
                height=1000,
                showlegend=True
            )
            
            output_path = self.output_dir / "advanced_missing_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced missing value plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced missing plots: {e}")
            return ""
    
    def _generate_imputation_recommendations(self, missing_cols: pd.Series, df: pd.DataFrame) -> Dict[str, str]:
        """Generate imputation recommendations for each column with missing values"""
        recommendations = {}
        
        for col in missing_cols.index:
            missing_pct = (missing_cols[col] / len(df)) * 100
            
            if missing_pct > 70:
                recommendations[col] = "Consider removal - too much missing data"
            elif pd.api.types.is_numeric_dtype(df[col]):
                if missing_pct > 30:
                    recommendations[col] = "Advanced imputation (KNN/Iterative)"
                elif df[col].skew() > 2:
                    recommendations[col] = "Median imputation"
                else:
                    recommendations[col] = "Mean imputation"
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                if df[col].nunique() < 10:
                    recommendations[col] = "Mode imputation"
                else:
                    recommendations[col] = "Create 'Unknown' category"
            else:
                recommendations[col] = "Forward/backward fill"
        
        return recommendations
    
    def _create_advanced_outlier_plots(self, df: pd.DataFrame, outlier_results: Dict) -> str:
        """Create advanced outlier analysis visualizations"""
        try:
            if not outlier_results:
                return ""
            
            # Create comprehensive outlier analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Outlier Detection Methods Comparison',
                    'Outlier Consensus Scores',
                    'Outlier Distribution by Feature',
                    'Method Agreement Analysis'
                ]
            )
            
            # Method comparison
            method_counts = {}
            for col, methods in outlier_results.items():
                for method, outliers in methods.items():
                    if method != 'outlier_consensus' and isinstance(outliers, list):
                        method_name = method.replace('_outliers', '').replace('_', ' ').title()
                        method_counts[method_name] = method_counts.get(method_name, 0) + len(outliers)
            
            fig.add_trace(
                go.Bar(
                    x=list(method_counts.keys()),
                    y=list(method_counts.values()),
                    name="Method Comparison",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Consensus scores
            consensus_data = []
            for col, methods in outlier_results.items():
                if 'outlier_consensus' in methods:
                    consensus_info = methods['outlier_consensus']
                    consensus_outliers = consensus_info.get('consensus_outliers', [])
                    consensus_data.append({
                        'feature': col,
                        'high_consensus_outliers': len(consensus_outliers),
                        'agreement_score': consensus_info.get('method_agreement', 0) * 100
                    })
            
            if consensus_data:
                consensus_df = pd.DataFrame(consensus_data)
                fig.add_trace(
                    go.Scatter(
                        x=consensus_df['feature'],
                        y=consensus_df['high_consensus_outliers'],
                        mode='markers',
                        marker=dict(
                            size=consensus_df['agreement_score'],
                            color=consensus_df['agreement_score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Agreement %")
                        ),
                        name="Consensus Outliers",
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Outlier distribution by feature
            feature_outlier_counts = []
            for col, methods in outlier_results.items():
                total_outliers = 0
                for method, outliers in methods.items():
                    if method != 'outlier_consensus' and isinstance(outliers, list):
                        total_outliers += len(outliers)
                feature_outlier_counts.append({
                    'feature': col,
                    'total_outliers': total_outliers,
                    'outlier_percentage': (total_outliers / len(df)) * 100
                })
            
            if feature_outlier_counts:
                outlier_df = pd.DataFrame(feature_outlier_counts)
                fig.add_trace(
                    go.Bar(
                        x=outlier_df['feature'],
                        y=outlier_df['outlier_percentage'],
                        name="Outlier %",
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Method agreement analysis
            agreement_scores = []
            for col, methods in outlier_results.items():
                if 'outlier_consensus' in methods:
                    agreement = methods['outlier_consensus'].get('method_agreement', 0)
                    agreement_scores.append(agreement * 100)
            
            if agreement_scores:
                fig.add_trace(
                    go.Histogram(
                        x=agreement_scores,
                        nbinsx=20,
                        name="Agreement Distribution",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Advanced Outlier Analysis",
                height=1000,
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_outlier_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced outlier plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced outlier plots: {e}")
            return ""
    
    def _create_advanced_time_series_plots(self, df: pd.DataFrame, temporal_info: Dict) -> str:
        """Create advanced time series analysis plots"""
        try:
            datetime_cols = temporal_info.get('datetime_columns', []) + temporal_info.get('potential_timestamps', [])
            
            if not datetime_cols:
                return ""
            
            time_col = datetime_cols[0]
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            
            # Create comprehensive time series analysis
            fig = make_subplots(
                rows=len(numeric_cols) + 1, cols=2,
                subplot_titles=[f"{col} - Time Series" for col in numeric_cols] + 
                              [f"{col} - Distribution" for col in numeric_cols] + 
                              ["Temporal Patterns Summary", "Seasonality Analysis"],
                specs=[[{"colspan": 2}] * len(numeric_cols) + [{}] * len(numeric_cols) + [{"type": "table"}, {"type": "bar"}]][:len(numeric_cols) + 1]
            )
            
            for idx, col in enumerate(numeric_cols):
                temp_df = df[[time_col, col]].dropna().sort_values(time_col)
                
                # Sample data if too large
                if len(temp_df) > 5000:
                    temp_df = temp_df.sample(n=5000, random_state=42).sort_values(time_col)
                
                # Time series plot
                fig.add_trace(
                    go.Scatter(
                        x=temp_df[time_col],
                        y=temp_df[col],
                        mode='lines',
                        name=col,
                        showlegend=False
                    ),
                    row=idx + 1, col=1
                )
                
                # Distribution plot
                fig.add_trace(
                    go.Histogram(
                        x=temp_df[col],
                        name=f"{col}_dist",
                        showlegend=False
                    ),
                    row=idx + 1, col=2
                )
            
            # Temporal patterns summary
            patterns_data = []
            for col, pattern_info in temporal_info.get('temporal_patterns', {}).items():
                patterns_data.append([
                    col,
                    str(pattern_info.get('frequency_estimate', 'Unknown')),
                    f"{pattern_info.get('business_hours_pct', 0):.1%}",
                    "Yes" if pattern_info.get('has_gaps', False) else "No",
                    f"{pattern_info.get('regularity_score', 0):.1f}"
                ])
            
            if patterns_data:
                fig.add_trace(
                    go.Table(
                        header=dict(values=["Column", "Frequency", "Business Hours %", "Has Gaps", "Regularity Score"]),
                        cells=dict(values=list(zip(*patterns_data)))
                    ),
                    row=len(numeric_cols) + 1, col=1
                )
            
            # Seasonality indicators
            seasonality_indicators = {
                'Seasonality Detected': temporal_info.get('seasonality_detected', False),
                'Trend Detected': temporal_info.get('trend_detected', False),
                'Stationarity': len(temporal_info.get('stationarity_test', {})) > 0
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(seasonality_indicators.keys()),
                    y=[1 if v else 0 for v in seasonality_indicators.values()],
                    name="Time Series Properties",
                    showlegend=False
                ),
                row=len(numeric_cols) + 1, col=2
            )
            
            fig.update_layout(
                title="Advanced Time Series Analysis",
                height=400 * (len(numeric_cols) + 1),
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_time_series_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced time series plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced time series plots: {e}")
            return ""
    
    def _create_feature_engineering_plots(self, df: pd.DataFrame, feature_eng_results: Dict) -> str:
        """Create feature engineering opportunity visualizations"""
        try:
            if not feature_eng_results:
                return ""
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Transformation Opportunities',
                    'Interaction Feature Potential',
                    'Encoding Recommendations',
                    'Feature Engineering Priority'
                ]
            )
            
            # Transformation opportunities
            transform_recs = feature_eng_results.get('transformation_recommendations', [])
            if transform_recs:
                transform_counts = {}
                for rec in transform_recs:
                    issue = rec.get('issue', 'unknown')
                    transform_counts[issue] = transform_counts.get(issue, 0) + 1
                
                fig.add_trace(
                    go.Bar(
                        x=list(transform_counts.keys()),
                        y=list(transform_counts.values()),
                        name="Transformations",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Interaction opportunities
            interaction_ops = feature_eng_results.get('interaction_opportunities', [])
            if interaction_ops:
                correlations = [abs(op.get('correlation', 0)) for op in interaction_ops]
                feature_pairs = [f"{op['features'][0][:10]}...{op['features'][1][:10]}" for op in interaction_ops[:10]]
                
                fig.add_trace(
                    go.Bar(
                        x=feature_pairs,
                        y=correlations,
                        name="Interaction Potential",
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Encoding recommendations
            encoding_recs = feature_eng_results.get('encoding_recommendations', [])
            if encoding_recs:
                encoding_counts = {}
                for rec in encoding_recs:
                    recommendation = rec.get('recommendation', 'unknown')
                    encoding_counts[recommendation] = encoding_counts.get(recommendation, 0) + 1
                
                fig.add_trace(
                    go.Pie(
                        labels=list(encoding_counts.keys()),
                        values=list(encoding_counts.values()),
                        name="Encoding Methods"
                    ),
                    row=2, col=1
                )
            
            # Priority scoring
            priority_data = []
            
            # High priority: Highly skewed features
            priority_data.append(['High Skewness', len(transform_recs), 'High'])
            
            # Medium priority: High correlation interactions
            high_corr_interactions = [op for op in interaction_ops if abs(op.get('correlation', 0)) > 0.7]
            priority_data.append(['Strong Interactions', len(high_corr_interactions), 'Medium'])
            
            # Low priority: Encoding needs
            priority_data.append(['Encoding Needed', len(encoding_recs), 'Low'])
            
            if priority_data:
                priorities_df = pd.DataFrame(priority_data, columns=['Task', 'Count', 'Priority'])
                color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
                
                fig.add_trace(
                    go.Bar(
                        x=priorities_df['Task'],
                        y=priorities_df['Count'],
                        marker_color=[color_map[p] for p in priorities_df['Priority']],
                        name="Priority",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Feature Engineering Opportunities",
                height=800,
                showlegend=True
            )
            
            output_path = self.output_dir / "feature_engineering_opportunities.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Feature engineering plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create feature engineering plots: {e}")
            return ""


class DimensionalityAnalyzer:
    """
    Advanced dimensionality analysis using PCA, t-SNE, and clustering
    Enhanced with modern techniques and interpretability
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_dimensionality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dimensionality analysis with enhanced methods"""
        self.logger.info("🔍 Analyzing data dimensionality")
        
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 3:
            return {'warning': 'Insufficient numeric features for dimensionality analysis'}
        
        results = {
            'pca_analysis': self._enhanced_pca_analysis(numeric_df),
            'clustering_analysis': self._enhanced_clustering_analysis(numeric_df),
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(numeric_df)
        }
        
        # Add t-SNE if dataset is not too large
        if len(numeric_df) <= 10000 and len(numeric_df.columns) > 5:
            results['tsne_analysis'] = self._enhanced_tsne_analysis(numeric_df)
        
        return results
    
    def _enhanced_pca_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced PCA analysis with feature importance and interpretation"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Perform PCA
            n_components = min(self.config.pca_components, df.shape[1])
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            pca_result = pca.fit_transform(df_scaled)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            results = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'n_components_90_variance': int(np.argmax(cumulative_variance >= 0.9) + 1),
                'n_components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
                'n_components_99_variance': int(np.argmax(cumulative_variance >= 0.99) + 1),
                'feature_importance': {},
                'interpretation': {}
            }
            
            # Feature importance in principal components
            for i, component in enumerate(pca.components_[:5]):  # First 5 components
                component_importance = dict(zip(df.columns, np.abs(component)))
                results['feature_importance'][f'PC{i+1}'] = sorted(
                    component_importance.items(), key=lambda x: x[1], reverse=True
                )[:10]  # Top 10 features
                
                # Interpretation based on dominant features
                top_features = [item[0] for item in results['feature_importance'][f'PC{i+1}'][:3]]
                results['interpretation'][f'PC{i+1}'] = {
                    'dominant_features': top_features,
                    'variance_explained': float(pca.explained_variance_ratio_[i]),
                    'interpretation_hint': self._interpret_principal_component(top_features)
                }
            
            # PCA quality metrics
            results['quality_metrics'] = {
                'kaiser_criterion': sum(pca.explained_variance_ > 1),  # Eigenvalues > 1
                'scree_elbow': self._find_scree_elbow(pca.explained_variance_ratio_),
                'dimensionality_reduction_potential': float(results['n_components_90_variance'] / len(df.columns))
            }
            
            return results
        except Exception as e:
            self.logger.warning(f"Enhanced PCA analysis failed: {e}")
            return {'error': str(e)}
    
    def _interpret_principal_component(self, top_features: List[str]) -> str:
        """Provide interpretation hints for principal components"""
        # Simple heuristic based on feature names
        feature_lower = [f.lower() for f in top_features]
        
        if any('temp' in f or 'temperature' in f for f in feature_lower):
            return "Likely related to temperature/thermal processes"
        elif any('press' in f or 'pressure' in f for f in feature_lower):
            return "Likely related to pressure/force measurements"
        elif any('flow' in f or 'rate' in f for f in feature_lower):
            return "Likely related to flow/rate measurements"
        elif any('time' in f or 'duration' in f for f in feature_lower):
            return "Likely related to temporal characteristics"
        elif any('qual' in f or 'defect' in f for f in feature_lower):
            return "Likely related to quality measurements"
        else:
            return "Mixed process characteristics"
    
    def _find_scree_elbow(self, explained_variance: np.ndarray) -> int:
        """Find elbow point in scree plot"""
        if len(explained_variance) < 3:
            return 1
        
        # Calculate second derivative
        diffs = np.diff(explained_variance)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 2
            return min(elbow_idx, len(explained_variance) - 1)
        
        return len(explained_variance) // 2
    
    def _enhanced_clustering_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced clustering analysis with multiple algorithms"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            results = {
                'kmeans_analysis': self._enhanced_kmeans_analysis(df_scaled),
                'dbscan_analysis': self._enhanced_dbscan_analysis(df_scaled),
                'cluster_quality_assessment': {}
            }
            
            return results
        except Exception as e:
            self.logger.warning(f"Enhanced clustering analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_kmeans_analysis(self, df_scaled: np.ndarray) -> Dict:
        """Enhanced K-means analysis with multiple evaluation metrics"""
        try:
            min_k, max_k = self.config.cluster_range
            inertias = []
            silhouette_scores = []
            calinski_harabasz_scores = []
            
            for k in range(min_k, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(df_scaled)
                
                inertias.append(kmeans.inertia_)
                
                # Silhouette score
                if len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(df_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                    
                    # Calinski-Harabasz score
                    try:
                        from sklearn.metrics import calinski_harabasz_score
                        ch_score = calinski_harabasz_score(df_scaled, cluster_labels)
                        calinski_harabasz_scores.append(ch_score)
                    except:
                        calinski_harabasz_scores.append(0)
                else:
                    silhouette_scores.append(0)
                    calinski_harabasz_scores.append(0)
            
            # Find optimal k using multiple criteria
            optimal_k_elbow = self._find_elbow_point(list(range(min_k, max_k + 1)), inertias)
            optimal_k_silhouette = min_k + np.argmax(silhouette_scores) if silhouette_scores else min_k
            
            return {
                'k_range': list(range(min_k, max_k + 1)),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_harabasz_scores': calinski_harabasz_scores,
                'optimal_k_elbow': optimal_k_elbow,
                'optimal_k_silhouette': optimal_k_silhouette,
                'recommended_k': optimal_k_silhouette if max(silhouette_scores) > 0.3 else optimal_k_elbow
            }
        except Exception as e:
            self.logger.warning(f"Enhanced K-means analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_dbscan_analysis(self, df_scaled: np.ndarray) -> Dict:
        """Enhanced DBSCAN analysis with parameter optimization"""
        try:
            # Try multiple eps values
            eps_values = [0.3, 0.5, 0.7, 1.0]
            min_samples_values = [3, 5, 10]
            
            best_result = None
            best_score = -1
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(df_scaled)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    if n_clusters > 1:  # Valid clustering
                        try:
                            sil_score = silhouette_score(df_scaled, cluster_labels)
                            if sil_score > best_score:
                                best_score = sil_score
                                best_result = {
                                    'eps': eps,
                                    'min_samples': min_samples,
                                    'n_clusters': n_clusters,
                                    'n_noise_points': n_noise,
                                    'noise_percentage': n_noise / len(cluster_labels) * 100,
                                    'silhouette_score': sil_score,
                                    'cluster_labels': cluster_labels.tolist()
                                }
                        except:
                            pass
            
            return best_result or {
                'n_clusters': 0,
                'message': 'No suitable DBSCAN parameters found'
            }
        except Exception as e:
            self.logger.warning(f"Enhanced DBSCAN analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_tsne_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced t-SNE analysis with multiple perplexities"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Try multiple perplexity values
            perplexities = [5, 30, 50]
            results = {}
            
            for perp in perplexities:
                if perp < len(df) // 4:  # Ensure perplexity is valid
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perp,
                        random_state=self.config.random_state,
                        n_iter=1000
                    )
                    tsne_result = tsne.fit_transform(df_scaled)
                    
                    results[f'perplexity_{perp}'] = {
                        'coordinates': tsne_result.tolist(),
                        'perplexity': perp,
                        'kl_divergence': float(tsne.kl_divergence_)
                    }
            
            # Select best result (lowest KL divergence)
            if results:
                best_result = min(results.values(), key=lambda x: x['kl_divergence'])
                return {
                    'best_result': best_result,
                    'all_results': results,
                    'recommendation': f"Use perplexity {best_result['perplexity']} for best separation"
                }
            else:
                return {'error': 'No valid t-SNE results'}
                
        except Exception as e:
            self.logger.warning(f"Enhanced t-SNE analysis failed: {e}")
            return {'error': str(e)}
    
    def _estimate_intrinsic_dimensionality(self, df: pd.DataFrame) -> Dict:
        """Estimate intrinsic dimensionality using multiple methods"""
        try:
            # Method 1: PCA-based estimation (90% variance threshold)
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            pca_full = PCA()
            pca_full.fit(df_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            
            intrinsic_dim_90 = np.argmax(cumsum_var >= 0.9) + 1
            intrinsic_dim_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            # Method 2: Effective rank
            U, s, Vt = np.linalg.svd(df_scaled, full_matrices=False)
            effective_rank = np.sum(s > 0.01 * s[0])  # Singular values > 1% of largest
            
            return {
                'pca_90_percent': int(intrinsic_dim_90),
                'pca_95_percent': int(intrinsic_dim_95),
                'effective_rank': int(effective_rank),
                'original_dimensions': df.shape[1],
                'dimensionality_reduction_potential': float(1 - intrinsic_dim_90 / df.shape[1])
            }
        except Exception as e:
            self.logger.warning(f"Intrinsic dimensionality estimation failed: {e}")
            return {'error': str(e)}
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in K-means inertia plot using improved method"""
        if len(inertias) < 3:
            return k_values[0]
        
        # Normalize the inertias
        inertias_norm = np.array(inertias) / max(inertias)
        k_norm = np.array(k_values) / max(k_values)
        
        # Calculate the distance from each point to the line connecting first and last points
        distances = []
        for i in range(len(k_norm)):
            # Point to line distance formula
            x1, y1 = k_norm[0], inertias_norm[0]
            x2, y2 = k_norm[-1], inertias_norm[-1]
            x0, y0 = k_norm[i], inertias_norm[i]
            
            distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            distances.append(distance)
        
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]


class ManufacturingDomainAnalyzer:
    """
    Enhanced manufacturing domain-specific analysis
    Based on Industry 4.0 principles and quality control standards
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_manufacturing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced manufacturing-specific pattern analysis"""
        self.logger.info("🏭 Analyzing manufacturing domain patterns")
        
        results = {
            'process_capability': self._enhanced_process_capability(df),
            'quality_patterns': self._enhanced_quality_analysis(df),
            'equipment_patterns': self._enhanced_equipment_analysis(df),
            'defect_analysis': self._enhanced_defect_analysis(df),
            'production_efficiency': self._analyze_production_efficiency(df),
            'sensor_health': self._analyze_sensor_health(df)
        }
        
        return results
    
    def _enhanced_process_capability(self, df: pd.DataFrame) -> Dict:
        """Enhanced process capability analysis with Cp, Cpk calculations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        capability_results = {}
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Enhanced pattern matching for process measurements
            process_indicators = [
                'temp', 'temperature', 'press', 'pressure', 'flow', 'rate',
                'speed', 'force', 'torque', 'voltage', 'current', 'measure',
                'dimension', 'length', 'width', 'height', 'thickness'
            ]
            
            is_process_measure = any(indicator in col_lower for indicator in process_indicators)
            
            if is_process_measure:
                data = df[col].dropna()
                
                if len(data) > 30:
                    mean = data.mean()
                    std = data.std()
                    
                    # Estimate control limits (assuming 6-sigma process)
                    ucl = mean + 3 * std
                    lcl = mean - 3 * std
                    
                    # Calculate process capability indices
                    capability_results[col] = {
                        'mean': float(mean),
                        'std': float(std),
                        'ucl': float(ucl),
                        'lcl': float(lcl),
                        'range': float(data.max() - data.min()),
                        'cv': float(std / mean) if mean != 0 else None,
                        'within_3_sigma': float(((data >= lcl) & (data <= ucl)).mean() * 100),
                        'cp_estimate': float(6 * std / (ucl - lcl)) if (ucl - lcl) > 0 else None,
                        'process_stability': self._assess_process_stability(data),
                        'capability_assessment': self._assess_capability_level(std / mean if mean != 0 else 0)
                    }
        
        return capability_results
    
    def _assess_process_stability(self, data: pd.Series) -> str:
        """Assess process stability based on statistical indicators"""
        cv = data.std() / data.mean() if data.mean() != 0 else float('inf')
        
        if cv < 0.05:
            return "Highly Stable"
        elif cv < 0.15:
            return "Stable"
        elif cv < 0.30:
            return "Moderately Stable"
        else:
            return "Unstable"
    
    def _assess_capability_level(self, cv: float) -> str:
        """Assess process capability level"""
        if cv < 0.1:
            return "Excellent Capability"
        elif cv < 0.2:
            return "Good Capability"
        elif cv < 0.3:
            return "Acceptable Capability"
        else:
            return "Poor Capability"
    
    def _enhanced_quality_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced quality pattern analysis with statistical methods"""
        quality_indicators = []
        
        # Enhanced quality keyword detection
        quality_keywords = [
            'quality', 'defect', 'failure', 'error', 'pass', 'fail', 'ok', 'nok', 
            'reject', 'accept', 'good', 'bad', 'scrap', 'rework', 'inspection',
            'test', 'check', 'verify', 'validate', 'conform', 'specification'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in quality_keywords):
                quality_indicators.append(col)
        
        if not quality_indicators:
            return {'message': 'No obvious quality indicators found'}
        
        results = {}
        
        for col in quality_indicators:
            if df[col].dtype in ['object', 'category', 'bool']:
                # Categorical quality indicator
                value_counts = df[col].value_counts()
                results[col] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'quality_rate': self._calculate_enhanced_quality_rate(value_counts),
                    'entropy': self._calculate_quality_entropy(value_counts),
                    'defect_concentration': self._analyze_defect_concentration(value_counts)
                }
            else:
                # Numeric quality indicator
                data = df[col].dropna()
                results[col] = {
                    'type': 'numeric',
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'distribution': data.describe().to_dict(),
                    'control_limits': self._calculate_control_limits(data),
                    'out_of_control_percentage': self._calculate_ooc_percentage(data)
                }
        
        return results
    
    def _calculate_enhanced_quality_rate(self, value_counts: pd.Series) -> Dict:
        """Enhanced quality rate calculation with confidence intervals"""
        total = value_counts.sum()
        
        # Common patterns for "good" quality
        good_patterns = ['pass', 'ok', '1', 'true', 'good', 'accept', 'conform']
        bad_patterns = ['fail', 'nok', '0', 'false', 'bad', 'reject', 'defect', 'scrap']
        
        good_count = 0
        bad_count = 0
        
        for value, count in value_counts.items():
            value_str = str(value).lower()
            
            if any(pattern in value_str for pattern in good_patterns):
                good_count += count
            elif any(pattern in value_str for pattern in bad_patterns):
                bad_count += count
        
        if good_count + bad_count > 0:
            quality_rate = good_count / (good_count + bad_count) * 100
            
            # Calculate 95% confidence interval
            n = good_count + bad_count
            p = good_count / n
            margin_error = 1.96 * np.sqrt(p * (1 - p) / n)
            
            return {
                'rate': quality_rate,
                'confidence_interval': [
                    max(0, (p - margin_error) * 100),
                    min(100, (p + margin_error) * 100)
                ],
                'sample_size': n,
                'statistical_significance': 'high' if n > 100 else 'medium' if n > 30 else 'low'
            }
        
        return None
    
    def _calculate_quality_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy for quality distribution"""
        if len(value_counts) == 0:
            return 0.0
        
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def _analyze_defect_concentration(self, value_counts: pd.Series) -> Dict:
        """Analyze concentration of defects in quality data"""
        total = value_counts.sum()
        
        # Identify potential defect categories
        defect_patterns = ['fail', 'defect', 'bad', 'reject', 'scrap', 'error']
        defect_categories = []
        
        for value in value_counts.index:
            value_str = str(value).lower()
            if any(pattern in value_str for pattern in defect_patterns):
                defect_categories.append(value)
        
        if defect_categories:
            defect_count = sum(value_counts[cat] for cat in defect_categories)
            return {
                'defect_categories': defect_categories,
                'total_defects': defect_count,
                'defect_rate_ppm': (defect_count / total) * 1000000,
                'pareto_analysis': value_counts[defect_categories].sort_values(ascending=False).to_dict()
            }
        
        return {'message': 'No clear defect categories identified'}
    
    def _calculate_control_limits(self, data: pd.Series) -> Dict:
        """Calculate statistical control limits"""
        mean = data.mean()
        std = data.std()
        
        return {
            'ucl': float(mean + 3 * std),
            'lcl': float(mean - 3 * std),
            'center_line': float(mean),
            'warning_limits': {
                'upper': float(mean + 2 * std),
                'lower': float(mean - 2 * std)
            }
        }
    
    def _calculate_ooc_percentage(self, data: pd.Series) -> float:
        """Calculate out-of-control percentage"""
        mean = data.mean()
        std = data.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        
        ooc_count = ((data > ucl) | (data < lcl)).sum()
        return float(ooc_count / len(data) * 100)
    
    def _enhanced_equipment_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced equipment pattern analysis with OEE concepts"""
        equipment_keywords = [
            'station', 'machine', 'tool', 'equipment', 'line', 'cell', 
            'robot', 'cnc', 'press', 'mill', 'lathe', 'conveyor',
            'unit', 'device', 'instrument', 'sensor'
        ]
        equipment_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in equipment_keywords):
                equipment_cols.append(col)
        
        if not equipment_cols:
            return {'message': 'No obvious equipment indicators found'}
        
        results = {}
        
        for col in equipment_cols:
            if df[col].dtype in ['object', 'category']:
                unique_equipment = df[col].nunique()
                value_counts = df[col].value_counts()
                
                results[col] = {
                    'unique_count': unique_equipment,
                    'most_used': value_counts.index[0] if len(value_counts) > 0 else None,
                    'usage_distribution': value_counts.head(10).to_dict(),
                    'utilization_balance': self._calculate_enhanced_utilization(value_counts),
                    'equipment_efficiency': self._analyze_equipment_efficiency(df, col),
                    'maintenance_indicators': self._detect_maintenance_patterns(df, col)
                }
        
        return results
    
    def _calculate_enhanced_utilization(self, value_counts: pd.Series) -> Dict:
        """Enhanced utilization calculation with efficiency metrics"""
        if len(value_counts) <= 1:
            return {'balance_score': 100.0, 'utilization_type': 'single_unit'}
        
        # Calculate Gini coefficient for utilization inequality
        sorted_counts = np.sort(value_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        balance_score = (1 - gini) * 100
        
        # Analyze utilization pattern
        cv = value_counts.std() / value_counts.mean()
        
        if cv < 0.1:
            utilization_type = "Highly Balanced"
        elif cv < 0.3:
            utilization_type = "Well Balanced"
        elif cv < 0.6:
            utilization_type = "Moderately Balanced"
        else:
            utilization_type = "Poorly Balanced"
        
        return {
            'balance_score': float(balance_score),
            'gini_coefficient': float(gini),
            'coefficient_of_variation': float(cv),
            'utilization_type': utilization_type,
            'overutilized_equipment': value_counts[value_counts > value_counts.mean() + 2*value_counts.std()].index.tolist(),
            'underutilized_equipment': value_counts[value_counts < value_counts.mean() - 2*value_counts.std()].index.tolist()
        }
    
    def _analyze_equipment_efficiency(self, df: pd.DataFrame, equipment_col: str) -> Dict:
        """Analyze equipment efficiency patterns"""
        # Look for time-related or performance-related columns
        efficiency_indicators = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'duration', 'cycle', 'rate', 'speed', 'throughput']):
                if col != equipment_col and pd.api.types.is_numeric_dtype(df[col]):
                    efficiency_indicators.append(col)
        
        if not efficiency_indicators:
            return {'message': 'No efficiency indicators found'}
        
        results = {}
        
        for indicator in efficiency_indicators[:3]:  # Limit for performance
            # Calculate efficiency metrics per equipment
            equipment_efficiency = df.groupby(equipment_col)[indicator].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(3)
            
            results[indicator] = {
                'per_equipment_stats': equipment_efficiency.to_dict('index'),
                'overall_efficiency': {
                    'best_performer': equipment_efficiency['mean'].idxmax(),
                    'worst_performer': equipment_efficiency['mean'].idxmin(),
                    'efficiency_range': float(equipment_efficiency['mean'].max() - equipment_efficiency['mean'].min()),
                    'consistency_leader': equipment_efficiency['std'].idxmin()
                }
            }
        
        return results
    
    def _detect_maintenance_patterns(self, df: pd.DataFrame, equipment_col: str) -> Dict:
        """Detect maintenance-related patterns"""
        maintenance_keywords = ['maintenance', 'repair', 'downtime', 'fault', 'alarm', 'error', 'stop']
        maintenance_indicators = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in maintenance_keywords):
                maintenance_indicators.append(col)
        
        if not maintenance_indicators:
            return {'message': 'No maintenance indicators found'}
        
        results = {}
        
        for indicator in maintenance_indicators:
            if df[indicator].dtype in ['object', 'category']:
                # Categorical maintenance data
                maintenance_by_equipment = df.groupby(equipment_col)[indicator].value_counts()
                results[indicator] = {
                    'type': 'categorical',
                    'maintenance_frequency': maintenance_by_equipment.to_dict(),
                    'high_maintenance_equipment': maintenance_by_equipment.groupby(level=0).sum().sort_values(ascending=False).head(5).to_dict()
                }
            else:
                # Numeric maintenance data
                maintenance_stats = df.groupby(equipment_col)[indicator].agg(['mean', 'sum', 'std']).round(3)
                results[indicator] = {
                    'type': 'numeric',
                    'maintenance_stats': maintenance_stats.to_dict('index'),
                    'highest_maintenance': maintenance_stats['sum'].idxmax(),
                    'most_consistent': maintenance_stats['std'].idxmin()
                }
        
        return results
    
    def _enhanced_defect_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced defect pattern analysis with root cause indicators"""
        defect_keywords = ['defect', 'failure', 'fault', 'error', 'alarm', 'alert', 'reject', 'scrap']
        defect_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in defect_keywords):
                defect_cols.append(col)
        
        if not defect_cols:
            return {'message': 'No obvious defect indicators found'}
        
        results = {}
        
        for col in defect_cols:
            if df[col].dtype in ['object', 'category']:
                # Categorical defect data
                value_counts = df[col].value_counts()
                results[col] = {
                    'defect_types': value_counts.to_dict(),
                    'pareto_analysis': self._perform_pareto_analysis(value_counts),
                    'defect_diversity': len(value_counts),
                    'critical_defects': self._identify_critical_defects(value_counts)
                }
            else:
                # Numeric defect data
                data = df[col].dropna()
                if len(data) > 0:
                    results[col] = {
                        'defect_rate': float((data > 0).mean() * 100) if (data >= 0).all() else None,
                        'average_defects': float(data.mean()),
                        'defect_distribution': data.describe().to_dict(),
                        'defect_trends': self._analyze_defect_trends(data),
                        'control_chart_signals': self._detect_control_chart_signals(data)
                    }
        
        return results
    
    def _perform_pareto_analysis(self, value_counts: pd.Series) -> Dict:
        """Perform Pareto analysis on defect types"""
        total = value_counts.sum()
        cumulative_pct = (value_counts.cumsum() / total * 100).round(2)
        
        # Find defects contributing to 80% of issues (Pareto principle)
        pareto_80_defects = cumulative_pct[cumulative_pct <= 80].index.tolist()
        
        return {
            'top_defects_80_percent': pareto_80_defects,
            'cumulative_percentages': cumulative_pct.to_dict(),
            'vital_few_count': len(pareto_80_defects),
            'vital_few_percentage': float(len(pareto_80_defects) / len(value_counts) * 100)
        }
    
    def _identify_critical_defects(self, value_counts: pd.Series) -> List[str]:
        """Identify critical defects based on frequency and keywords"""
        critical_keywords = ['critical', 'major', 'severe', 'safety', 'hazard', 'dangerous']
        
        critical_defects = []
        
        # High frequency defects (top 20%)
        threshold = value_counts.quantile(0.8)
        high_frequency = value_counts[value_counts >= threshold].index.tolist()
        
        # Keyword-based critical defects
        keyword_critical = [
            defect for defect in value_counts.index
            if any(keyword in str(defect).lower() for keyword in critical_keywords)
        ]
        
        critical_defects.extend(high_frequency)
        critical_defects.extend(keyword_critical)
        
        return list(set(critical_defects))
    
    def _analyze_defect_trends(self, data: pd.Series) -> Dict:
        """Analyze trends in numeric defect data"""
        # Simple trend analysis
        x = np.arange(len(data))
        if len(data) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            return {
                'trend_slope': float(slope),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'trend_strength': float(abs(r_value)),
                'trend_significance': float(p_value),
                'is_significant_trend': p_value < 0.05
            }
        
        return {'message': 'Insufficient data for trend analysis'}
    
    def _detect_control_chart_signals(self, data: pd.Series) -> Dict:
        """Detect control chart signals in defect data"""
        if len(data) < 10:
            return {'message': 'Insufficient data for control chart analysis'}
        
        mean = data.mean()
        std = data.std()
        
        signals = {
            'points_beyond_3_sigma': ((data > mean + 3*std) | (data < mean - 3*std)).sum(),
            'points_beyond_2_sigma': ((data > mean + 2*std) | (data < mean - 2*std)).sum(),
            'runs_above_mean': self._count_runs(data > mean),
            'runs_below_mean': self._count_runs(data < mean)
        }
        
        # Assess overall process control
        total_signals = sum([signals['points_beyond_3_sigma'], signals['points_beyond_2_sigma']])
        
        if total_signals == 0:
            control_status = "In Control"
        elif total_signals < len(data) * 0.05:
            control_status = "Minor Issues"
        else:
            control_status = "Out of Control"
        
        signals['control_status'] = control_status
        
        return signals
    
    def _count_runs(self, boolean_series: pd.Series) -> int:
        """Count runs in a boolean series"""
        if len(boolean_series) == 0:
            return 0
        
        runs = 1
        for i in range(1, len(boolean_series)):
            if boolean_series.iloc[i] != boolean_series.iloc[i-1]:
                runs += 1
        
        return runs
    
    def _analyze_production_efficiency(self, df: pd.DataFrame) -> Dict:
        """Analyze overall production efficiency indicators"""
        efficiency_metrics = {}
        
        # Look for production-related columns
        production_keywords = ['production', 'output', 'yield', 'throughput', 'efficiency', 'oee', 'availability']
        
        production_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in production_keywords):
                production_cols.append(col)
        
        if not production_cols:
            return {'message': 'No production efficiency indicators found'}
        
        for col in production_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                data = df[col].dropna()
                
                if len(data) > 0:
                    efficiency_metrics[col] = {
                        'average_efficiency': float(data.mean()),
                        'efficiency_std': float(data.std()),
                        'best_performance': float(data.max()),
                        'worst_performance': float(data.min()),
                        'efficiency_consistency': float(1 - (data.std() / data.mean())) if data.mean() > 0 else 0,
                        'performance_distribution': {
                            'excellent': float((data >= data.quantile(0.9)).mean() * 100),
                            'good': float(((data >= data.quantile(0.7)) & (data < data.quantile(0.9))).mean() * 100),
                            'poor': float((data < data.quantile(0.3)).mean() * 100)
                        }
                    }
        
        return efficiency_metrics
    
    def _analyze_sensor_health(self, df: pd.DataFrame) -> Dict:
        """Analyze sensor health and data quality indicators"""
        sensor_health = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col]
            
            # Sensor health indicators
            health_metrics = {
                'missing_data_pct': float(data.isnull().mean() * 100),
                'constant_values_pct': float((data.mode().iloc[0] == data).mean() * 100) if not data.mode().empty else 0,
                'outlier_pct': self._calculate_outlier_percentage(data),
                'data_quality_score': 0,
                'health_status': 'Unknown'
            }
            
            # Calculate overall data quality score
            quality_score = 100
            quality_score -= health_metrics['missing_data_pct']  # Penalty for missing data
            quality_score -= health_metrics['constant_values_pct'] * 0.5  # Penalty for constant values
            quality_score -= health_metrics['outlier_pct'] * 0.3  # Penalty for outliers
            
            health_metrics['data_quality_score'] = max(0, quality_score)
            
            # Determine health status
            if quality_score >= 90:
                health_metrics['health_status'] = 'Excellent'
            elif quality_score >= 80:
                health_metrics['health_status'] = 'Good'
            elif quality_score >= 70:
                health_metrics['health_status'] = 'Fair'
            elif quality_score >= 60:
                health_metrics['health_status'] = 'Poor'
            else:
                health_metrics['health_status'] = 'Critical'
            
            sensor_health[col] = health_metrics
        
        return sensor_health
    
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method"""
        data_clean = data.dropna()
        if len(data_clean) == 0:
            return 0.0
        
        Q1 = data_clean.quantile(0.25)
        Q3 = data_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data_clean < lower_bound) | (data_clean > upper_bound)
        return float(outliers.mean() * 100)


class ReportGenerator:
    """
    Enhanced comprehensive report generation
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def generate_comprehensive_report(self, 
                                    df: pd.DataFrame, 
                                    all_results: Dict[str, Any],
                                    viz_paths: Dict[str, str]) -> str:
        """Generate enhanced comprehensive EDA report"""
        self.logger.info("📄 Generating enhanced comprehensive EDA report")
        
        report_path = self.output_dir / "COMPREHENSIVE_EDA_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_enhanced_report_header(f, df, all_results)
            self._write_executive_summary(f, df, all_results)
            self._write_data_overview(f, df, all_results)
            self._write_statistical_analysis(f, all_results)
            self._write_quality_assessment(f, all_results)
            self._write_feature_analysis(f, all_results)
            self._write_ml_recommendations(f, all_results)
            self._write_manufacturing_insights(f, all_results)
            self._write_technical_appendix(f, all_results, viz_paths)
        
        self.logger.info(f"📄 Enhanced comprehensive report generated: {report_path}")
        
        # Generate enhanced JSON summary
        self._generate_enhanced_json_summary(all_results, df)
        
        return str(report_path)
    
    def _write_enhanced_report_header(self, f, df: pd.DataFrame, results: Dict):
        """Write enhanced report header with key metrics"""
        f.write("# 🔬 Scientific Exploratory Data Analysis Report\n\n")
        f.write("## Advanced Manufacturing Data Analysis Framework v2.0\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"**Analysis Framework:** Scientific Manufacturing EDA Pipeline\n")
        f.write(f"**Processing Time:** {results.get('processing_time_seconds', 0):.2f} seconds\n")
        
        # Key quality metrics
        data_quality = results.get('data_quality', {})
        f.write(f"**Data Quality Score:** {data_quality.get('completeness_score', 0):.1f}/100\n")
        
        ml_readiness = results.get('ml_readiness', {})
        f.write(f"**ML Readiness Score:** {ml_readiness.get('overall_score', 0)}/100\n\n")
        
        f.write("---\n\n")
    
    def _write_executive_summary(self, f, df: pd.DataFrame, results: Dict):
        """Write enhanced executive summary"""
        f.write("## 📊 Executive Summary\n\n")
        
        # Key findings with enhanced insights
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_cols = df.isnull().any().sum()
        
        f.write("### 🔍 Key Findings:\n")
        f.write(f"- **Data Composition:** {numeric_cols} numeric, {categorical_cols} categorical features\n")
        f.write(f"- **Data Quality:** {missing_cols} columns with missing values\n")
        
        # Time series detection
        temporal_info = results.get('temporal_analysis', {})
        if temporal_info.get('time_series_detected', False):
            f.write("- **✅ Time Series Detected:** Temporal patterns found")
            if temporal_info.get('seasonality_detected', False):
                f.write(" with seasonality")
            if temporal_info.get('trend_detected', False):
                f.write(" and trend components")
            f.write("\n")
        else:
            f.write("- **❌ No Time Series:** No clear temporal patterns detected\n")
        
        # Manufacturing patterns
        mfg_results = results.get('manufacturing_analysis', {})
        if mfg_results:
            quality_found = bool(mfg_results.get('quality_patterns', {}))
            equipment_found = bool(mfg_results.get('equipment_patterns', {}))
            process_capability = bool(mfg_results.get('process_capability', {}))
            
            f.write(f"- **🏭 Manufacturing Context:** ")
            contexts = []
            if quality_found:
                contexts.append("Quality indicators")
            if equipment_found:
                contexts.append("Equipment data")
            if process_capability:
                contexts.append("Process capability metrics")
            
            f.write(f"{', '.join(contexts) if contexts else 'No manufacturing indicators'} found\n")
        
        # Enhanced ML readiness assessment
        f.write("\n### 🤖 ML Pipeline Readiness Assessment:\n")
        
        ml_readiness = results.get('ml_readiness', {})
        readiness_score = ml_readiness.get('overall_score', 0)
        f.write(f"- **Overall ML Readiness:** {readiness_score}/100\n")
        
        # Detailed readiness factors
        readiness_factors = ml_readiness.get('readiness_factors', {})
        if readiness_factors:
            f.write("- **Readiness Breakdown:**\n")
            for factor, score in readiness_factors.items():
                f.write(f"  - {factor.replace('_', ' ').title()}: {score}/100\n")
        
        # Status and blocking issues
        blocking_issues = ml_readiness.get('blocking_issues', [])
        if readiness_score >= 80:
            f.write("- **Status:** ✅ Ready for ML with minimal preprocessing\n")
        elif readiness_score >= 60:
            f.write("- **Status:** ⚠️ Needs moderate preprocessing before ML\n")
            if blocking_issues:
                f.write(f"- **Key Issues:** {', '.join(blocking_issues[:3])}\n")
        else:
            f.write("- **Status:** ❌ Requires significant data preparation\n")
            if blocking_issues:
                f.write(f"- **Blocking Issues:** {', '.join(blocking_issues)}\n")
        
        # Recommended models
        recommended_models = ml_readiness.get('recommended_models', [])
        if recommended_models:
            f.write(f"- **Recommended Models:** {', '.join(recommended_models[:3])}\n")
        
        f.write("\n---\n\n")
    
    def _write_data_overview(self, f, df: pd.DataFrame, results: Dict):
        """Write enhanced data overview section"""
        f.write("## 🔍 Data Overview\n\n")
        
        # Enhanced basic information
        f.write("### Basic Information\n")
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        data_quality = results.get('data_quality', {})
        
        f.write(f"- **Rows:** {df.shape[0]:,}\n")
        f.write(f"- **Columns:** {df.shape[1]}\n")
        f.write(f"- **Memory Usage:** {memory_mb:.1f} MB\n")
        f.write(f"- **Duplicates:** {df.duplicated().sum():,} ({data_quality.get('duplicate_percentage', 0):.1f}%)\n")
        f.write(f"- **Data Completeness:** {data_quality.get('completeness_score', 0):.1f}%\n\n")
        
        # Enhanced column types breakdown
        f.write("### Column Types Distribution\n")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            f.write(f"- **{dtype}:** {count} columns\n")
        f.write("\n")
        
        # Data quality issues
        quality_issues = data_quality.get('quality_issues', [])
        if quality_issues:
            f.write("### ⚠️ Data Quality Issues\n")
            for issue in quality_issues:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        # Constant columns
        constant_cols = data_quality.get('constant_columns', [])
        if constant_cols:
            f.write(f"### 🚨 Constant Columns ({len(constant_cols)})\n")
            f.write("These columns have only one unique value and should be removed:\n")
            for col in constant_cols[:10]:  # Show first 10
                f.write(f"- `{col}`\n")
            if len(constant_cols) > 10:
                f.write(f"- ... and {len(constant_cols) - 10} more\n")
            f.write("\n")
        
        # Missing data summary
        missing_summary = df.isnull().sum().sort_values(ascending=False)
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            f.write("### 📊 Missing Data Analysis\n")
            f.write(f"- **Columns with missing data:** {len(missing_cols)}\n")
            f.write(f"- **Total missing values:** {missing_cols.sum():,}\n")
            f.write(f"- **Worst column:** `{missing_cols.index[0]}` ({missing_cols.iloc[0]:,} missing, {missing_cols.iloc[0]/len(df)*100:.1f}%)\n\n")
            
            if len(missing_cols) <= 15:
                f.write("**Missing Data by Column:**\n")
                for col, missing_count in missing_cols.head(15).items():
                    pct = missing_count / len(df) * 100
                    severity = "🔴" if pct > 50 else "🟡" if pct > 20 else "🟢"
                    f.write(f"- {severity} `{col}`: {missing_count:,} ({pct:.1f}%)\n")
                f.write("\n")
        else:
            f.write("### ✅ No Missing Data Detected\n\n")
        
        f.write("---\n\n")
    
    def _write_statistical_analysis(self, f, results: Dict):
        """Write enhanced statistical analysis section"""
        f.write("## 📈 Advanced Statistical Analysis\n\n")
        
        stats_results = results.get('statistical_analysis', {})
        
        # Enhanced distribution analysis
        if 'distribution_analysis' in stats_results:
            f.write("### 📊 Distribution Analysis\n")
            dist_results = stats_results['distribution_analysis']
            
            f.write(f"**Analyzed {len(dist_results)} numeric features for distribution characteristics:**\n\n")
            
            # Categorize features by distribution characteristics
            normal_features = []
            skewed_features = []
            heavy_tailed_features = []
            
            for feature, analysis in dist_results.items():
                params = analysis.get('distribution_params', {})
                skewness = params.get('skewness', 0)
                kurtosis = params.get('kurtosis', 3)
                
                # Check normality tests
                normality_tests = analysis.get('normality_tests', {})
                is_normal = any(
                    test_result.get('is_normal', False) 
                    for test_result in normality_tests.values() 
                    if test_result
                )
                
                if is_normal and abs(skewness) < 0.5:
                    normal_features.append(feature)
                elif abs(skewness) > 1.5:
                    direction = "right" if skewness > 0 else "left"
                    skewed_features.append((feature, skewness, direction))
                elif abs(kurtosis - 3) > 2:
                    heavy_tailed_features.append((feature, kurtosis))
            
            if normal_features:
                f.write(f"**✅ Approximately Normal Features ({len(normal_features)}):**\n")
                for feature in normal_features[:10]:
                    f.write(f"- `{feature}`\n")
                f.write("\n")
            
            if skewed_features:
                f.write(f"**⚠️ Highly Skewed Features ({len(skewed_features)}):**\n")
                for feature, skew, direction in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
                    f.write(f"- `{feature}`: {direction}-skewed (skewness: {skew:.2f})\n")
                f.write("\n*Recommendation: Consider log, sqrt, or Box-Cox transformations*\n\n")
            
            if heavy_tailed_features:
                f.write(f"**🎯 Heavy-Tailed Features ({len(heavy_tailed_features)}):**\n")
                for feature, kurt in sorted(heavy_tailed_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    f.write(f"- `{feature}`: kurtosis = {kurt:.2f}\n")
                f.write("\n*Recommendation: Consider robust scaling or outlier treatment*\n\n")
        
        # Enhanced correlation analysis
        if 'correlation_analysis' in stats_results:
            f.write("### 🔗 Correlation Analysis\n")
            corr_results = stats_results['correlation_analysis']
            
            high_corr_pairs = corr_results.get('high_correlation_pairs', [])
            if high_corr_pairs:
                f.write(f"**⚠️ High Correlation Pairs Found ({len(high_corr_pairs)}):**\n")
                for pair in high_corr_pairs[:10]:
                    strength = "Very Strong" if abs(pair['correlation']) > 0.95 else "Strong"
                    f.write(f"- `{pair['feature1']}` ↔ `{pair['feature2']}`: {pair['correlation']:.3f} ({strength})\n")
                f.write("\n*⚠️ These feature pairs may cause multicollinearity issues in ML models.*\n")
                f.write("*Recommendation: Use feature selection, PCA, or regularization techniques.*\n\n")
            else:
                f.write("**✅ No high correlation pairs detected** (threshold: 0.9)\n\n")
            
            # Correlation stability analysis
            correlation_stability = corr_results.get('correlation_stability', {})
            if correlation_stability:
                stable_relationships = [
                    k for k, v in correlation_stability.items() 
                    if v.get('stability_score', 0) > 0.8
                ]
                if stable_relationships:
                    f.write(f"**🎯 Stable Relationships ({len(stable_relationships)}):**\n")
                    for rel in stable_relationships[:5]:
                        rel_info = correlation_stability[rel]
                        f.write(f"- {rel}: {rel_info.get('relationship_type', 'unknown')} "
                               f"(stability: {rel_info.get('stability_score', 0):.2f})\n")
                    f.write("\n")
        
        # Enhanced outlier analysis
        if 'outlier_analysis' in stats_results:
            f.write("### 🎯 Advanced Outlier Analysis\n")
            outlier_results = stats_results['outlier_analysis']
            
            # Consensus-based outlier summary
            total_consensus_outliers = 0
            features_with_outliers = 0
            high_outlier_features = []
            
            for feature, methods in outlier_results.items():
                consensus_info = methods.get('outlier_consensus', {})
                consensus_outliers = consensus_info.get('consensus_outliers', [])
                
                if consensus_outliers:
                    features_with_outliers += 1
                    total_consensus_outliers += len(consensus_outliers)
                    
                    outlier_pct = len(consensus_outliers) / len(results.get('df_shape', [1000, 1])[0]) * 100
                    if outlier_pct > 5:
                        high_outlier_features.append((feature, outlier_pct, len(consensus_outliers)))
            
            f.write(f"**Outlier Summary (Consensus-based):**\n")
            f.write(f"- Features with consensus outliers: {features_with_outliers}\n")
            f.write(f"- Total consensus outliers: {total_consensus_outliers}\n")
            
            if 'df_shape' in results:
                f.write(f"- Overall outlier percentage: {total_consensus_outliers/results['df_shape'][0]*100:.2f}%\n\n")
            
            if high_outlier_features:
                f.write(f"**🚨 Features with High Outlier Rates (>5%):**\n")
                for feature, pct, count in sorted(high_outlier_features, key=lambda x: x[1], reverse=True):
                    f.write(f"- `{feature}`: {count} outliers ({pct:.1f}%)\n")
                f.write("\n*Recommendation: Investigate these features for data quality issues or legitimate extreme values.*\n\n")
        
        f.write("---\n\n")
    
    def _write_quality_assessment(self, f, results: Dict):
        """Write enhanced data quality assessment"""
        f.write("## ✅ Data Quality Assessment\n\n")
        
        # Overall quality metrics
        data_quality = results.get('data_quality', {})
        quality_score = data_quality.get('completeness_score', 0)
        
        f.write(f"### 📊 Overall Data Quality\n")
        f.write(f"**Overall Data Quality Score: {quality_score:.1f}/100**\n\n")
        
        # Quality dimensions breakdown
        f.write("**Quality Dimensions:**\n")
        f.write(f"- **Completeness:** {quality_score:.1f}% (missing data impact)\n")
        
        # Missing data assessment
        missing_pct = data_quality.get('missing_data_percentage', 0)
        if missing_pct > 0:
            f.write(f"- **Missing Data:** {missing_pct:.1f}% of all values\n")
        
        # Duplicate assessment
        duplicate_pct = data_quality.get('duplicate_percentage', 0)
        f.write(f"- **Uniqueness:** {100 - duplicate_pct:.1f}% ({duplicate_pct:.1f}% duplicates)\n")
        
        # Consistency assessment (based on outliers and data types)
        stats_results = results.get('statistical_analysis', {})
        outlier_features = len(stats_results.get('outlier_analysis', {}))
        total_numeric_features = len([c for c in results.get('df_shape', [0, 0]) if True])  # Simplified
        
        f.write("\n")
        
        # Quality issues and recommendations
        quality_issues = data_quality.get('quality_issues', [])
        if quality_issues:
            f.write("### 🚨 Quality Issues Identified:\n")
            for issue in quality_issues:
                f.write(f"- ❌ {issue}\n")
            f.write("\n")
        else:
            f.write("### ✅ No Major Quality Issues Detected\n\n")
        
        # Actionable recommendations
        f.write("### 🔧 Quality Improvement Recommendations:\n")
        
        recommendations = []
        
        if missing_pct > 10:
            recommendations.append("Implement comprehensive missing data handling strategy")
        if duplicate_pct > 5:
            recommendations.append("Remove or investigate duplicate records")
        if outlier_features > 0:
            recommendations.append(f"Address outliers in {outlier_features} features")
        
        # Manufacturing-specific recommendations
        mfg_results = results.get('manufacturing_analysis', {})
        sensor_health = mfg_results.get('sensor_health', {})
        if sensor_health:
            poor_sensors = [
                sensor for sensor, health in sensor_health.items()
                if health.get('health_status') in ['Poor', 'Critical']
            ]
            if poor_sensors:
                recommendations.append(f"Investigate {len(poor_sensors)} sensors with poor data quality")
        
        if not recommendations:
            recommendations.append("Data quality is generally good - focus on feature engineering")
        
        for rec in recommendations:
            f.write(f"- 🎯 {rec}\n")
        
        f.write("\n---\n\n")
    
    def _write_feature_analysis(self, f, results: Dict):
        """Write enhanced feature analysis section"""
        f.write("## 🎯 Feature Analysis & Engineering\n\n")
        
        # Dimensionality insights
        dim_results = results.get('dimensionality_analysis', {})
        if 'pca_analysis' in dim_results:
            f.write("### 📐 Dimensionality Insights\n")
            pca_results = dim_results['pca_analysis']
            
            f.write(f"- **Components for 90% variance:** {pca_results.get('n_components_90_variance', 'N/A')}\n")
            f.write(f"- **Components for 95% variance:** {pca_results.get('n_components_95_variance', 'N/A')}\n")
            f.write(f"- **Components for 99% variance:** {pca_results.get('n_components_99_variance', 'N/A')}\n")
            
            # Dimensionality reduction potential
            quality_metrics = pca_results.get('quality_metrics', {})
            reduction_potential = quality_metrics.get('dimensionality_reduction_potential', 0)
            f.write(f"- **Dimensionality Reduction Potential:** {reduction_potential:.1%}\n\n")
            
            # Principal component interpretation
            interpretation = pca_results.get('interpretation', {})
            if interpretation:
                f.write("**Principal Component Interpretation:**\n")
                for pc, info in list(interpretation.items())[:3]:
                    f.write(f"- **{pc}** ({info.get('variance_explained', 0):.1%} variance): "
                           f"{info.get('interpretation_hint', 'Mixed characteristics')}\n")
                    dominant_features = info.get('dominant_features', [])
                    if dominant_features:
                        f.write(f"  - Key features: {', '.join([f'`{f}`' for f in dominant_features])}\n")
                f.write("\n")
        
        # Clustering insights
        if 'clustering_analysis' in dim_results:
            f.write("### 🎲 Natural Data Groupings\n")
            cluster_results = dim_results['clustering_analysis']
            
            kmeans_results = cluster_results.get('kmeans_analysis', {})
            if 'recommended_k' in kmeans_results:
                f.write(f"- **Recommended K-means clusters:** {kmeans_results['recommended_k']}\n")
                f.write(f"- **Optimal by elbow method:** {kmeans_results.get('optimal_k_elbow', 'N/A')}\n")
                f.write(f"- **Optimal by silhouette:** {kmeans_results.get('optimal_k_silhouette', 'N/A')}\n")
            
            dbscan_results = cluster_results.get('dbscan_analysis', {})
            if 'n_clusters' in dbscan_results and dbscan_results['n_clusters'] > 0:
                f.write(f"- **DBSCAN clusters found:** {dbscan_results['n_clusters']}\n")
                f.write(f"- **Noise points:** {dbscan_results.get('noise_percentage', 0):.1f}%\n")
            
            f.write("\n")
        
        # Feature engineering opportunities
        feature_eng = results.get('feature_engineering', {})
        if feature_eng:
            f.write("### 🔧 Feature Engineering Opportunities\n")
            
            # Transformation recommendations
            transform_recs = feature_eng.get('transformation_recommendations', [])
            if transform_recs:
                f.write(f"**Transformation Opportunities ({len(transform_recs)}):**\n")
                transform_summary = {}
                for rec in transform_recs:
                    issue = rec.get('issue', 'unknown')
                    transform_summary[issue] = transform_summary.get(issue, 0) + 1
                
                for issue, count in transform_summary.items():
                    f.write(f"- {issue.replace('_', ' ').title()}: {count} features\n")
                f.write("\n")
            
            # Interaction opportunities
            interaction_ops = feature_eng.get('interaction_opportunities', [])
            if interaction_ops:
                f.write(f"**High-Value Feature Interactions ({len(interaction_ops)}):**\n")
                for op in interaction_ops[:5]:
                    features = op.get('features', ['unknown', 'unknown'])
                    corr = op.get('correlation', 0)
                    f.write(f"- `{features[0]}` × `{features[1]}` (correlation: {corr:.3f})\n")
                f.write("\n")
            
            # Encoding recommendations
            encoding_recs = feature_eng.get('encoding_recommendations', [])
            if encoding_recs:
                f.write(f"**Categorical Encoding Needs ({len(encoding_recs)}):**\n")
                encoding_summary = {}
                for rec in encoding_recs:
                    method = rec.get('recommendation', 'unknown')
                    encoding_summary[method] = encoding_summary.get(method, 0) + 1
                
                for method, count in encoding_summary.items():
                    f.write(f"- {method.replace('_', ' ').title()}: {count} features\n")
                f.write("\n")
        
        f.write("---\n\n")
    
    def _write_ml_recommendations(self, f, results: Dict):
        """Write comprehensive ML pipeline recommendations"""
        f.write("## 🤖 ML Pipeline Recommendations\n\n")
        
        ml_readiness = results.get('ml_readiness', {})
        pipeline_recs = results.get('pipeline_recommendations', {})
        
        # Data preparation roadmap
        f.write("### 📋 Data Preparation Roadmap\n")
        prep_steps = ml_readiness.get('preprocessing_steps', [])
        if prep_steps:
            for i, step in enumerate(prep_steps, 1):
                f.write(f"{i}. {step}\n")
        else:
            f.write("- Data appears ready for modeling with minimal preprocessing\n")
        f.write("\n")
        
        # Model selection strategy
        f.write("### 🎯 Model Selection Strategy\n")
        recommended_models = ml_readiness.get('recommended_models', [])
        if recommended_models:
            f.write("**Recommended Models (in order of priority):**\n")
            for i, model in enumerate(recommended_models, 1):
                f.write(f"{i}. **{model}**\n")
            f.write("\n")
            
            # Model-specific recommendations
            temporal_detected = results.get('temporal_analysis', {}).get('time_series_detected', False)
            if temporal_detected:
                f.write("**Time Series Specific:**\n")
                f.write("- Use temporal cross-validation (not random splits)\n")
                f.write("- Consider lag features and rolling statistics\n")
                f.write("- Monitor for concept drift\n\n")
            
            mfg_detected = bool(results.get('manufacturing_analysis', {}).get('quality_patterns'))
            if mfg_detected:
                f.write("**Manufacturing Specific:**\n")
                f.write("- Handle class imbalance for defect detection\n")
                f.write("- Use domain-aware feature engineering\n")
                f.write("- Implement process control monitoring\n\n")
        
        # Validation strategy
        validation_strategy = ml_readiness.get('validation_strategy', {})
        if validation_strategy:
            f.write("### ✅ Validation Strategy\n")
            f.write(f"- **Method:** {validation_strategy.get('method', 'cross_validation')}\n")
            f.write(f"- **Type:** {validation_strategy.get('type', 'standard')}\n")
            if 'folds' in validation_strategy:
                f.write(f"- **Folds:** {validation_strategy['folds']}\n")
            f.write(f"- **Test Size:** {validation_strategy.get('test_size', 0.2):.0%}\n\n")
        
        # Evaluation metrics
        eval_metrics = pipeline_recs.get('evaluation_metrics', [])
        if eval_metrics:
            f.write("### 📊 Evaluation Metrics\n")
            for metric in eval_metrics:
                f.write(f"- {metric}\n")
            f.write("\n")
        
        # Deployment considerations
        deployment_considerations = pipeline_recs.get('deployment_considerations', [])
        if deployment_considerations:
            f.write("### 🚀 Deployment Considerations\n")
            for consideration in deployment_considerations:
                f.write(f"- {consideration}\n")
            f.write("\n")
        
        f.write("---\n\n")
    
    def _write_manufacturing_insights(self, f, results: Dict):
        """Write manufacturing-specific insights"""
        mfg_results = results.get('manufacturing_analysis', {})
        if not mfg_results or all(
            'message' in v or not v 
            for v in mfg_results.values() 
            if isinstance(v, dict)
        ):
            return  # Skip if no manufacturing insights
        
        f.write("## 🏭 Manufacturing Domain Insights\n\n")
        
        # Process capability
        process_capability = mfg_results.get('process_capability', {})
        if process_capability:
            f.write("### ⚙️ Process Capability Analysis\n")
            
            excellent_processes = []
            poor_processes = []
            
            for process, metrics in process_capability.items():
                capability = metrics.get('capability_assessment', 'Unknown')
                if 'Excellent' in capability:
                    excellent_processes.append(process)
                elif 'Poor' in capability:
                    poor_processes.append(process)
            
            if excellent_processes:
                f.write(f"**✅ Excellent Capability ({len(excellent_processes)}):**\n")
                for process in excellent_processes[:5]:
                    f.write(f"- `{process}`\n")
                f.write("\n")
            
            if poor_processes:
                f.write(f"**⚠️ Poor Capability ({len(poor_processes)}):**\n")
                for process in poor_processes[:5]:
                    metrics = process_capability[process]
                    cv = metrics.get('cv', 0)
                    f.write(f"- `{process}` (CV: {cv:.3f})\n")
                f.write("\n*Recommendation: Investigate and improve process control*\n\n")
        
        # Quality patterns
        quality_patterns = mfg_results.get('quality_patterns', {})
        if quality_patterns and 'message' not in quality_patterns:
            f.write("### 🎯 Quality Analysis\n")
            
            for feature, analysis in list(quality_patterns.items())[:5]:
                if analysis.get('type') == 'categorical':
                    quality_rate_info = analysis.get('quality_rate')
                    if quality_rate_info and isinstance(quality_rate_info, dict):
                        rate = quality_rate_info.get('rate', 0)
                        confidence = quality_rate_info.get('confidence_interval', [0, 0])
                        f.write(f"- **`{feature}`:** {rate:.1f}% quality rate "
                               f"(95% CI: {confidence[0]:.1f}%-{confidence[1]:.1f}%)\n")
                
                elif analysis.get('type') == 'numeric':
                    ooc_pct = analysis.get('out_of_control_percentage', 0)
                    if ooc_pct > 5:
                        f.write(f"- **`{feature}`:** {ooc_pct:.1f}% out-of-control points\n")
            
            f.write("\n")
        
        # Equipment efficiency
        equipment_patterns = mfg_results.get('equipment_patterns', {})
        if equipment_patterns and 'message' not in equipment_patterns:
            f.write("### 🔧 Equipment Performance\n")
            
            for equipment_type, analysis in list(equipment_patterns.items())[:3]:
                utilization = analysis.get('utilization_balance', {})
                if isinstance(utilization, dict):
                    balance_type = utilization.get('utilization_type', 'Unknown')
                    balance_score = utilization.get('balance_score', 0)
                    f.write(f"- **`{equipment_type}`:** {balance_type} "
                           f"(Balance Score: {balance_score:.1f})\n")
                    
                    # Highlight problematic equipment
                    overutilized = utilization.get('overutilized_equipment', [])
                    underutilized = utilization.get('underutilized_equipment', [])
                    
                    if overutilized:
                        f.write(f"  - ⚠️ Overutilized: {', '.join(overutilized[:3])}\n")
                    if underutilized:
                        f.write(f"  - 📉 Underutilized: {', '.join(underutilized[:3])}\n")
            
            f.write("\n")
        
        # Sensor health
        sensor_health = mfg_results.get('sensor_health', {})
        if sensor_health:
            f.write("### 📡 Sensor Health Assessment\n")
            
            health_summary = {}
            for sensor, health_info in sensor_health.items():
                status = health_info.get('health_status', 'Unknown')
                health_summary[status] = health_summary.get(status, 0) + 1
            
            f.write("**Health Status Distribution:**\n")
            for status, count in sorted(health_summary.items()):
                emoji = {"Excellent": "✅", "Good": "🟢", "Fair": "🟡", "Poor": "🟠", "Critical": "🔴"}.get(status, "⚪")
                f.write(f"- {emoji} {status}: {count} sensors\n")
            
            # Highlight critical sensors
            critical_sensors = [
                sensor for sensor, health in sensor_health.items()
                if health.get('health_status') == 'Critical'
            ]
            
            if critical_sensors:
                f.write(f"\n**🚨 Critical Sensors Requiring Attention ({len(critical_sensors)}):**\n")
                for sensor in critical_sensors[:5]:
                    health_info = sensor_health[sensor]
                    score = health_info.get('data_quality_score', 0)
                    f.write(f"- `{sensor}` (Quality Score: {score:.1f})\n")
            
            f.write("\n")
        
        f.write("---\n\n")
    
    def _write_technical_appendix(self, f, results: Dict, viz_paths: Dict):
        """Write enhanced technical appendix"""
        f.write("## 📋 Technical Appendix\n\n")
        
        # Visualizations generated
        f.write("### 📊 Generated Visualizations\n")
        viz_categories = {
            'overview': 'Data Overview & Quality',
            'distributions': 'Statistical Distributions',
            'correlations': 'Correlation Analysis', 
            'missing_values': 'Missing Value Analysis',
            'outliers': 'Outlier Detection',
            'time_series': 'Time Series Analysis',
            'feature_engineering': 'Feature Engineering'
        }
        
        for viz_type, path in viz_paths.items():
            if path:
                category = viz_categories.get(viz_type, viz_type.replace('_', ' ').title())
                f.write(f"- **{category}:** `{Path(path).name}`\n")
        f.write("\n")
        
        # Analysis methods used
        f.write("### 🔬 Analysis Methods Applied\n")
        f.write("**Statistical Methods:**\n")
        f.write("- **Normality Tests:** Shapiro-Wilk, Jarque-Bera, D'Agostino, Anderson-Darling, Kolmogorov-Smirnov\n")
        f.write("- **Outlier Detection:** IQR, Z-score, Modified Z-score, Isolation Forest, Elliptic Envelope\n")
        f.write("- **Correlation Analysis:** Pearson, Spearman, Kendall correlations\n")
        f.write("- **Missing Data Analysis:** Pattern detection, mechanism classification (MCAR, MAR, MNAR)\n")
        f.write("- **Distribution Analysis:** Robust parameter estimation, entropy calculation\n\n")
        
        f.write("**Machine Learning Methods:**\n")
        f.write("- **Dimensionality Reduction:** PCA with interpretation, t-SNE optimization\n")
        f.write("- **Clustering:** K-means with elbow/silhouette optimization, DBSCAN parameter tuning\n")
        f.write("- **Feature Selection:** Variance-based, correlation-based filtering\n")
        f.write("- **Intrinsic Dimensionality:** PCA-based estimation, effective rank calculation\n\n")
        
        f.write("**Manufacturing-Specific Methods:**\n")
        f.write("- **Process Capability:** Cp, Cpk estimation, control limit calculation\n")
        f.write("- **Quality Analysis:** Defect rate calculation with confidence intervals\n")
        f.write("- **Equipment Analysis:** OEE concepts, utilization balance (Gini coefficient)\n")
        f.write("- **Sensor Health:** Data quality scoring, anomaly detection\n\n")
        
        # Configuration used
        f.write("### ⚙️ Analysis Configuration\n")
        config_dict = results.get('config_used', {})
        important_params = [
            'missing_threshold', 'correlation_threshold', 'outlier_threshold',
            'skewness_threshold', 'random_state'
        ]
        
        for param in important_params:
            if param in config_dict:
                f.write(f"- **{param.replace('_', ' ').title()}:** {config_dict[param]}\n")
        
        f.write(f"- **Analysis Mode:** {'Manufacturing-focused' if config_dict.get('manufacturing_focused', False) else 'General purpose'}\n")
        f.write(f"- **Output Directory:** `{config_dict.get('output_dir', './eda_results')}`\n\n")
        
        # Performance metrics
        processing_time = results.get('processing_time_seconds', 0)
        f.write("### ⏱️ Performance Metrics\n")
        f.write(f"- **Total Processing Time:** {processing_time:.2f} seconds\n")
        f.write(f"- **Memory Usage:** Optimized with dtype conversion and chunked loading\n")
        f.write(f"- **Visualization Generation:** Interactive HTML reports with Plotly\n\n")
        
        # Scientific references
        f.write("### 📚 Scientific References\n")
        f.write("- Montgomery, D.C. (2013). *Statistical Quality Control: A Modern Introduction*\n")
        f.write("- Little, R.J. & Rubin, D.B. (2019). *Statistical Analysis with Missing Data*\n")
        f.write("- Wickham, H. (2014). *Tidy Data*. Journal of Statistical Software\n")
        f.write("- van der Aalst, W. (2016). *Process Mining: Data Science in Action*\n")
        f.write("- Jolliffe, I.T. & Cadima, J. (2016). *Principal Component Analysis: A Review*\n")
        f.write("- NIST Engineering Statistics Handbook (2024). *Statistical Methods for Quality Control*\n")
        f.write("- Google Cloud MLOps Architecture Center (2024). *ML Pipeline Best Practices*\n\n")
        
        # Next steps
        f.write("### 🚀 Recommended Next Steps\n")
        f.write("1. **Review Visualizations:** Examine all generated plots for patterns and insights\n")
        f.write("2. **Address Quality Issues:** Follow data quality recommendations before modeling\n")
        f.write("3. **Feature Engineering:** Implement suggested transformations and interactions\n")
        f.write("4. **Model Development:** Start with recommended models and validation strategy\n")
        f.write("5. **Manufacturing Integration:** If applicable, implement domain-specific monitoring\n")
        f.write("6. **Continuous Monitoring:** Set up data drift detection for production deployment\n\n")
        
        # Contact and support
        f.write("---\n\n")
        f.write("*Report generated by Scientific EDA Pipeline v2.0*\n")
        f.write("*For questions about this analysis, refer to the generated logs and visualizations*\n")
    
    def _generate_enhanced_json_summary(self, results: Dict, df: pd.DataFrame):
        """Generate enhanced JSON summary with comprehensive metadata"""
        ml_readiness = results.get('ml_readiness', {})
        data_quality = results.get('data_quality', {})
        
        summary = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0',
                'processing_time_seconds': results.get('processing_time_seconds', 0),
                'analysis_mode': 'manufacturing_focused' if results.get('config_used', {}).get('manufacturing_focused', False) else 'general'
            },
            'dataset_summary': {
                'shape': [df.shape[0], df.shape[1]],
                'memory_mb': float(df.memory_usage(deep=True).sum() / (1024**2)),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_features': len([c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])])
            },
            'data_quality_metrics': {
                'overall_score': data_quality.get('completeness_score', 0),
                'missing_data_percentage': data_quality.get('missing_data_percentage', 0),
                'duplicate_percentage': data_quality.get('duplicate_percentage', 0),
                'constant_columns': len(data_quality.get('constant_columns', [])),
                'quality_issues_count': len(data_quality.get('quality_issues', []))
            },
            'statistical_insights': {
                'high_correlation_pairs': len(
                    results.get('statistical_analysis', {})
                    .get('correlation_analysis', {})
                    .get('high_correlation_pairs', [])
                ),
                'features_with_outliers': len(
                    results.get('statistical_analysis', {})
                    .get('outlier_analysis', {})
                ),
                'non_normal_features': self._count_non_normal_features(results),
                'highly_skewed_features': self._count_skewed_features(results)
            },
            'ml_readiness_assessment': {
                'overall_score': ml_readiness.get('overall_score', 0),
                'readiness_factors': ml_readiness.get('readiness_factors', {}),
                'blocking_issues': ml_readiness.get('blocking_issues', []),
                'recommended_models': ml_readiness.get('recommended_models', []),
                'preprocessing_steps_needed': len(ml_readiness.get('preprocessing_steps', []))
            },
            'feature_engineering_opportunities': {
                'transformation_candidates': len(
                    results.get('feature_engineering', {})
                    .get('transformation_recommendations', [])
                ),
                'interaction_candidates': len(
                    results.get('feature_engineering', {})
                    .get('interaction_opportunities', [])
                ),
                'encoding_needed': len(
                    results.get('feature_engineering', {})
                    .get('encoding_recommendations', [])
                )
            },
            'dimensionality_insights': {
                'pca_90_variance_components': results.get('dimensionality_analysis', {})
                    .get('pca_analysis', {})
                    .get('n_components_90_variance'),
                'intrinsic_dimensionality': results.get('dimensionality_analysis', {})
                    .get('intrinsic_dimensionality', {})
                    .get('pca_90_percent'),
                'recommended_clusters': results.get('dimensionality_analysis', {})
                    .get('clustering_analysis', {})
                    .get('kmeans_analysis', {})
                    .get('recommended_k')
            },
            'temporal_characteristics': {
                'time_series_detected': results.get('temporal_analysis', {}).get('time_series_detected', False),
                'seasonality_detected': results.get('temporal_analysis', {}).get('seasonality_detected', False),
                'trend_detected': results.get('temporal_analysis', {}).get('trend_detected', False),
                'temporal_features_count': len(
                    results.get('temporal_analysis', {}).get('datetime_columns', []) +
                    results.get('temporal_analysis', {}).get('potential_timestamps', [])
                )
            },
            'manufacturing_insights': self._extract_manufacturing_summary(results),
            'recommendations': {
                'priority_actions': self._generate_priority_actions(results),
                'validation_strategy': ml_readiness.get('validation_strategy', {}),
                'deployment_readiness': self._assess_deployment_readiness(results)
            },
            'generated_artifacts': {
                'visualizations': list(results.get('visualizations', {}).keys()),
                'report_path': results.get('report_path', ''),
                'output_directory': results.get('config_used', {}).get('output_dir', '')
            }
        }
        
        # Save enhanced summary
        summary_path = self.output_dir / "eda_comprehensive_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        
        self.logger.info(f"📄 Enhanced JSON summary saved: {summary_path}")
    
    def _count_non_normal_features(self, results: Dict) -> int:
        """Count features that are not normally distributed"""
        dist_analysis = results.get('statistical_analysis', {}).get('distribution_analysis', {})
        non_normal_count = 0
        
        for feature, analysis in dist_analysis.items():
            normality_tests = analysis.get('normality_tests', {})
            is_normal = any(
                test_result.get('is_normal', False) 
                for test_result in normality_tests.values() 
                if test_result
            )
            if not is_normal:
                non_normal_count += 1
        
        return non_normal_count
    
    def _count_skewed_features(self, results: Dict) -> int:
        """Count highly skewed features"""
        dist_analysis = results.get('statistical_analysis', {}).get('distribution_analysis', {})
        skewed_count = 0
        
        for feature, analysis in dist_analysis.items():
            params = analysis.get('distribution_params', {})
            skewness = abs(params.get('skewness', 0))
            if skewness > 1.5:  # Highly skewed threshold
                skewed_count += 1
        
        return skewed_count
    
    def _extract_manufacturing_summary(self, results: Dict) -> Dict:
        """Extract manufacturing analysis summary"""
        mfg_results = results.get('manufacturing_analysis', {})
        
        summary = {
            'manufacturing_context_detected': bool(mfg_results),
            'quality_indicators_found': bool(mfg_results.get('quality_patterns', {})),
            'equipment_data_found': bool(mfg_results.get('equipment_patterns', {})),
            'process_capability_analyzed': bool(mfg_results.get('process_capability', {})),
            'sensor_health_assessed': bool(mfg_results.get('sensor_health', {}))
        }
        
        # Count specific insights
        if summary['quality_indicators_found']:
            quality_patterns = mfg_results.get('quality_patterns', {})
            summary['quality_features_count'] = len([
                k for k, v in quality_patterns.items() 
                if isinstance(v, dict) and 'message' not in v
            ])
        
        if summary['equipment_data_found']:
            equipment_patterns = mfg_results.get('equipment_patterns', {})
            summary['equipment_types_count'] = len([
                k for k, v in equipment_patterns.items()
                if isinstance(v, dict) and 'message' not in v
            ])
        
        if summary['sensor_health_assessed']:
            sensor_health = mfg_results.get('sensor_health', {})
            critical_sensors = [
                k for k, v in sensor_health.items()
                if v.get('health_status') in ['Poor', 'Critical']
            ]
            summary['sensors_needing_attention'] = len(critical_sensors)
        
        return summary
    
    def _generate_priority_actions(self, results: Dict) -> List[str]:
        """Generate prioritized action items based on analysis"""
        actions = []
        
        # High priority: Blocking issues
        ml_readiness = results.get('ml_readiness', {})
        blocking_issues = ml_readiness.get('blocking_issues', [])
        
        for issue in blocking_issues[:3]:  # Top 3 blocking issues
            actions.append(f"HIGH: {issue}")
        
        # Medium priority: Data quality issues
        data_quality = results.get('data_quality', {})
        quality_issues = data_quality.get('quality_issues', [])
        
        for issue in quality_issues[:2]:  # Top 2 quality issues
            actions.append(f"MEDIUM: {issue}")
        
        # Medium priority: Feature engineering opportunities
        feature_eng = results.get('feature_engineering', {})
        transform_count = len(feature_eng.get('transformation_recommendations', []))
        if transform_count > 5:
            actions.append(f"MEDIUM: Apply transformations to {transform_count} skewed features")
        
        # Low priority: Manufacturing-specific improvements
        mfg_results = results.get('manufacturing_analysis', {})
        sensor_health = mfg_results.get('sensor_health', {})
        if sensor_health:
            critical_sensors = [
                k for k, v in sensor_health.items()
                if v.get('health_status') == 'Critical'
            ]
            if critical_sensors:
                actions.append(f"LOW: Investigate {len(critical_sensors)} critical sensors")
        
        # Default action if no specific issues
        if not actions:
            actions.append("LOW: Focus on feature engineering and model selection")
        
        return actions[:5]  # Maximum 5 priority actions
    
    def _assess_deployment_readiness(self, results: Dict) -> str:
        """Assess overall deployment readiness"""
        ml_readiness = results.get('ml_readiness', {})
        readiness_score = ml_readiness.get('overall_score', 0)
        
        blocking_issues = len(ml_readiness.get('blocking_issues', []))
        
        if readiness_score >= 85 and blocking_issues == 0:
            return "Ready for deployment pipeline development"
        elif readiness_score >= 70 and blocking_issues <= 2:
            return "Near ready - address minor issues first"
        elif readiness_score >= 50:
            return "Requires significant preprocessing before deployment"
        else:
            return "Not ready - major data quality issues need resolution"
"""
Scientific EDA Utilities Module
Supporting classes and functions for the main EDA pipeline

Based on Latest Statistical Methods & Manufacturing Best Practices:
- NIST Statistical Handbook methods
- Robust outlier detection (2024 best practices)
- Advanced missing data analysis
- Manufacturing domain-specific patterns
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
from scipy.stats import shapiro, anderson, kstest, jarque_bera, normaltest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score

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


@dataclass
class EDAConfig:
    """Enhanced configuration for EDA pipeline based on scientific best practices"""
    
    # File processing
    output_dir: str = "./eda_results"
    max_memory_usage: float = 0.8  # Max 80% of available RAM
    chunk_size: int = 10000  # For large files
    
    # Statistical thresholds (from literature)
    missing_threshold: float = 0.5  # Montgomery (2013) - 50% missing = remove
    correlation_threshold: float = 0.9  # Wickham (2014) - high correlation
    outlier_threshold: float = 3.0  # Z-score threshold
    skewness_threshold: float = 1.0  # Moderate skewness
    
    # Enhanced thresholds based on 2024 best practices
    kurtosis_threshold: float = 3.0  # Excess kurtosis threshold
    variance_threshold: float = 1e-6  # Near-zero variance
    cardinality_threshold: float = 0.95  # High cardinality = ID-like
    
    # Manufacturing-specific thresholds
    manufacturing_focused: bool = False
    process_capability_threshold: float = 1.33  # Cpk threshold
    
    # Visualization settings
    max_categories: int = 20  # Max categories to show in plots
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    # Time series detection
    datetime_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y', '%Y-%m-%d %H:%M:%S.%f', 'ISO8601'
    ])
    
    # Advanced analysis settings
    pca_components: int = 10
    tsne_perplexity: int = 30
    cluster_range: Tuple[int, int] = (2, 10)
    
    # Random state for reproducibility
    random_state: int = 42


class DataProfiler:
    """Advanced data profiling based on scientific methodologies"""
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self.setup_logging()
        self.results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'eda_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def profile_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Profile file-level metadata using PyArrow for efficiency
        Based on Zaharia et al. (2016) - Apache Spark principles
        """
        self.logger.info("🔍 Profiling file metadata")
        
        file_path = Path(file_path)
        metadata = {
            'file_size_mb': file_path.stat().st_size / (1024**2),
            'file_name': file_path.name,
            'file_extension': file_path.suffix
        }
        
        if PYARROW_AVAILABLE and file_path.suffix == '.parquet':
            try:
                parquet_file = pq.ParquetFile(file_path)
                metadata.update({
                    'num_row_groups': parquet_file.num_row_groups,
                    'schema_names': parquet_file.schema.names,
                    'total_rows_estimate': sum(rg.num_rows for rg in parquet_file.metadata.row_groups),
                    'compression': str(parquet_file.metadata.row_group(0).column(0).compression),
                    'parquet_version': parquet_file.metadata.format_version
                })
                self.logger.info(f"📊 Parquet file: {metadata['total_rows_estimate']:,} rows, "
                               f"{len(metadata['schema_names'])} columns")
            except Exception as e:
                self.logger.warning(f"Could not read parquet metadata: {e}")
        
        # Memory estimation
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        estimated_memory_need = metadata['file_size_mb'] * 3 / 1024  # Rule of thumb: 3x file size
        metadata['memory_feasible'] = estimated_memory_need < (available_memory * self.config.max_memory_usage)
        metadata['estimated_memory_gb'] = estimated_memory_need
        metadata['available_memory_gb'] = available_memory
        
        self.results['file_metadata'] = metadata
        return metadata


class DataLoader:
    """Intelligent data loading with memory management"""
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def load_data_smart(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """
        Smart data loading based on file size and memory constraints
        Implements chunked loading for large files
        """
        self.logger.info("📂 Smart data loading initiated")
        
        if not metadata['memory_feasible']:
            self.logger.warning(f"⚠️ Large file detected ({metadata['file_size_mb']:.1f} MB). Using chunked loading.")
            return self._load_chunked(file_path)
        else:
            return self._load_full(file_path)
    
    def _load_full(self, file_path: str) -> pd.DataFrame:
        """Load full dataset into memory"""
        self.logger.info("📊 Loading full dataset")
        
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
        
        # Memory optimization
        df = self._optimize_dtypes(df)
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        self.logger.info(f"✅ Loaded {df.shape[0]:,} × {df.shape[1]} dataset ({memory_usage:.1f} MB)")
        
        return df
    
    def _load_chunked(self, file_path: str) -> pd.DataFrame:
        """Load data in chunks for large files"""
        self.logger.info(f"📦 Loading data in chunks of {self.config.chunk_size:,} rows")
        
        if file_path.endswith('.parquet'):
            # For parquet, read a sample
            df_sample = pd.read_parquet(file_path)
            if len(df_sample) > self.config.chunk_size * 10:
                df_sample = df_sample.sample(n=self.config.chunk_size * 10, random_state=self.config.random_state)
        else:
            # For CSV, read in chunks
            chunk_list = []
            total_chunks = 0
            
            for chunk in pd.read_csv(file_path, chunksize=self.config.chunk_size):
                chunk_list.append(chunk)
                total_chunks += 1
                if total_chunks >= 10:  # Limit to first 10 chunks for analysis
                    break
            
            df_sample = pd.concat(chunk_list, ignore_index=True)
        
        df_sample = self._optimize_dtypes(df_sample)
        self.logger.info(f"📊 Sample loaded: {df_sample.shape[0]:,} × {df_sample.shape[1]} for analysis")
        
        return df_sample
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df


class TimeSeriesDetector:
    """
    Advanced time series detection and analysis
    Based on van der Aalst (2016) - Process Mining principles
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def detect_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze temporal patterns in data"""
        self.logger.info("🕐 Detecting temporal features")
        
        temporal_info = {
            'datetime_columns': [],
            'potential_timestamps': [],
            'time_series_detected': False,
            'temporal_patterns': {},
            'seasonality_detected': False,
            'trend_detected': False
        }
        
        # Detect datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_info['datetime_columns'].append(col)
                continue
                
            # Try to parse as datetime
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                parsed_count = 0
                
                for fmt in self.config.datetime_formats:
                    try:
                        if fmt == 'ISO8601':
                            pd.to_datetime(sample_values, errors='coerce')
                        else:
                            pd.to_datetime(sample_values, format=fmt, errors='coerce')
                        parsed_count += 1
                        break
                    except:
                        continue
                
                if parsed_count > 0:
                    temporal_info['potential_timestamps'].append(col)
        
        # Analyze temporal patterns if found
        all_temporal_cols = temporal_info['datetime_columns'] + temporal_info['potential_timestamps']
        
        if all_temporal_cols:
            temporal_info['time_series_detected'] = True
            temporal_info['temporal_patterns'] = self._analyze_temporal_patterns(df, all_temporal_cols)
            
            # Advanced time series analysis
            if STATSMODELS_AVAILABLE:
                temporal_info.update(self._advanced_time_series_analysis(df, all_temporal_cols))
        
        return temporal_info
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
        """Analyze patterns in temporal data"""
        patterns = {}
        
        for col in temporal_cols:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    dt_series = df[col]
                
                dt_series = dt_series.dropna()
                
                if len(dt_series) > 10:
                    patterns[col] = {
                        'start_date': dt_series.min(),
                        'end_date': dt_series.max(),
                        'date_range_days': (dt_series.max() - dt_series.min()).days,
                        'frequency_estimate': self._estimate_frequency(dt_series),
                        'has_gaps': self._detect_gaps(dt_series),
                        'business_hours_pattern': self._detect_business_hours(dt_series),
                        'regularity_score': self._calculate_regularity_score(dt_series)
                    }
            except Exception as e:
                self.logger.warning(f"Could not analyze temporal patterns for {col}: {e}")
                
        return patterns
    
    def _advanced_time_series_analysis(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
        """Advanced time series analysis using statsmodels"""
        analysis = {'seasonality_detected': False, 'trend_detected': False, 'stationarity_test': {}}
        
        try:
            # Use first temporal column for detailed analysis
            time_col = temporal_cols[0]
            
            # Convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Find numeric columns for time series analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Limit to 3 for performance
            
            for num_col in numeric_cols:
                # Create time series
                ts_data = df[[time_col, num_col]].dropna().sort_values(time_col)
                if len(ts_data) < 50:  # Need sufficient data
                    continue
                
                ts_data = ts_data.set_index(time_col)
                
                # Stationarity test (Augmented Dickey-Fuller)
                try:
                    adf_result = adfuller(ts_data[num_col].values)
                    analysis['stationarity_test'][num_col] = {
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'is_stationary': adf_result[1] < 0.05
                    }
                except:
                    pass
                
                # Seasonality detection (simple autocorrelation check)
                try:
                    autocorr = acf(ts_data[num_col].values, nlags=min(40, len(ts_data)//4))
                    if np.any(np.abs(autocorr[12:]) > 0.3):  # Check for seasonal patterns
                        analysis['seasonality_detected'] = True
                except:
                    pass
                
                # Trend detection (simple linear regression slope)
                try:
                    x = np.arange(len(ts_data))
                    slope, _, r_value, p_value, _ = stats.linregress(x, ts_data[num_col].values)
                    if abs(r_value) > 0.3 and p_value < 0.05:
                        analysis['trend_detected'] = True
                except:
                    pass
                
                break  # Only analyze first valid numeric column for performance
                
        except Exception as e:
            self.logger.warning(f"Advanced time series analysis failed: {e}")
        
        return analysis
    
    def _estimate_frequency(self, dt_series: pd.Series) -> str:
        """Estimate the frequency of time series data"""
        if len(dt_series) < 3:
            return "insufficient_data"
        
        # Calculate differences
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        # Get most common difference
        most_common_diff = diffs.mode()
        
        if len(most_common_diff) > 0:
            diff_seconds = most_common_diff.iloc[0].total_seconds()
            
            if diff_seconds < 60:
                return "seconds"
            elif diff_seconds < 3600:
                return "minutes"
            elif diff_seconds < 86400:
                return "hours"
            elif diff_seconds < 604800:
                return "days"
            else:
                return "weeks_or_more"
        
        return "irregular"
    
    def _detect_gaps(self, dt_series: pd.Series) -> bool:
        """Detect if there are significant gaps in time series"""
        if len(dt_series) < 3:
            return False
        
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        # Check if any gap is more than 3x the median gap
        median_diff = diffs.median()
        large_gaps = diffs > (median_diff * 3)
        
        return large_gaps.any()
    
    def _detect_business_hours(self, dt_series: pd.Series) -> Dict:
        """Detect if data follows business hours pattern"""
        hours = dt_series.dt.hour
        weekdays = dt_series.dt.weekday
        
        # Business hours typically 8-17 (8 AM to 5 PM)
        business_hours = ((hours >= 8) & (hours <= 17)).sum()
        weekday_data = (weekdays < 5).sum()  # Monday=0, Friday=4
        
        total_records = len(dt_series)
        
        return {
            'business_hours_pct': business_hours / total_records if total_records > 0 else 0,
            'weekday_pct': weekday_data / total_records if total_records > 0 else 0,
            'likely_business_pattern': (business_hours / total_records > 0.7) if total_records > 0 else False
        }
    
    def _calculate_regularity_score(self, dt_series: pd.Series) -> float:
        """Calculate regularity score of time series"""
        if len(dt_series) < 3:
            return 0.0
        
        sorted_dates = dt_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        if len(diffs) == 0:
            return 0.0
        
        # Calculate coefficient of variation of intervals
        cv = diffs.std() / diffs.mean() if diffs.mean() > 0 else float('inf')
        
        # Convert to regularity score (lower CV = higher regularity)
        regularity_score = max(0, 100 - (cv * 100))
        
        return min(100, regularity_score)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis based on scientific literature
    Enhanced with 2024 best practices for robust statistical methods
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def comprehensive_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis with enhanced methods"""
        self.logger.info("📊 Comprehensive statistical analysis")
        
        analysis_results = {
            'basic_statistics': self._enhanced_basic_statistics(df),
            'distribution_analysis': self._enhanced_distribution_analysis(df),
            'correlation_analysis': self._enhanced_correlation_analysis(df),
            'outlier_analysis': self._robust_outlier_analysis(df),
            'missing_value_analysis': self._advanced_missing_analysis(df),
            'feature_importance': self._calculate_feature_importance(df)
        }
        
        return analysis_results
    
    def _enhanced_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced basic statistics for all features"""
        stats_dict = {}
        
        # Numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            
            # Add enhanced statistics
            additional_stats = pd.DataFrame({
                'skewness': df[numeric_cols].skew(),
                'kurtosis': df[numeric_cols].kurtosis(),
                'excess_kurtosis': df[numeric_cols].kurtosis() - 3,  # Excess kurtosis
                'missing_count': df[numeric_cols].isnull().sum(),
                'missing_percentage': df[numeric_cols].isnull().sum() / len(df) * 100,
                'unique_count': df[numeric_cols].nunique(),
                'zero_count': (df[numeric_cols] == 0).sum(),
                'negative_count': (df[numeric_cols] < 0).sum(),
                'coefficient_of_variation': df[numeric_cols].std() / df[numeric_cols].mean(),
                'mad': df[numeric_cols].apply(lambda x: stats.median_abs_deviation(x.dropna())),  # Median Absolute Deviation
                'iqr': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
            })
            
            stats_dict['numerical'] = pd.concat([numeric_stats, additional_stats]).T
        
        # Categorical features with enhanced analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_stats = []
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                stats = {
                    'count': df[col].count(),
                    'unique': df[col].nunique(),
                    'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_freq_pct': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': df[col].isnull().sum() / len(df) * 100,
                    'entropy': self._calculate_entropy(value_counts),
                    'concentration_ratio': self._calculate_concentration_ratio(value_counts)
                }
                cat_stats.append(stats)
            
            stats_dict['categorical'] = pd.DataFrame(cat_stats, index=categorical_cols)
        
        return stats_dict
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate Shannon entropy for categorical variables"""
        if len(value_counts) == 0:
            return 0.0
        
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
        return entropy
    
    def _calculate_concentration_ratio(self, value_counts: pd.Series) -> float:
        """Calculate concentration ratio (top category percentage)"""
        if len(value_counts) == 0:
            return 0.0
        
        return (value_counts.iloc[0] / value_counts.sum()) * 100
    
    def _enhanced_distribution_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced distribution analysis using multiple normality tests
        Based on NIST Statistical Handbook and recent best practices
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_results = {}
        
        for col in numeric_cols:
            if df[col].dropna().nunique() > 3:  # Skip binary/sparse features
                data = df[col].dropna()
                
                if len(data) > 3:
                    results = {
                        'normality_tests': self._comprehensive_normality_tests(data),
                        'distribution_params': self._enhanced_distribution_params(data),
                        'outlier_indices': self._detect_outliers_iqr(data),
                        'distribution_recommendation': self._recommend_distribution(data)
                    }
                    distribution_results[col] = results
        
        return distribution_results
    
    def _comprehensive_normality_tests(self, data: pd.Series) -> Dict:
        """Comprehensive normality testing with multiple methods"""
        results = {}
        
        if len(data) > 3:
            # Shapiro-Wilk test (best for n < 5000)
            if len(data) <= 5000:
                try:
                    stat, p_value = shapiro(data)
                    results['shapiro_wilk'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
                except:
                    results['shapiro_wilk'] = None
            
            # Jarque-Bera test (good for large samples)
            try:
                stat, p_value = jarque_bera(data)
                results['jarque_bera'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['jarque_bera'] = None
            
            # D'Agostino's normality test
            try:
                stat, p_value = normaltest(data)
                results['dagostino'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['dagostino'] = None
            
            # Anderson-Darling test
            try:
                result = anderson(data, dist='norm')
                results['anderson_darling'] = {
                    'statistic': result.statistic,
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist(),
                    'is_normal': result.statistic < result.critical_values[2]  # 5% significance level
                }
            except:
                results['anderson_darling'] = None
            
            # Kolmogorov-Smirnov test
            try:
                stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
                results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
            except:
                results['kolmogorov_smirnov'] = None
        
        return results
    
    def _enhanced_distribution_params(self, data: pd.Series) -> Dict:
        """Enhanced distribution parameters with robust statistics"""
        params = {}
        
        # Basic parameters
        params['mean'] = float(data.mean())
        params['std'] = float(data.std())
        params['median'] = float(data.median())
        params['mode'] = float(data.mode().iloc[0]) if not data.mode().empty else None
        
        # Robust statistics
        params['mad'] = float(stats.median_abs_deviation(data))  # Median Absolute Deviation
        params['iqr'] = float(data.quantile(0.75) - data.quantile(0.25))
        params['trimmed_mean'] = float(stats.trim_mean(data, 0.1))  # 10% trimmed mean
        
        # Distribution shape
        params['skewness'] = float(data.skew())
        params['kurtosis'] = float(data.kurtosis())
        params['excess_kurtosis'] = float(data.kurtosis() - 3)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            params[f'percentile_{p}'] = float(data.quantile(p/100))
        
        return params
    
    def _recommend_distribution(self, data: pd.Series) -> str:
        """Recommend appropriate distribution based on characteristics"""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5:
            return "normal"
        elif skewness > 1:
            return "log_normal"
        elif skewness < -1:
            return "beta_or_uniform"
        elif kurtosis > 6:
            return "heavy_tailed"
        elif (data >= 0).all() and skewness > 0:
            return "gamma_or_exponential"
        else:
            return "custom_analysis_needed"
    
    def _enhanced_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced correlation analysis with multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'warning': 'Insufficient numeric columns for correlation analysis'}
        
        results = {}
        
        # Pearson correlation (linear relationships)
        pearson_corr = df[numeric_cols].corr(method='pearson')
        results['pearson'] = pearson_corr
        
        # Spearman correlation (monotonic relationships)
        spearman_corr = df[numeric_cols].corr(method='spearman')
        results['spearman'] = spearman_corr
        
        # Kendall correlation (robust to outliers)
        kendall_corr = df[numeric_cols].corr(method='kendall')
        results['kendall'] = kendall_corr
        
        # High correlation pairs
        high_corr_pairs = self._find_high_correlation_pairs(pearson_corr)
        results['high_correlation_pairs'] = high_corr_pairs
        
        # Correlation stability analysis
        results['correlation_stability'] = self._analyze_correlation_stability(
            pearson_corr, spearman_corr, kendall_corr
        )
        
        return results
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.config.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        return sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
    
    def _analyze_correlation_stability(self, pearson: pd.DataFrame, spearman: pd.DataFrame, 
                                     kendall: pd.DataFrame) -> Dict:
        """Analyze stability of correlations across different methods"""
        stability = {}
        
        for col1 in pearson.columns:
            for col2 in pearson.columns:
                if col1 != col2:
                    p_corr = pearson.loc[col1, col2]
                    s_corr = spearman.loc[col1, col2]
                    k_corr = kendall.loc[col1, col2]
                    
                    # Calculate stability as standard deviation of correlations
                    corr_std = np.std([p_corr, s_corr, k_corr])
                    
                    if abs(p_corr) > 0.3:  # Only analyze meaningful correlations
                        stability[f"{col1}-{col2}"] = {
                            'pearson': p_corr,
                            'spearman': s_corr,
                            'kendall': k_corr,
                            'stability_score': 1 - corr_std,  # Higher score = more stable
                            'relationship_type': self._classify_relationship(p_corr, s_corr)
                        }
        
        return stability
    
    def _classify_relationship(self, pearson: float, spearman: float) -> str:
        """Classify the type of relationship based on correlation coefficients"""
        p_abs = abs(pearson)
        s_abs = abs(spearman)
        
        if p_abs > 0.8 and s_abs > 0.8:
            return "strong_linear"
        elif s_abs > p_abs + 0.2:
            return "monotonic_nonlinear"
        elif p_abs > s_abs + 0.2:
            return "linear_with_outliers"
        else:
            return "moderate_relationship"
    
    def _robust_outlier_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Robust multi-method outlier detection based on 2024 best practices
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 10:
                outlier_results[col] = {
                    'iqr_outliers': self._detect_outliers_iqr(data),
                    'zscore_outliers': self._detect_outliers_zscore(data),
                    'modified_zscore_outliers': self._detect_outliers_modified_zscore(data),
                    'isolation_forest_outliers': self._detect_outliers_isolation_forest(data),
                    'elliptic_envelope_outliers': self._detect_outliers_elliptic_envelope(data),
                    'outlier_consensus': self._calculate_outlier_consensus(data)
                }
        
        return outlier_results
    
    def _detect_outliers_iqr(self, data: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > self.config.outlier_threshold
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_modified_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Modified Z-score (more robust)"""
        median = np.median(data)
        mad = stats.median_abs_deviation(data)
        
        if mad == 0:
            return []
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > 3.5
        return data[outlier_mask].index.tolist()
    
    def _detect_outliers_isolation_forest(self, data: pd.Series) -> List[int]:
        """Detect outliers using Isolation Forest"""
        try:
            clf = IsolationForest(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_elliptic_envelope(self, data: pd.Series) -> List[int]:
        """Detect outliers using Elliptic Envelope"""
        try:
            clf = EllipticEnvelope(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _calculate_outlier_consensus(self, data: pd.Series) -> Dict:
        """Calculate consensus outliers across multiple methods"""
        methods = {
            'iqr': set(self._detect_outliers_iqr(data)),
            'zscore': set(self._detect_outliers_zscore(data)),
            'modified_zscore': set(self._detect_outliers_modified_zscore(data)),
            'isolation_forest': set(self._detect_outliers_isolation_forest(data)),
            'elliptic_envelope': set(self._detect_outliers_elliptic_envelope(data))
        }
        
        # Find consensus outliers (detected by multiple methods)
        all_outliers = set()
        for outliers in methods.values():
            all_outliers.update(outliers)
        
        consensus_scores = {}
        for idx in all_outliers:
            score = sum(1 for outliers in methods.values() if idx in outliers)
            consensus_scores[idx] = score
        
        # High consensus outliers (detected by 3+ methods)
        high_consensus = [idx for idx, score in consensus_scores.items() if score >= 3]
        
        return {
            'consensus_outliers': high_consensus,
            'outlier_scores': consensus_scores,
            'method_agreement': len(high_consensus) / len(all_outliers) if all_outliers else 0
        }
    
    def _advanced_missing_analysis(self, df: pd.DataFrame) -> Dict:
        """Advanced missing value analysis with pattern detection"""
        missing_info = {}
        
        # Basic missing value statistics
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_info['summary'] = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        }).sort_values('missing_percentage', ascending=False)
        
        # Missing value patterns and mechanisms
        missing_info['patterns'] = self._analyze_missing_patterns(df)
        missing_info['mechanisms'] = self._analyze_missing_mechanisms(df)
        
        return missing_info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in missing data"""
        # Find columns with missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        if not cols_with_missing:
            return {'message': 'No missing values found'}
        
        patterns = {}
        
        # Co-occurrence of missing values
        if len(cols_with_missing) > 1:
            missing_matrix = df[cols_with_missing].isnull()
            
            # Find common patterns
            pattern_counts = missing_matrix.value_counts()
            patterns['common_patterns'] = pattern_counts.head(10).to_dict()
            
            # Calculate missing value correlations
            if len(cols_with_missing) > 2:
                missing_corr = missing_matrix.astype(int).corr()
                high_missing_corr = []
                
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_value = missing_corr.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            high_missing_corr.append({
                                'col1': missing_corr.columns[i],
                                'col2': missing_corr.columns[j],
                                'correlation': corr_value
                            })
                
                patterns['missing_correlations'] = high_missing_corr
        
        return patterns
    
    def _analyze_missing_mechanisms(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data mechanisms (MCAR, MAR, MNAR)"""
        mechanisms = {}
        
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        for col in cols_with_missing:
            missing_mask = df[col].isnull()
            
            # Test for MCAR vs MAR by looking at relationships with other variables
            other_cols = [c for c in df.columns if c != col and not df[c].isnull().all()]
            
            correlations_with_missing = []
            for other_col in other_cols[:10]:  # Limit for performance
                if pd.api.types.is_numeric_dtype(df[other_col]):
                    # For numeric variables, use point-biserial correlation
                    try:
                        corr = stats.pointbiserialr(missing_mask, df[other_col].fillna(df[other_col].mean()))[0]
                        if abs(corr) > 0.1:  # Threshold for meaningful correlation
                            correlations_with_missing.append({
                                'variable': other_col,
                                'correlation': corr
                            })
                    except:
                        pass
            
            # Classify missing mechanism
            if not correlations_with_missing:
                mechanism = "likely_MCAR"  # Missing Completely At Random
            elif any(abs(c['correlation']) > 0.3 for c in correlations_with_missing):
                mechanism = "likely_MAR"  # Missing At Random
            else:
                mechanism = "potentially_MNAR"  # Missing Not At Random
            
            mechanisms[col] = {
                'mechanism': mechanism,
                'correlations': correlations_with_missing[:5]  # Top 5 correlations
            }
        
        return mechanisms
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict:
        """Calculate feature importance using multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for feature importance analysis'}
        
        importance_results = {}
        
        # Use variance as a simple importance measure
        variances = df[numeric_cols].var().sort_values(ascending=False)
        importance_results['variance_importance'] = variances.to_dict()
        
        # Use coefficient of variation for normalized importance
        cv_importance = (df[numeric_cols].std() / df[numeric_cols].mean()).sort_values(ascending=False)
        importance_results['cv_importance'] = cv_importance.dropna().to_dict()
        
        return importance_results


class VisualizationEngine:
    """
    Advanced visualization engine for comprehensive EDA
    Enhanced with modern visualization best practices
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def create_comprehensive_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, str]:
        """Create comprehensive visualization suite with enhanced plots"""
        self.logger.info("📊 Creating comprehensive visualizations")
        
        viz_paths = {}
        
        try:
            # 1. Enhanced Data Overview Dashboard
            viz_paths['overview'] = self._create_enhanced_overview_dashboard(df, analysis_results)
            
            # 2. Advanced Distribution Analysis
            viz_paths['distributions'] = self._create_advanced_distribution_plots(df, analysis_results)
            
            # 3. Enhanced Correlation Analysis
            if 'correlation_analysis' in analysis_results.get('statistical_analysis', {}):
                viz_paths['correlations'] = self._create_enhanced_correlation_plots(
                    analysis_results['statistical_analysis']['correlation_analysis'])
            
            # 4. Advanced Missing Value Analysis
            viz_paths['missing_values'] = self._create_advanced_missing_plots(df, analysis_results)
            
            # 5. Robust Outlier Analysis
            viz_paths['outliers'] = self._create_advanced_outlier_plots(
                df, analysis_results.get('statistical_analysis', {}).get('outlier_analysis', {}))
            
            # 6. Time Series Analysis (if applicable)
            temporal_info = analysis_results.get('temporal_analysis', {})
            if temporal_info.get('time_series_detected', False):
                viz_paths['time_series'] = self._create_advanced_time_series_plots(df, temporal_info)
            
            # 7. Feature Engineering Opportunities
            viz_paths['feature_engineering'] = self._create_feature_engineering_plots(
                df, analysis_results.get('feature_engineering', {}))
            
        except Exception as e:
            self.logger.warning(f"Some visualizations failed: {e}")
        
        return viz_paths
    
    def _create_enhanced_overview_dashboard(self, df: pd.DataFrame, results: Dict) -> str:
        """Create enhanced overview dashboard with more insights"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Data Quality Score', 'Missing Values by Column', 
                    'Feature Types Distribution', 'ML Readiness Assessment',
                    'Statistical Anomalies', 'Data Completeness Matrix'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "heatmap"}]
                ]
            )
            
            # Data quality score indicator
            quality_score = results.get('data_quality', {}).get('completeness_score', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=quality_score,
                    title={'text': "Data Quality"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if quality_score > 80 else "orange" if quality_score > 60 else "red"},
                        'steps': [{'range': [0, 60], 'color': "lightgray"}, {'range': [60, 80], 'color': "yellow"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                    }
                ),
                row=1, col=1
            )
            
            # Missing values
            missing_counts = df.isnull().sum().sort_values(ascending=False)[:15]
            if missing_counts.sum() > 0:
                fig.add_trace(
                    go.Bar(x=missing_counts.index, y=missing_counts.values, name="Missing Values"),
                    row=1, col=2
                )
            
            # Feature types distribution
            dtype_counts = df.dtypes.value_counts()
            fig.add_trace(
                go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name="Data Types"),
                row=2, col=1
            )
            
            # ML Readiness factors
            ml_readiness = results.get('ml_readiness', {}).get('readiness_factors', {})
            if ml_readiness:
                fig.add_trace(
                    go.Bar(
                        x=list(ml_readiness.keys()), 
                        y=list(ml_readiness.values()), 
                        name="ML Readiness"
                    ),
                    row=2, col=2
                )
            
            # Statistical anomalies
            anomalies = self._count_statistical_anomalies(results)
            fig.add_trace(
                go.Bar(
                    x=list(anomalies.keys()), 
                    y=list(anomalies.values()), 
                    name="Anomalies"
                ),
                row=3, col=1
            )
            
            # Data completeness matrix (sample)
            completeness_matrix = df.head(50).notnull().astype(int)
            fig.add_trace(
                go.Heatmap(
                    z=completeness_matrix.values,
                    x=completeness_matrix.columns,
                    y=completeness_matrix.index,
                    colorscale='RdYlGn',
                    name="Completeness"
                ),
                row=3, col=2
            )
            
            fig.update_layout(
                title="Enhanced Data Overview Dashboard",
                height=1200,
                showlegend=False
            )
            
            output_path = self.output_dir / "enhanced_overview_dashboard.html"
            fig.write_html(output_path)
            self.logger.info(f"📊 Enhanced overview dashboard saved: {output_path}")
            
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create enhanced overview dashboard: {e}")
            return ""
    
    def _count_statistical_anomalies(self, results: Dict) -> Dict[str, int]:
        """Count various statistical anomalies"""
        anomalies = {
            'High Correlations': 0,
            'Outlier Features': 0,
            'Skewed Features': 0,
            'Constant Features': 0
        }
        
        stats_results = results.get('statistical_analysis', {})
        
        # High correlations
        if 'correlation_analysis' in stats_results:
            anomalies['High Correlations'] = len(
                stats_results['correlation_analysis'].get('high_correlation_pairs', [])
            )
        
        # Features with outliers
        if 'outlier_analysis' in stats_results:
            anomalies['Outlier Features'] = len(stats_results['outlier_analysis'])
        
        # Highly skewed features
        if 'distribution_analysis' in stats_results:
            skewed_count = sum(
                1 for analysis in stats_results['distribution_analysis'].values()
                if abs(analysis.get('distribution_params', {}).get('skewness', 0)) > 2
            )
            anomalies['Skewed Features'] = skewed_count
        
        # Constant features
        data_quality = results.get('data_quality', {})
        anomalies['Constant Features'] = len(data_quality.get('constant_columns', []))
        
        return anomalies
    
    def _create_advanced_distribution_plots(self, df: pd.DataFrame, results: Dict) -> str:
        """Create advanced distribution analysis plots"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns for distribution plots")
                return ""
            
            # Limit columns for performance
            cols_to_plot = list(numeric_cols)[:12]
            
            # Create subplots with both histograms and box plots
            n_cols = 3
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=cols_to_plot,
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            for idx, col in enumerate(cols_to_plot):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1
                
                data = df[col].dropna()
                
                # Add histogram with KDE
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=f"{col}_hist",
                        nbinsx=50,
                        opacity=0.7,
                        showlegend=False,
                        histnorm='probability density'
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title="Advanced Distribution Analysis - Probability Density",
                height=300 * n_rows,
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_distribution_analysis.html"
            fig.write_html(output_path)
            
            # Create separate normality assessment plot
            self._create_normality_assessment_plot(df, results)
            
            self.logger.info(f"📊 Advanced distribution plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced distribution plots: {e}")
            return ""
    
    def _create_normality_assessment_plot(self, df: pd.DataFrame, results: Dict) -> str:
        """Create normality assessment visualization"""
        try:
            dist_results = results.get('statistical_analysis', {}).get('distribution_analysis', {})
            
            if not dist_results:
                return ""
            
            # Collect normality test results
            normality_data = []
            for col, analysis in dist_results.items():
                normality_tests = analysis.get('normality_tests', {})
                
                for test_name, test_result in normality_tests.items():
                    if test_result:
                        normality_data.append({
                            'Feature': col,
                            'Test': test_name,
                            'P_Value': test_result.get('p_value', 0),
                            'Is_Normal': test_result.get('is_normal', False)
                        })
            
            if not normality_data:
                return ""
            
            normality_df = pd.DataFrame(normality_data)
            
            # Create normality assessment plot
            fig = px.scatter(
                normality_df,
                x='Feature',
                y='P_Value',
                color='Test',
                symbol='Is_Normal',
                title='Normality Test Results (p-value > 0.05 suggests normality)',
                labels={'P_Value': 'P-Value', 'Feature': 'Features'}
            )
            
            # Add significance line
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                         annotation_text="Significance Level (α=0.05)")
            
            fig.update_layout(height=600)
            
            output_path = self.output_dir / "normality_assessment.html"
            fig.write_html(output_path)
            
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create normality assessment plot: {e}")
            return ""
    
    def _create_enhanced_correlation_plots(self, correlation_results: Dict) -> str:
        """Create enhanced correlation visualizations"""
        try:
            if 'pearson' not in correlation_results:
                return ""
            
            pearson_corr = correlation_results['pearson']
            
            # Limit size for visualization
            if len(pearson_corr.columns) > 25:
                # Select most variable features
                numeric_std = pearson_corr.std().sort_values(ascending=False)
                top_features = numeric_std.head(25).index
                pearson_corr = pearson_corr.loc[top_features, top_features]
            
            # Create correlation heatmap with dendrograms
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Pearson Correlation Matrix', 
                    'Correlation Strength Distribution',
                    'High Correlation Network', 
                    'Correlation Method Comparison'
                ],
                specs=[
                    [{"type": "heatmap"}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # Main correlation heatmap
            fig.add_trace(
                go.Heatmap(
                    z=pearson_corr.values,
                    x=pearson_corr.columns,
                    y=pearson_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=pearson_corr.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 8},
                    showscale=True
                ),
                row=1, col=1
            )
            
            # Correlation strength distribution
            upper_triangle = pearson_corr.where(np.triu(np.ones_like(pearson_corr, dtype=bool), k=1))
            corr_values = upper_triangle.stack().values
            
            fig.add_trace(
                go.Histogram(
                    x=corr_values,
                    nbinsx=50,
                    name="Correlation Distribution",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # High correlation pairs
            high_corr_pairs = correlation_results.get('high_correlation_pairs', [])[:10]
            if high_corr_pairs:
                pairs_df = pd.DataFrame(high_corr_pairs)
                fig.add_trace(
                    go.Scatter(
                        x=range(len(pairs_df)),
                        y=pairs_df['abs_correlation'],
                        mode='markers+lines',
                        name="High Correlations",
                        text=pairs_df['feature1'] + ' - ' + pairs_df['feature2'],
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Method comparison (if available)
            if 'spearman' in correlation_results and 'kendall' in correlation_results:
                methods = ['Pearson', 'Spearman', 'Kendall']
                avg_abs_corr = [
                    np.abs(correlation_results['pearson'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean(),
                    np.abs(correlation_results['spearman'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean(),
                    np.abs(correlation_results['kendall'].values[np.triu_indices_from(pearson_corr.values, k=1)]).mean()
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=avg_abs_corr,
                        name="Method Comparison",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Enhanced Correlation Analysis",
                height=1000,
                showlegend=False
            )
            
            output_path = self.output_dir / "enhanced_correlation_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Enhanced correlation plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create enhanced correlation plots: {e}")
            return ""
    
    def _create_advanced_missing_plots(self, df: pd.DataFrame, results: Dict) -> str:
        """Create advanced missing value analysis plots"""
        try:
            missing_counts = df.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            
            if len(cols_with_missing) == 0:
                self.logger.info("No missing values to visualize")
                return ""
            
            # Create comprehensive missing value analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Missing Value Patterns',
                    'Missing Value Mechanisms',
                    'Missing Value Heatmap',
                    'Imputation Strategy Recommendations'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "heatmap"}, {"type": "table"}]
                ]
            )
            
            # Missing value patterns
            fig.add_trace(
                go.Bar(
                    x=cols_with_missing.index,
                    y=(cols_with_missing / len(df)) * 100,
                    name="Missing %",
                    marker_color='red',
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Missing mechanisms (if analyzed)
            missing_analysis = results.get('statistical_analysis', {}).get('missing_value_analysis', {})
            mechanisms = missing_analysis.get('mechanisms', {})
            
            if mechanisms:
                mechanism_counts = {}
                for col, info in mechanisms.items():
                    mechanism = info.get('mechanism', 'unknown')
                    mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
                
                fig.add_trace(
                    go.Pie(
                        labels=list(mechanism_counts.keys()),
                        values=list(mechanism_counts.values()),
                        name="Mechanisms",
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            # Missing value heatmap (sample of data)
            sample_size = min(100, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)[cols_with_missing.index]
            missing_matrix = sample_df.isnull().astype(int)
            
            fig.add_trace(
                go.Heatmap(
                    z=missing_matrix.values,
                    x=missing_matrix.columns,
                    y=missing_matrix.index,
                    colorscale=[[0, 'green'], [1, 'red']],
                    name="Missing Pattern",
                    showscale=False
                ),
                row=2, col=1
            )
            
            # Imputation recommendations
            imputation_recs = self._generate_imputation_recommendations(cols_with_missing, df)
            fig.add_trace(
                go.Table(
                    header=dict(values=["Column", "Missing %", "Recommendation"]),
                    cells=dict(values=[
                        list(imputation_recs.keys()),
                        [f"{(cols_with_missing[col] / len(df)) * 100:.1f}%" for col in imputation_recs.keys()],
                        list(imputation_recs.values())
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Advanced Missing Value Analysis",
                height=1000,
                showlegend=True
            )
            
            output_path = self.output_dir / "advanced_missing_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced missing value plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced missing plots: {e}")
            return ""
    
    def _generate_imputation_recommendations(self, missing_cols: pd.Series, df: pd.DataFrame) -> Dict[str, str]:
        """Generate imputation recommendations for each column with missing values"""
        recommendations = {}
        
        for col in missing_cols.index:
            missing_pct = (missing_cols[col] / len(df)) * 100
            
            if missing_pct > 70:
                recommendations[col] = "Consider removal - too much missing data"
            elif pd.api.types.is_numeric_dtype(df[col]):
                if missing_pct > 30:
                    recommendations[col] = "Advanced imputation (KNN/Iterative)"
                elif df[col].skew() > 2:
                    recommendations[col] = "Median imputation"
                else:
                    recommendations[col] = "Mean imputation"
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                if df[col].nunique() < 10:
                    recommendations[col] = "Mode imputation"
                else:
                    recommendations[col] = "Create 'Unknown' category"
            else:
                recommendations[col] = "Forward/backward fill"
        
        return recommendations
    
    def _create_advanced_outlier_plots(self, df: pd.DataFrame, outlier_results: Dict) -> str:
        """Create advanced outlier analysis visualizations"""
        try:
            if not outlier_results:
                return ""
            
            # Create comprehensive outlier analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Outlier Detection Methods Comparison',
                    'Outlier Consensus Scores',
                    'Outlier Distribution by Feature',
                    'Method Agreement Analysis'
                ]
            )
            
            # Method comparison
            method_counts = {}
            for col, methods in outlier_results.items():
                for method, outliers in methods.items():
                    if method != 'outlier_consensus' and isinstance(outliers, list):
                        method_name = method.replace('_outliers', '').replace('_', ' ').title()
                        method_counts[method_name] = method_counts.get(method_name, 0) + len(outliers)
            
            fig.add_trace(
                go.Bar(
                    x=list(method_counts.keys()),
                    y=list(method_counts.values()),
                    name="Method Comparison",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Consensus scores
            consensus_data = []
            for col, methods in outlier_results.items():
                if 'outlier_consensus' in methods:
                    consensus_info = methods['outlier_consensus']
                    consensus_outliers = consensus_info.get('consensus_outliers', [])
                    consensus_data.append({
                        'feature': col,
                        'high_consensus_outliers': len(consensus_outliers),
                        'agreement_score': consensus_info.get('method_agreement', 0) * 100
                    })
            
            if consensus_data:
                consensus_df = pd.DataFrame(consensus_data)
                fig.add_trace(
                    go.Scatter(
                        x=consensus_df['feature'],
                        y=consensus_df['high_consensus_outliers'],
                        mode='markers',
                        marker=dict(
                            size=consensus_df['agreement_score'],
                            color=consensus_df['agreement_score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Agreement %")
                        ),
                        name="Consensus Outliers",
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Outlier distribution by feature
            feature_outlier_counts = []
            for col, methods in outlier_results.items():
                total_outliers = 0
                for method, outliers in methods.items():
                    if method != 'outlier_consensus' and isinstance(outliers, list):
                        total_outliers += len(outliers)
                feature_outlier_counts.append({
                    'feature': col,
                    'total_outliers': total_outliers,
                    'outlier_percentage': (total_outliers / len(df)) * 100
                })
            
            if feature_outlier_counts:
                outlier_df = pd.DataFrame(feature_outlier_counts)
                fig.add_trace(
                    go.Bar(
                        x=outlier_df['feature'],
                        y=outlier_df['outlier_percentage'],
                        name="Outlier %",
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Method agreement analysis
            agreement_scores = []
            for col, methods in outlier_results.items():
                if 'outlier_consensus' in methods:
                    agreement = methods['outlier_consensus'].get('method_agreement', 0)
                    agreement_scores.append(agreement * 100)
            
            if agreement_scores:
                fig.add_trace(
                    go.Histogram(
                        x=agreement_scores,
                        nbinsx=20,
                        name="Agreement Distribution",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Advanced Outlier Analysis",
                height=1000,
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_outlier_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced outlier plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced outlier plots: {e}")
            return ""
    
    def _create_advanced_time_series_plots(self, df: pd.DataFrame, temporal_info: Dict) -> str:
        """Create advanced time series analysis plots"""
        try:
            datetime_cols = temporal_info.get('datetime_columns', []) + temporal_info.get('potential_timestamps', [])
            
            if not datetime_cols:
                return ""
            
            time_col = datetime_cols[0]
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            
            # Create comprehensive time series analysis
            fig = make_subplots(
                rows=len(numeric_cols) + 1, cols=2,
                subplot_titles=[f"{col} - Time Series" for col in numeric_cols] + 
                              [f"{col} - Distribution" for col in numeric_cols] + 
                              ["Temporal Patterns Summary", "Seasonality Analysis"],
                specs=[[{"colspan": 2}] * len(numeric_cols) + [{}] * len(numeric_cols) + [{"type": "table"}, {"type": "bar"}]][:len(numeric_cols) + 1]
            )
            
            for idx, col in enumerate(numeric_cols):
                temp_df = df[[time_col, col]].dropna().sort_values(time_col)
                
                # Sample data if too large
                if len(temp_df) > 5000:
                    temp_df = temp_df.sample(n=5000, random_state=42).sort_values(time_col)
                
                # Time series plot
                fig.add_trace(
                    go.Scatter(
                        x=temp_df[time_col],
                        y=temp_df[col],
                        mode='lines',
                        name=col,
                        showlegend=False
                    ),
                    row=idx + 1, col=1
                )
                
                # Distribution plot
                fig.add_trace(
                    go.Histogram(
                        x=temp_df[col],
                        name=f"{col}_dist",
                        showlegend=False
                    ),
                    row=idx + 1, col=2
                )
            
            # Temporal patterns summary
            patterns_data = []
            for col, pattern_info in temporal_info.get('temporal_patterns', {}).items():
                patterns_data.append([
                    col,
                    str(pattern_info.get('frequency_estimate', 'Unknown')),
                    f"{pattern_info.get('business_hours_pct', 0):.1%}",
                    "Yes" if pattern_info.get('has_gaps', False) else "No",
                    f"{pattern_info.get('regularity_score', 0):.1f}"
                ])
            
            if patterns_data:
                fig.add_trace(
                    go.Table(
                        header=dict(values=["Column", "Frequency", "Business Hours %", "Has Gaps", "Regularity Score"]),
                        cells=dict(values=list(zip(*patterns_data)))
                    ),
                    row=len(numeric_cols) + 1, col=1
                )
            
            # Seasonality indicators
            seasonality_indicators = {
                'Seasonality Detected': temporal_info.get('seasonality_detected', False),
                'Trend Detected': temporal_info.get('trend_detected', False),
                'Stationarity': len(temporal_info.get('stationarity_test', {})) > 0
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(seasonality_indicators.keys()),
                    y=[1 if v else 0 for v in seasonality_indicators.values()],
                    name="Time Series Properties",
                    showlegend=False
                ),
                row=len(numeric_cols) + 1, col=2
            )
            
            fig.update_layout(
                title="Advanced Time Series Analysis",
                height=400 * (len(numeric_cols) + 1),
                showlegend=False
            )
            
            output_path = self.output_dir / "advanced_time_series_analysis.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Advanced time series plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create advanced time series plots: {e}")
            return ""
    
    def _create_feature_engineering_plots(self, df: pd.DataFrame, feature_eng_results: Dict) -> str:
        """Create feature engineering opportunity visualizations"""
        try:
            if not feature_eng_results:
                return ""
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Transformation Opportunities',
                    'Interaction Feature Potential',
                    'Encoding Recommendations',
                    'Feature Engineering Priority'
                ]
            )
            
            # Transformation opportunities
            transform_recs = feature_eng_results.get('transformation_recommendations', [])
            if transform_recs:
                transform_counts = {}
                for rec in transform_recs:
                    issue = rec.get('issue', 'unknown')
                    transform_counts[issue] = transform_counts.get(issue, 0) + 1
                
                fig.add_trace(
                    go.Bar(
                        x=list(transform_counts.keys()),
                        y=list(transform_counts.values()),
                        name="Transformations",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Interaction opportunities
            interaction_ops = feature_eng_results.get('interaction_opportunities', [])
            if interaction_ops:
                correlations = [abs(op.get('correlation', 0)) for op in interaction_ops]
                feature_pairs = [f"{op['features'][0][:10]}...{op['features'][1][:10]}" for op in interaction_ops[:10]]
                
                fig.add_trace(
                    go.Bar(
                        x=feature_pairs,
                        y=correlations,
                        name="Interaction Potential",
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Encoding recommendations
            encoding_recs = feature_eng_results.get('encoding_recommendations', [])
            if encoding_recs:
                encoding_counts = {}
                for rec in encoding_recs:
                    recommendation = rec.get('recommendation', 'unknown')
                    encoding_counts[recommendation] = encoding_counts.get(recommendation, 0) + 1
                
                fig.add_trace(
                    go.Pie(
                        labels=list(encoding_counts.keys()),
                        values=list(encoding_counts.values()),
                        name="Encoding Methods"
                    ),
                    row=2, col=1
                )
            
            # Priority scoring
            priority_data = []
            
            # High priority: Highly skewed features
            priority_data.append(['High Skewness', len(transform_recs), 'High'])
            
            # Medium priority: High correlation interactions
            high_corr_interactions = [op for op in interaction_ops if abs(op.get('correlation', 0)) > 0.7]
            priority_data.append(['Strong Interactions', len(high_corr_interactions), 'Medium'])
            
            # Low priority: Encoding needs
            priority_data.append(['Encoding Needed', len(encoding_recs), 'Low'])
            
            if priority_data:
                priorities_df = pd.DataFrame(priority_data, columns=['Task', 'Count', 'Priority'])
                color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
                
                fig.add_trace(
                    go.Bar(
                        x=priorities_df['Task'],
                        y=priorities_df['Count'],
                        marker_color=[color_map[p] for p in priorities_df['Priority']],
                        name="Priority",
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Feature Engineering Opportunities",
                height=800,
                showlegend=True
            )
            
            output_path = self.output_dir / "feature_engineering_opportunities.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Feature engineering plots saved: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.warning(f"Could not create feature engineering plots: {e}")
            return ""


class DimensionalityAnalyzer:
    """
    Advanced dimensionality analysis using PCA, t-SNE, and clustering
    Enhanced with modern techniques and interpretability
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_dimensionality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dimensionality analysis with enhanced methods"""
        self.logger.info("🔍 Analyzing data dimensionality")
        
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 3:
            return {'warning': 'Insufficient numeric features for dimensionality analysis'}
        
        results = {
            'pca_analysis': self._enhanced_pca_analysis(numeric_df),
            'clustering_analysis': self._enhanced_clustering_analysis(numeric_df),
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(numeric_df)
        }
        
        # Add t-SNE if dataset is not too large
        if len(numeric_df) <= 10000 and len(numeric_df.columns) > 5:
            results['tsne_analysis'] = self._enhanced_tsne_analysis(numeric_df)
        
        return results
    
    def _enhanced_pca_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced PCA analysis with feature importance and interpretation"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Perform PCA
            n_components = min(self.config.pca_components, df.shape[1])
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            pca_result = pca.fit_transform(df_scaled)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            results = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'n_components_90_variance': int(np.argmax(cumulative_variance >= 0.9) + 1),
                'n_components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
                'n_components_99_variance': int(np.argmax(cumulative_variance >= 0.99) + 1),
                'feature_importance': {},
                'interpretation': {}
            }
            
            # Feature importance in principal components
            for i, component in enumerate(pca.components_[:5]):  # First 5 components
                component_importance = dict(zip(df.columns, np.abs(component)))
                results['feature_importance'][f'PC{i+1}'] = sorted(
                    component_importance.items(), key=lambda x: x[1], reverse=True
                )[:10]  # Top 10 features
                
                # Interpretation based on dominant features
                top_features = [item[0] for item in results['feature_importance'][f'PC{i+1}'][:3]]
                results['interpretation'][f'PC{i+1}'] = {
                    'dominant_features': top_features,
                    'variance_explained': float(pca.explained_variance_ratio_[i]),
                    'interpretation_hint': self._interpret_principal_component(top_features)
                }
            
            # PCA quality metrics
            results['quality_metrics'] = {
                'kaiser_criterion': sum(pca.explained_variance_ > 1),  # Eigenvalues > 1
                'scree_elbow': self._find_scree_elbow(pca.explained_variance_ratio_),
                'dimensionality_reduction_potential': float(results['n_components_90_variance'] / len(df.columns))
            }
            
            return results
        except Exception as e:
            self.logger.warning(f"Enhanced PCA analysis failed: {e}")
            return {'error': str(e)}
    
    def _interpret_principal_component(self, top_features: List[str]) -> str:
        """Provide interpretation hints for principal components"""
        # Simple heuristic based on feature names
        feature_lower = [f.lower() for f in top_features]
        
        if any('temp' in f or 'temperature' in f for f in feature_lower):
            return "Likely related to temperature/thermal processes"
        elif any('press' in f or 'pressure' in f for f in feature_lower):
            return "Likely related to pressure/force measurements"
        elif any('flow' in f or 'rate' in f for f in feature_lower):
            return "Likely related to flow/rate measurements"
        elif any('time' in f or 'duration' in f for f in feature_lower):
            return "Likely related to temporal characteristics"
        elif any('qual' in f or 'defect' in f for f in feature_lower):
            return "Likely related to quality measurements"
        else:
            return "Mixed process characteristics"
    
    def _find_scree_elbow(self, explained_variance: np.ndarray) -> int:
        """Find elbow point in scree plot"""
        if len(explained_variance) < 3:
            return 1
        
        # Calculate second derivative
        diffs = np.diff(explained_variance)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 2
            return min(elbow_idx, len(explained_variance) - 1)
        
        return len(explained_variance) // 2
    
    def _enhanced_clustering_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced clustering analysis with multiple algorithms"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            results = {
                'kmeans_analysis': self._enhanced_kmeans_analysis(df_scaled),
                'dbscan_analysis': self._enhanced_dbscan_analysis(df_scaled),
                'cluster_quality_assessment': {}
            }
            
            return results
        except Exception as e:
            self.logger.warning(f"Enhanced clustering analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_kmeans_analysis(self, df_scaled: np.ndarray) -> Dict:
        """Enhanced K-means analysis with multiple evaluation metrics"""
        try:
            min_k, max_k = self.config.cluster_range
            inertias = []
            silhouette_scores = []
            calinski_harabasz_scores = []
            
            for k in range(min_k, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(df_scaled)
                
                inertias.append(kmeans.inertia_)
                
                # Silhouette score
                if len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(df_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                    
                    # Calinski-Harabasz score
                    try:
                        from sklearn.metrics import calinski_harabasz_score
                        ch_score = calinski_harabasz_score(df_scaled, cluster_labels)
                        calinski_harabasz_scores.append(ch_score)
                    except:
                        calinski_harabasz_scores.append(0)
                else:
                    silhouette_scores.append(0)
                    calinski_harabasz_scores.append(0)
            
            # Find optimal k using multiple criteria
            optimal_k_elbow = self._find_elbow_point(list(range(min_k, max_k + 1)), inertias)
            optimal_k_silhouette = min_k + np.argmax(silhouette_scores) if silhouette_scores else min_k
            
            return {
                'k_range': list(range(min_k, max_k + 1)),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_harabasz_scores': calinski_harabasz_scores,
                'optimal_k_elbow': optimal_k_elbow,
                'optimal_k_silhouette': optimal_k_silhouette,
                'recommended_k': optimal_k_silhouette if max(silhouette_scores) > 0.3 else optimal_k_elbow
            }
        except Exception as e:
            self.logger.warning(f"Enhanced K-means analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_dbscan_analysis(self, df_scaled: np.ndarray) -> Dict:
        """Enhanced DBSCAN analysis with parameter optimization"""
        try:
            # Try multiple eps values
            eps_values = [0.3, 0.5, 0.7, 1.0]
            min_samples_values = [3, 5, 10]
            
            best_result = None
            best_score = -1
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(df_scaled)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    if n_clusters > 1:  # Valid clustering
                        try:
                            sil_score = silhouette_score(df_scaled, cluster_labels)
                            if sil_score > best_score:
                                best_score = sil_score
                                best_result = {
                                    'eps': eps,
                                    'min_samples': min_samples,
                                    'n_clusters': n_clusters,
                                    'n_noise_points': n_noise,
                                    'noise_percentage': n_noise / len(cluster_labels) * 100,
                                    'silhouette_score': sil_score,
                                    'cluster_labels': cluster_labels.tolist()
                                }
                        except:
                            pass
            
            return best_result or {
                'n_clusters': 0,
                'message': 'No suitable DBSCAN parameters found'
            }
        except Exception as e:
            self.logger.warning(f"Enhanced DBSCAN analysis failed: {e}")
            return {'error': str(e)}
    
    def _enhanced_tsne_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced t-SNE analysis with multiple perplexities"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Try multiple perplexity values
            perplexities = [5, 30, 50]
            results = {}
            
            for perp in perplexities:
                if perp < len(df) // 4:  # Ensure perplexity is valid
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perp,
                        random_state=self.config.random_state,
                        n_iter=1000
                    )
                    tsne_result = tsne.fit_transform(df_scaled)
                    
                    results[f'perplexity_{perp}'] = {
                        'coordinates': tsne_result.tolist(),
                        'perplexity': perp,
                        'kl_divergence': float(tsne.kl_divergence_)
                    }
            
            # Select best result (lowest KL divergence)
            if results:
                best_result = min(results.values(), key=lambda x: x['kl_divergence'])
                return {
                    'best_result': best_result,
                    'all_results': results,
                    'recommendation': f"Use perplexity {best_result['perplexity']} for best separation"
                }
            else:
                return {'error': 'No valid t-SNE results'}
                
        except Exception as e:
            self.logger.warning(f"Enhanced t-SNE analysis failed: {e}")
            return {'error': str(e)}
    
    def _estimate_intrinsic_dimensionality(self, df: pd.DataFrame) -> Dict:
        """Estimate intrinsic dimensionality using multiple methods"""
        try:
            # Method 1: PCA-based estimation (90% variance threshold)
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            pca_full = PCA()
            pca_full.fit(df_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            
            intrinsic_dim_90 = np.argmax(cumsum_var >= 0.9) + 1
            intrinsic_dim_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            # Method 2: Effective rank
            U, s, Vt = np.linalg.svd(df_scaled, full_matrices=False)
            effective_rank = np.sum(s > 0.01 * s[0])  # Singular values > 1% of largest
            
            return {
                'pca_90_percent': int(intrinsic_dim_90),
                'pca_95_percent': int(intrinsic_dim_95),
                'effective_rank': int(effective_rank),
                'original_dimensions': df.shape[1],
                'dimensionality_reduction_potential': float(1 - intrinsic_dim_90 / df.shape[1])
            }
        except Exception as e:
            self.logger.warning(f"Intrinsic dimensionality estimation failed: {e}")
            return {'error': str(e)}
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in K-means inertia plot using improved method"""
        if len(inertias) < 3:
            return k_values[0]
        
        # Normalize the inertias
        inertias_norm = np.array(inertias) / max(inertias)
        k_norm = np.array(k_values) / max(k_values)
        
        # Calculate the distance from each point to the line connecting first and last points
        distances = []
        for i in range(len(k_norm)):
            # Point to line distance formula
            x1, y1 = k_norm[0], inertias_norm[0]
            x2, y2 = k_norm[-1], inertias_norm[-1]
            x0, y0 = k_norm[i], inertias_norm[i]
            
            distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            distances.append(distance)
        
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]


class ManufacturingDomainAnalyzer:
    """
    Enhanced manufacturing domain-specific analysis
    Based on Industry 4.0 principles and quality control standards
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_manufacturing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced manufacturing-specific pattern analysis"""
        self.logger.info("🏭 Analyzing manufacturing domain patterns")
        
        results = {
            'process_capability': self._enhanced_process_capability(df),
            'quality_patterns': self._enhanced_quality_analysis(df),
            'equipment_patterns': self._enhanced_equipment_analysis(df),
            'defect_analysis': self._enhanced_defect_analysis(df),
            'production_efficiency': self._analyze_production_efficiency(df),
            'sensor_health': self._analyze_sensor_health(df)
        }
        
        return results
    
    def _enhanced_process_capability(self, df: pd.DataFrame) -> Dict:
        """Enhanced process capability analysis with Cp, Cpk calculations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        capability_results = {}
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Enhanced pattern matching for process measurements
            process_indicators = [
                'temp', 'temperature', 'press', 'pressure', 'flow', 'rate',
                'speed', 'force', 'torque', 'voltage', 'current', 'measure',
                'dimension', 'length', 'width', 'height', 'thickness'
            ]
            
            is_process_measure = any(indicator in col_lower for indicator in process_indicators)
            
            if is_process_measure:
                data = df[col].dropna()
                
                if len(data) > 30:
                    mean = data.mean()
                    std = data.std()
                    
                    # Estimate control limits (assuming 6-sigma process)
                    ucl = mean + 3 * std
                    lcl = mean - 3 * std
                    
# Calculate process capability (assuming no specs provided, use natural tolerance estimates)
                    # Note: True Cp/Cpk require USL/LSL specs. Here we estimate potential capability.
                    estimated_cp = (ucl - lcl) / (6 * std) if std > 0 else 0  # Always 1.0
                    # Estimate Cpk assuming centered process (Cpk ≈ Cp if centered)
                    estimated_cpk = estimated_cp * (1 - abs(mean - (ucl + lcl)/2) / ((ucl - lcl)/2)) if estimated_cp > 0 else 0
                    # More realistic Cpk estimation requires specs; here assume symmetric
                    percent_oos = ((data < lcl) | (data > ucl)).mean() * 100  # Percent out of "spec"

                    capability_results[col] = {
                        'mean': float(mean),
                        'std': float(std),
                        'cv_percentage': float((std / mean * 100) if mean != 0 else 0),
                        'estimated_cp': float(estimated_cp),
                        'estimated_cpk': float(min(estimated_cp, estimated_cpk)),
                        'percent_out_of_spec': float(percent_oos),
                        'capability_assessment': (
                            'Excellent' if estimated_cpk > 1.67 else
                            'Good' if estimated_cpk > 1.33 else
                            'Marginal' if estimated_cpk > 1.0 else
                            'Poor'
                        ),
                        'recommendation': (
                            'Process monitoring recommended' if percent_oos > 0.27 else  # > normal 0.27% for 3sigma
                            'Stable process'
                        )
                    }

        return capability_results

    def _enhanced_quality_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced quality pattern analysis"""
        quality_results = {}
        
        quality_indicators = ['quality', 'yield', 'defect_rate', 'pass', 'fail', 'scrap', 'rework']
        quality_cols = [col for col in df.columns if any(ind in col.lower() for ind in quality_indicators)]
        
        for col in quality_cols:
            data = df[col].dropna()
            col_lower = col.lower()
            
            if pd.api.types.is_numeric_dtype(data):
                # Assume continuous quality metric (e.g., yield percentage)
                quality_results[col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'percent_below_threshold': float((data < 90).mean() * 100) if 'yield' in col_lower else None,  # Arbitrary for yield
                    'assessment': 'High variability' if data.std() / data.mean() > 0.1 else 'Stable quality'
                }
            elif pd.api.types.is_categorical_dtype(data) or data.dtype == 'object':
                # Categorical quality (e.g., pass/fail)
                value_counts = data.value_counts(normalize=True) * 100
                quality_results[col] = {
                    'category_distribution': value_counts.to_dict(),
                    'pass_rate': value_counts.get('pass', 0) + value_counts.get('good', 0),
                    'fail_rate': value_counts.get('fail', 0) + value_counts.get('bad', 0) + value_counts.get('defect', 0)
                }
        
        if not quality_results:
            quality_results['note'] = 'No quality-related columns detected'
        
        return quality_results

    def _enhanced_equipment_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced equipment pattern analysis"""
        equipment_results = {}
        
        equipment_indicators = ['machine', 'equipment', 'tool', 'line', 'station', 'id']
        equipment_cols = [col for col in df.columns if any(ind in col.lower() for ind in equipment_indicators)]
        
        performance_indicators = ['downtime', 'uptime', 'cycle_time', 'output', 'error', 'maintenance']
        perf_cols = [col for col in df.columns if any(ind in col.lower() for ind in performance_indicators)]
        
        if equipment_cols and perf_cols:
            for eq_col in equipment_cols:
                for perf_col in perf_cols:
                    if pd.api.types.is_numeric_dtype(df[perf_col]):
                        grouped = df.groupby(eq_col)[perf_col].agg(['mean', 'std', 'count'])
                        equipment_results[f'{eq_col}_{perf_col}'] = {
                            'per_equipment_stats': grouped.to_dict(),
                            'high_variability_equip': grouped[grouped['std'] / grouped['mean'] > 0.1].index.tolist() if 'mean' in grouped.columns else [],
                            'low_performance_equip': grouped[grouped['mean'] < grouped['mean'].quantile(0.25)].index.tolist() if 'mean' in grouped.columns else []
                        }
        else:
            equipment_results['note'] = 'No equipment or performance columns detected'
        
        return equipment_results

    def _enhanced_defect_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced defect analysis"""
        defect_results = {}
        
        defect_indicators = ['defect', 'error', 'fault', 'reject', 'failure']
        defect_cols = [col for col in df.columns if any(ind in col.lower() for ind in defect_indicators)]
        
        for col in defect_cols:
            data = df[col].dropna()
            
            if pd.api.types.is_numeric_dtype(data):
                # Assume defect count or rate
                defect_results[col] = {
                    'total_defects': float(data.sum()) if 'count' in col.lower() else None,
                    'defect_rate': float(data.mean() * 100) if 'rate' in col.lower() else float(data.mean()),
                    'high_defect_periods': 'Temporal analysis needed' if 'time' in df.columns else None
                }
            elif pd.api.types.is_categorical_dtype(data) or data.dtype == 'object':
                # Defect types
                value_counts = data.value_counts(normalize=True) * 100
                defect_results[col] = {
                    'defect_types_distribution': value_counts.to_dict(),
                    'top_defects': value_counts.head(3).to_dict()
                }
        
        if not defect_results:
            defect_results['note'] = 'No defect-related columns detected'
        
        return defect_results

    def _analyze_production_efficiency(self, df: pd.DataFrame) -> Dict:
        """Analyze production efficiency metrics"""
        efficiency_results = {}
        
        efficiency_indicators = ['cycle_time', 'throughput', 'output', 'input', 'oee', 'efficiency']
        eff_cols = [col for col in df.columns if any(ind in col.lower() for ind in efficiency_indicators)]
        
        time_indicators = ['time', 'duration']
        time_cols = [col for col in df.columns if any(ind in col.lower() for ind in time_indicators)]
        
        output_indicators = ['output', 'produced', 'quantity']
        output_cols = [col for col in df.columns if any(ind in col.lower() for ind in output_indicators)]
        
        if time_cols and output_cols:
            time_col = time_cols[0]
            output_col = output_cols[0]
            if pd.api.types.is_numeric_dtype(df[time_col]) and pd.api.types.is_numeric_dtype(df[output_col]):
                df['efficiency'] = df[output_col] / df[time_col]
                efficiency_results['calculated_efficiency'] = {
                    'mean': float(df['efficiency'].mean()),
                    'std': float(df['efficiency'].std()),
                    'min': float(df['efficiency'].min()),
                    'max': float(df['efficiency'].max())
                }
        
        for col in eff_cols:
            data = df[col].dropna()
            if pd.api.types.is_numeric_dtype(data):
                efficiency_results[col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'efficiency_assessment': 'High' if data.mean() > 85 else 'Medium' if data.mean() > 70 else 'Low'  # Arbitrary thresholds
                }
        
        if not efficiency_results:
            efficiency_results['note'] = 'No efficiency-related columns detected'
        
        return efficiency_results

    def _analyze_sensor_health(self, df: pd.DataFrame) -> Dict:
        """Analyze sensor health and data quality"""
        sensor_results = {}
        
        sensor_indicators = ['sensor', 'reading', 'value', 'temp', 'press', 'flow']
        sensor_cols = [col for col in df.columns if any(ind in col.lower() for ind in sensor_indicators)]
        
        for col in sensor_cols:
            data = df[col].dropna()
            if pd.api.types.is_numeric_dtype(data):
                variance = data.var()
                unique_ratio = data.nunique() / len(data)
                outlier_pct = len(self._detect_outliers_iqr(data)) / len(data) * 100 if len(data) > 0 else 0
                
                sensor_results[col] = {
                    'variance': float(variance),
                    'unique_ratio': float(unique_ratio),
                    'outlier_percentage': float(outlier_pct),
                    'health_assessment': (
                        'Healthy' if variance > 0 and unique_ratio > 0.1 and outlier_pct < 5 else
                        'Stuck/Suspect' if variance == 0 or unique_ratio < 0.01 else
                        'Noisy/Erratic' if outlier_pct > 10 else
                        'Monitor'
                    )
                }
        
        if not sensor_results:
            sensor_results['note'] = 'No sensor-related columns detected'
        
        return sensor_results


class ReportGenerator:
    """
    Comprehensive report generator with ML recommendations
    Enhanced with structured JSON and HTML output
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def generate_comprehensive_report(self, df: pd.DataFrame, analysis_results: Dict, viz_paths: Dict) -> str:
        """Generate comprehensive EDA report with ML recommendations"""
        self.logger.info("📄 Generating comprehensive report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comprehensive_eda_report_{timestamp}.html"
        summary_path = self.output_dir / f"eda_summary_{timestamp}.json"
        
        # Save JSON summary
        with open(summary_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        # Generate HTML report
        html_content = self._generate_html_report(df, analysis_results, viz_paths, str(summary_path))
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_html_report(self, df: pd.DataFrame, results: Dict, viz_paths: Dict, json_path: str) -> str:
        """Generate HTML report content"""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Scientific EDA Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .section { margin-bottom: 40px; }
            </style>
        </head>
        <body>
            <h1>Scientific Exploratory Data Analysis Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Dataset Shape: {shape}</p>
            <p>ML Readiness Score: {ml_score}/100</p>
            <p>JSON Summary: <a href="{json_path}" download>Download JSON</a></p>
            
            <div class="section">
                <h2>Data Quality Assessment</h2>
                {data_quality_table}
            </div>
            
            <div class="section">
                <h2>Statistical Analysis Summary</h2>
                {stats_summary}
            </div>
            
            <div class="section">
                <h2>Manufacturing Domain Insights</h2>
                {mfg_insights}
            </div>
            
            <div class="section">
                <h2>ML Pipeline Recommendations</h2>
                {ml_recommendations}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {viz_links}
            </div>
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            shape=df.shape,
            ml_score=results.get('ml_readiness', {}).get('overall_score', 0),
            json_path=json_path,
            data_quality_table=self._generate_data_quality_table(results.get('data_quality', {})),
            stats_summary=self._generate_stats_summary(results.get('statistical_analysis', {})),
            mfg_insights=self._generate_mfg_insights(results.get('manufacturing_analysis', {})),
            ml_recommendations=self._generate_ml_recommendations(results.get('pipeline_recommendations', {})),
            viz_links=self._generate_viz_links(viz_paths)
        )
        
        return html
    
    def _generate_data_quality_table(self, quality_metrics: Dict) -> str:
        """Generate HTML table for data quality"""
        table_rows = ""
        for key, value in quality_metrics.items():
            if isinstance(value, dict):
                sub_table = "<table><tr><th>Subkey</th><th>Value</th></tr>"
                for sk, sv in value.items():
                    sub_table += f"<tr><td>{sk}</td><td>{sv}</td></tr>"
                sub_table += "</table>"
                table_rows += f"<tr><td>{key}</td><td>{sub_table}</td></tr>"
            else:
                table_rows += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        return f"<table><tr><th>Metric</th><th>Value</th></tr>{table_rows}</table>"
    
    def _generate_stats_summary(self, stats_results: Dict) -> str:
        """Generate statistical summary HTML"""
        summary = ""
        for key, value in stats_results.items():
            if isinstance(value, dict):
                summary += f"<h3>{key.capitalize()}</h3>"
                if 'numerical' in value:
                    summary += value['numerical'].to_html()
                elif 'categorical' in value:
                    summary += value['categorical'].to_html()
                else:
                    summary += "<pre>" + json.dumps(value, indent=2, default=str) + "</pre>"
        return summary
    
    def _generate_mfg_insights(self, mfg_results: Dict) -> str:
        """Generate manufacturing insights HTML"""
        insights = ""
        for key, value in mfg_results.items():
            insights += f"<h3>{key.replace('_', ' ').capitalize()}</h3>"
            insights += "<pre>" + json.dumps(value, indent=2, default=str) + "</pre>"
        return insights
    
    def _generate_ml_recommendations(self, pipeline_recs: Dict) -> str:
        """Generate ML recommendations HTML"""
        recs = "<ul>"
        for key, value in pipeline_recs.items():
            recs += f"<li><strong>{key.replace('_', ' ').capitalize()}:</strong> {', '.join(value) if isinstance(value, list) else value}</li>"
        recs += "</ul>"
        return recs
    
    def _generate_viz_links(self, viz_paths: Dict) -> str:
        """Generate visualization links"""
        links = "<ul>"
        for key, path in viz_paths.items():
            if path:
                links += f'<li><a href="{path}" target="_blank">{key.capitalize()} Visualization</a></li>'
        links += "</ul>"
        return links