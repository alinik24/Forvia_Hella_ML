"""
Scientific EDA Utilities Module - PHASE 1 REVISION
Supporting classes and functions for the main EDA pipeline

IMPROVEMENTS:
- Enhanced error handling and validation
- Better dependency management
- Memory optimization
- Robust logging
- Performance improvements
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

# Core statistical libraries
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, jarque_bera, normaltest

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
        return False

# Initialize optional dependencies
check_optional_import("plotly_express", "plotly.express as px", globals())
check_optional_import("plotly_graph", "plotly.graph_objects as go", globals())
check_optional_import("plotly_subplots", "plotly.subplots", globals())
check_optional_import("sklearn_decomp", "sklearn.decomposition", globals())
check_optional_import("sklearn_manifold", "sklearn.manifold", globals())
check_optional_import("sklearn_preprocessing", "sklearn.preprocessing", globals())
check_optional_import("sklearn_cluster", "sklearn.cluster", globals())
check_optional_import("sklearn_ensemble", "sklearn.ensemble", globals())
check_optional_import("sklearn_covariance", "sklearn.covariance", globals())
check_optional_import("sklearn_metrics", "sklearn.metrics", globals())
check_optional_import("missingno", "missingno as msno", globals())
check_optional_import("ydata_profiling", "ydata_profiling", globals())
check_optional_import("pyarrow", "pyarrow as pa", globals())
check_optional_import("pyarrow_parquet", "pyarrow.parquet as pq", globals())
check_optional_import("statsmodels_seasonal", "statsmodels.tsa.seasonal", globals())
check_optional_import("statsmodels_stattools", "statsmodels.tsa.stattools", globals())
check_optional_import("statsmodels_diagnostic", "statsmodels.stats.diagnostic", globals())

warnings.filterwarnings('ignore')


@dataclass
class EDAConfig:
    """Enhanced configuration for EDA pipeline with validation"""
    
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
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        try:
            # Validate numeric parameters
            if not (0 < self.max_memory_usage <= 1):
                raise ValueError("max_memory_usage must be between 0 and 1")
            
            if self.chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            
            if not (0 <= self.missing_threshold <= 1):
                raise ValueError("missing_threshold must be between 0 and 1")
            
            if not (0 <= self.correlation_threshold <= 1):
                raise ValueError("correlation_threshold must be between 0 and 1")
            
            if self.outlier_threshold <= 0:
                raise ValueError("outlier_threshold must be positive")
            
            # Validate cluster range
            if len(self.cluster_range) != 2 or self.cluster_range[0] >= self.cluster_range[1]:
                raise ValueError("cluster_range must be (min, max) with min < max")
            
            # Create output directory
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")


class DataProfiler:
    """Advanced data profiling with enhanced error handling"""
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self.setup_logging()
        self.results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging with error handling"""
        try:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Configure logging
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            log_file = Path(self.config.output_dir) / 'eda_analysis.log'
            
            # Remove existing handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file, mode='w'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("✅ Logging system initialized")
            
        except Exception as e:
            # Fallback to console logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"⚠️ Logging setup failed, using console only: {e}")
    
    def profile_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Profile file-level metadata with comprehensive error handling
        """
        self.logger.info(f"📋 Profiling file metadata: {file_path}")
        
        try:
            file_path = Path(file_path)
            
            # Basic validation
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Basic metadata
            file_stats = file_path.stat()
            metadata = {
                'file_size_mb': file_stats.st_size / (1024**2),
                'file_name': file_path.name,
                'file_extension': file_path.suffix.lower(),
                'file_size_bytes': file_stats.st_size,
                'created_time': datetime.fromtimestamp(file_stats.st_ctime),
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime)
            }
            
            # Detailed metadata for supported formats
            if metadata['file_extension'] == '.parquet' and OPTIONAL_LIBS.get('pyarrow_parquet', False):
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(file_path)
                    parquet_metadata = parquet_file.metadata
                    
                    metadata.update({
                        'num_row_groups': parquet_file.num_row_groups,
                        'schema_names': parquet_file.schema.names,
                        'num_columns': len(parquet_file.schema.names),
                        'total_rows_estimate': parquet_metadata.num_rows,
                        'parquet_version': parquet_metadata.format_version,
                        'compression': str(parquet_metadata.row_group(0).column(0).compression) if parquet_metadata.num_row_groups > 0 else 'unknown'
                    })
                    
                    self.logger.info(f"📊 Parquet file: {metadata['total_rows_estimate']:,} rows, "
                                   f"{metadata['num_columns']} columns")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read parquet metadata: {e}")
                    metadata['parquet_error'] = str(e)
            
            # Memory estimation and feasibility
            try:
                available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                estimated_memory_need = metadata['file_size_mb'] * 3 / 1024  # Rule of thumb: 3x file size
                
                metadata.update({
                    'memory_feasible': estimated_memory_need < (available_memory * self.config.max_memory_usage),
                    'estimated_memory_gb': estimated_memory_need,
                    'available_memory_gb': available_memory,
                    'memory_usage_percentage': (estimated_memory_need / available_memory) * 100
                })
                
                if metadata['memory_feasible']:
                    self.logger.info(f"💾 Memory check: ✅ Feasible ({estimated_memory_need:.2f}GB needed)")
                else:
                    self.logger.warning(f"💾 Memory check: ⚠️ Large file ({estimated_memory_need:.2f}GB needed, "
                                      f"{available_memory:.2f}GB available)")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Memory estimation failed: {e}")
                metadata['memory_feasible'] = True  # Default to feasible
                metadata['memory_error'] = str(e)
            
            self.results['file_metadata'] = metadata
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ File metadata profiling failed: {e}")
            raise


class DataLoader:
    """Intelligent data loading with enhanced memory management and error handling"""
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.supported_formats = {'.csv', '.parquet', '.xlsx', '.xls', '.json'}
        
    def load_data_smart(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """
        Smart data loading with comprehensive error handling and validation
        """
        self.logger.info("📂 Initiating smart data loading")
        
        try:
            # Validate input
            file_path = Path(file_path)
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. "
                               f"Supported formats: {self.supported_formats}")
            
            # Choose loading strategy based on memory feasibility
            if not metadata.get('memory_feasible', True):
                self.logger.warning("⚠️ Large file detected. Using chunked loading strategy.")
                df = self._load_chunked(file_path, metadata)
            else:
                df = self._load_full(file_path, metadata)
            
            # Post-loading validation
            df = self._validate_and_clean_dataframe(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Smart data loading failed: {e}")
            raise
    
    def _load_full(self, file_path: Path, metadata: Dict) -> pd.DataFrame:
        """Load full dataset with format-specific optimizations"""
        self.logger.info("📊 Loading full dataset")
        
        try:
            file_ext = file_path.suffix.lower()
            
            # Format-specific loading with optimizations
            if file_ext == '.parquet':
                df = self._load_parquet(file_path)
            elif file_ext == '.csv':
                df = self._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            elif file_ext == '.json':
                df = self._load_json(file_path)
            else:
                raise ValueError(f"Loading method not implemented for {file_ext}")
            
            # Memory optimization
            df = self._optimize_dtypes(df)
            
            # Log results
            memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
            self.logger.info(f"✅ Loaded {df.shape[0]:,} × {df.shape[1]} dataset ({memory_usage:.1f} MB)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Full dataset loading failed: {e}")
            raise
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load parquet file with error handling"""
        try:
            # Use pyarrow engine if available for better performance
            if OPTIONAL_LIBS.get('pyarrow', False):
                df = pd.read_parquet(file_path, engine='pyarrow')
            else:
                df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load parquet file: {e}")
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with intelligent parameter detection"""
        try:
            # Try to detect encoding and separator
            with open(file_path, 'rb') as f:
                sample = f.read(10000)
            
            # Simple encoding detection
            try:
                sample.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                try:
                    sample.decode('latin-1')
                    encoding = 'latin-1'
                except UnicodeDecodeError:
                    encoding = 'utf-8'  # Fallback
            
            # Load with detected parameters
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
                skipinitialspace=True
            )
            
            return df
            
        except Exception as e:
            # Fallback to basic loading
            self.logger.warning(f"⚠️ Advanced CSV loading failed, using basic method: {e}")
            return pd.read_csv(file_path)
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file with error handling"""
        try:
            # Try to load first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Check if multiple sheets exist and warn user
            try:
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) > 1:
                    self.logger.warning(f"⚠️ Excel file has {len(excel_file.sheet_names)} sheets. "
                                      f"Loading first sheet: {excel_file.sheet_names[0]}")
            except:
                pass
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file with error handling"""
        try:
            # Try different JSON orientations
            try:
                df = pd.read_json(file_path, orient='records')
            except:
                try:
                    df = pd.read_json(file_path, orient='index')
                except:
                    df = pd.read_json(file_path)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")
    
    def _load_chunked(self, file_path: Path, metadata: Dict) -> pd.DataFrame:
        """Load data in chunks for large files with enhanced error handling"""
        self.logger.info(f"📦 Loading data in chunks of {self.config.chunk_size:,} rows")
        
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.parquet':
                # For parquet, read a representative sample
                df_sample = self._load_parquet(file_path)
                
                # If still too large, take a random sample
                if len(df_sample) > self.config.chunk_size * 10:
                    sample_size = min(self.config.chunk_size * 10, len(df_sample))
                    df_sample = df_sample.sample(
                        n=sample_size, 
                        random_state=self.config.random_state
                    ).reset_index(drop=True)
                    
                    self.logger.info(f"📊 Sampled {sample_size:,} rows from large parquet file")
                
                return df_sample
                
            elif file_ext == '.csv':
                # For CSV, read in actual chunks
                chunk_list = []
                total_chunks = 0
                max_chunks = 10
                
                try:
                    chunk_iterator = pd.read_csv(file_path, chunksize=self.config.chunk_size)
                    
                    for chunk in chunk_iterator:
                        chunk_list.append(chunk)
                        total_chunks += 1
                        
                        if total_chunks >= max_chunks:
                            self.logger.info(f"📦 Loaded {max_chunks} chunks for analysis")
                            break
                    
                    if chunk_list:
                        df_sample = pd.concat(chunk_list, ignore_index=True)
                    else:
                        raise ValueError("No chunks could be loaded")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Chunked CSV loading failed: {e}")
                    # Fallback: try to load full file and then sample
                    df_full = self._load_csv(file_path)
                    sample_size = min(self.config.chunk_size * 10, len(df_full))
                    df_sample = df_full.sample(n=sample_size, random_state=self.config.random_state)
                
                return df_sample
                
            else:
                # For other formats, load full and then sample if too large
                self.logger.warning(f"⚠️ Chunked loading not supported for {file_ext}, "
                                  f"loading full file and sampling")
                df_full = self._load_full(file_path, metadata)
                
                if len(df_full) > self.config.chunk_size * 10:
                    sample_size = self.config.chunk_size * 10
                    df_sample = df_full.sample(n=sample_size, random_state=self.config.random_state)
                    return df_sample
                
                return df_full
                
        except Exception as e:
            self.logger.error(f"❌ Chunked loading failed: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency with error handling"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)
            
            # Optimize numeric types
            for col in df.columns:
                try:
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
                    elif pd.api.types.is_float_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')
                except:
                    # Skip columns that can't be optimized
                    continue
            
            # Convert object columns to category if low cardinality
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        df[col] = df[col].astype('category')
                except:
                    continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0.1:  # Only log if significant savings
                self.logger.info(f"💾 Memory optimized: {memory_saved:.1f} MB saved "
                               f"({(memory_saved/original_memory)*100:.1f}% reduction)")
            
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return df
    
    def _validate_and_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded dataframe"""
        try:
            # Basic validation
            if df.empty:
                raise ValueError("Loaded dataframe is empty")
            
            if df.shape[0] == 0:
                raise ValueError("Dataframe has no rows")
            
            if df.shape[1] == 0:
                raise ValueError("Dataframe has no columns")
            
            # Clean column names
            original_columns = df.columns.tolist()
            df.columns = df.columns.astype(str)  # Ensure string column names
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Remove completely empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                df = df.drop(columns=empty_cols)
                self.logger.warning(f"⚠️ Removed {len(empty_cols)} completely empty columns")
            
            # Log basic info
            self.logger.info(f"📊 Dataset validated: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Check for potential issues
            if df.shape[1] > 1000:
                self.logger.warning(f"⚠️ Very wide dataset ({df.shape[1]} columns)")
            
            if df.shape[0] < 10:
                self.logger.warning(f"⚠️ Very small dataset ({df.shape[0]} rows)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Dataframe validation failed: {e}")
            raise


class TimeSeriesDetector:
    """
    Enhanced time series detection and analysis with improved error handling
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def detect_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze temporal patterns with comprehensive error handling"""
        self.logger.info("🕐 Detecting temporal features")
        
        temporal_info = {
            'datetime_columns': [],
            'potential_timestamps': [],
            'time_series_detected': False,
            'temporal_patterns': {},
            'seasonality_detected': False,
            'trend_detected': False,
            'analysis_errors': []
        }
        
        try:
            # Detect explicit datetime columns
            for col in df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        temporal_info['datetime_columns'].append(col)
                        self.logger.info(f"📅 Found datetime column: {col}")
                        continue
                except:
                    continue
                    
                # Try to parse object columns as datetime
                if df[col].dtype == 'object':
                    try:
                        sample_values = df[col].dropna().head(100)
                        if len(sample_values) == 0:
                            continue
                            
                        parsed_count = 0
                        for fmt in self.config.datetime_formats:
                            try:
                                if fmt == 'ISO8601':
                                    test_parse = pd.to_datetime(sample_values.iloc[:10], errors='coerce')
                                else:
                                    test_parse = pd.to_datetime(sample_values.iloc[:10], format=fmt, errors='coerce')
                                
                                if test_parse.notna().sum() > 5:  # At least 5 successful parses
                                    parsed_count += 1
                                    temporal_info['potential_timestamps'].append(col)
                                    self.logger.info(f"📅 Found potential timestamp column: {col} (format: {fmt})")
                                    break
                                    
                            except:
                                continue
                                
                    except Exception as e:
                        temporal_info['analysis_errors'].append(f"Error parsing {col}: {str(e)}")
                        continue
            
            # Analyze temporal patterns if found
            all_temporal_cols = temporal_info['datetime_columns'] + temporal_info['potential_timestamps']
            
            if all_temporal_cols:
                temporal_info['time_series_detected'] = True
                self.logger.info(f"✅ Time series detected with {len(all_temporal_cols)} temporal columns")
                
                try:
                    temporal_info['temporal_patterns'] = self._analyze_temporal_patterns(df, all_temporal_cols)
                except Exception as e:
                    self.logger.warning(f"⚠️ Temporal pattern analysis failed: {e}")
                    temporal_info['analysis_errors'].append(f"Pattern analysis failed: {str(e)}")
                
                # Advanced time series analysis if statsmodels available
                if OPTIONAL_LIBS.get('statsmodels_seasonal', False):
                    try:
                        advanced_analysis = self._advanced_time_series_analysis(df, all_temporal_cols)
                        temporal_info.update(advanced_analysis)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Advanced time series analysis failed: {e}")
                        temporal_info['analysis_errors'].append(f"Advanced analysis failed: {str(e)}")
            else:
                self.logger.info("ℹ️ No temporal features detected")
            
        except Exception as e:
            self.logger.error(f"❌ Temporal feature detection failed: {e}")
            temporal_info['analysis_errors'].append(f"Detection failed: {str(e)}")
        
        return temporal_info
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
        """Analyze patterns in temporal data with enhanced error handling and progress tracking"""
        patterns = {}
        total_cols = len(temporal_cols)
        
        self.logger.info(f"🔍 Analyzing {total_cols} temporal columns...")
        
        for idx, col in enumerate(temporal_cols):
            try:
                self.logger.info(f"📅 Processing temporal column {idx+1}/{total_cols}: {col}")
                
                # More aggressive sampling for large datasets
                if len(df) > 100000:
                    sample_size = min(5000, len(df))
                    self.logger.info(f"   📊 Sampling {sample_size:,} rows from {len(df):,} for analysis")
                    sample_df = df[col].sample(n=sample_size, random_state=self.config.random_state)
                elif len(df) > 10000:
                    sample_size = min(10000, len(df))
                    sample_df = df[col].sample(n=sample_size, random_state=self.config.random_state)
                else:
                    sample_df = df[col]
                
                # Convert to datetime with timeout protection
                self.logger.info(f"   🔄 Converting to datetime...")
                if not pd.api.types.is_datetime64_any_dtype(sample_df):
                    # Use faster conversion method
                    dt_series = pd.to_datetime(sample_df, errors='coerce', cache=True)
                else:
                    dt_series = sample_df
                
                # Clean and validate
                dt_series = dt_series.dropna()
                
                if len(dt_series) < 10:
                    patterns[col] = {'error': 'Insufficient valid datetime values'}
                    self.logger.warning(f"   ⚠️ Insufficient data for {col}")
                    continue
                
                self.logger.info(f"   📈 Calculating temporal statistics...")
                
                # Calculate basic temporal statistics (optimized)
                start_date = dt_series.min()
                end_date = dt_series.max()
                date_range = end_date - start_date
                
                # Quick pattern analysis
                patterns[col] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'date_range_days': date_range.days if date_range else None,
                    'total_points': len(df[col].dropna()),
                    'sample_size': len(dt_series),
                    'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
                }
                
                # Add detailed analysis only for reasonable sized samples
                if len(dt_series) <= 1000:
                    patterns[col].update({
                        'frequency_estimate': self._estimate_frequency(dt_series),
                        'has_gaps': self._detect_gaps(dt_series),
                        'business_hours_pattern': self._detect_business_hours(dt_series),
                        'regularity_score': self._calculate_regularity_score(dt_series)
                    })
                else:
                    # Use simplified analysis for large samples
                    patterns[col].update({
                        'frequency_estimate': 'large_dataset_detected',
                        'has_gaps': 'not_analyzed_due_to_size',
                        'business_hours_pattern': {'analysis': 'skipped_for_performance'},
                        'regularity_score': 0.0
                    })
                
                self.logger.info(f"   ✅ Completed analysis for {col}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not analyze temporal patterns for {col}: {e}")
                patterns[col] = {'error': str(e)}
        
        self.logger.info(f"✅ Temporal pattern analysis completed for {len(patterns)} columns")
        return patterns
    
    def _estimate_frequency(self, dt_series: pd.Series) -> str:
        """Estimate frequency with improved logic"""
        try:
            if len(dt_series) < 3:
                return "insufficient_data"
            
            # Sort and calculate differences
            sorted_dates = dt_series.sort_values()
            diffs = sorted_dates.diff().dropna()
            
            if len(diffs) == 0:
                return "no_differences"
            
            # Get most common difference (mode)
            try:
                most_common_diff = diffs.mode().iloc[0]
                diff_seconds = most_common_diff.total_seconds()
                
                # Classify frequency
                if diff_seconds < 60:
                    return "seconds"
                elif diff_seconds < 3600:
                    return "minutes"
                elif diff_seconds < 86400:
                    return "hours"
                elif diff_seconds < 604800:
                    return "days"
                elif diff_seconds < 2629746:  # ~30.44 days
                    return "weeks"
                else:
                    return "months_or_more"
                    
            except Exception:
                return "irregular"
                
        except Exception as e:
            return f"error: {str(e)}"
    
    def _detect_gaps(self, dt_series: pd.Series) -> bool:
        """Detect gaps in time series with vectorized operations and timeout protection"""
        try:
            if len(dt_series) < 3:
                return False
            
            # Further limit sample size for gap detection
            if len(dt_series) > 1000:
                dt_series = dt_series.sample(n=1000, random_state=42)
            
            # Ensure datetime and sort
            if not pd.api.types.is_datetime64_any_dtype(dt_series):
                dt_series = pd.to_datetime(dt_series, errors='coerce')
            
            dates_arr = dt_series.dropna().sort_values()
            if len(dates_arr) < 3:
                return False
            
            # Use pandas diff for better performance
            diffs = dates_arr.diff().dropna()
            
            if len(diffs) == 0:
                return False
            
            # Simple gap detection - check if any gap is > 3x median
            median_diff = diffs.median()
            return bool((diffs > (median_diff * 3)).any())
            
        except Exception:
            return False
    
    def _detect_business_hours(self, dt_series: pd.Series) -> Dict:
        """Detect business hours pattern with error handling"""
        try:
            if len(dt_series) == 0:
                return {'error': 'Empty series'}
                
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
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_regularity_score(self, dt_series: pd.Series) -> float:
        """Calculate regularity score with improved robustness"""
        try:
            if len(dt_series) < 3:
                return 0.0
            
            sorted_dates = dt_series.sort_values()
            diffs = sorted_dates.diff().dropna()
            
            if len(diffs) == 0 or diffs.mean() == 0:
                return 0.0
            
            # Calculate coefficient of variation
            cv = diffs.std() / diffs.mean()
            
            # Convert to regularity score (0-100, higher = more regular)
            regularity_score = max(0, 100 - (cv * 100))
            
            return float(min(100, regularity_score))
            
        except Exception:
            return 0.0
    
    def _advanced_time_series_analysis(self, df: pd.DataFrame, temporal_cols: List[str]) -> Dict:
        """Advanced time series analysis using statsmodels"""
        analysis = {
            'seasonality_detected': False, 
            'trend_detected': False, 
            'stationarity_test': {},
            'analysis_errors': []
        }
        
        try:
            if not OPTIONAL_LIBS.get('statsmodels_stattools', False):
                analysis['analysis_errors'].append("Statsmodels not available")
                return analysis
                
            # Import required functions
            from statsmodels.tsa.stattools import adfuller, acf
            
            # Use first temporal column for detailed analysis
            time_col = temporal_cols[0]
            
            # Convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Find numeric columns for time series analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Limit for performance
            
            for num_col in numeric_cols:
                try:
                    # Create time series
                    ts_data = df[[time_col, num_col]].dropna().sort_values(time_col)
                    if len(ts_data) < 50:  # Need sufficient data
                        continue
                    
                    ts_data = ts_data.set_index(time_col)
                    
                    # Stationarity test (Augmented Dickey-Fuller)
                    try:
                        adf_result = adfuller(ts_data[num_col].values)
                        analysis['stationarity_test'][num_col] = {
                            'adf_statistic': float(adf_result[0]),
                            'p_value': float(adf_result[1]),
                            'is_stationary': adf_result[1] < 0.05
                        }
                    except Exception as e:
                        analysis['analysis_errors'].append(f"ADF test failed for {num_col}: {str(e)}")
                    
                    # Seasonality detection (autocorrelation check)
                    try:
                        autocorr = acf(ts_data[num_col].values, nlags=min(40, len(ts_data)//4), fft=False)
                        if np.any(np.abs(autocorr[12:]) > 0.3):  # Check for seasonal patterns
                            analysis['seasonality_detected'] = True
                    except Exception as e:
                        analysis['analysis_errors'].append(f"Seasonality test failed for {num_col}: {str(e)}")
                    
                    # Trend detection (linear regression slope)
                    try:
                        x = np.arange(len(ts_data))
                        slope, _, r_value, p_value, _ = stats.linregress(x, ts_data[num_col].values)
                        if abs(r_value) > 0.3 and p_value < 0.05:
                            analysis['trend_detected'] = True
                    except Exception as e:
                        analysis['analysis_errors'].append(f"Trend test failed for {num_col}: {str(e)}")
                    
                    break  # Only analyze first valid numeric column for performance
                    
                except Exception as e:
                    analysis['analysis_errors'].append(f"Analysis failed for {num_col}: {str(e)}")
                    continue
                    
        except Exception as e:
            analysis['analysis_errors'].append(f"Advanced analysis failed: {str(e)}")
        
        return analysis


class StatisticalAnalyzer:
    """
    Enhanced statistical analysis with improved error handling and performance
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def comprehensive_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis with enhanced error handling and progress tracking"""
        self.logger.info("📈 Starting comprehensive statistical analysis")
        
        analysis_results = {
            'basic_statistics': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'outlier_analysis': {},
            'missing_value_analysis': {},
            'feature_importance': {},
            'analysis_errors': []
        }
        
        # Track progress
        analysis_steps = [
            ('basic_statistics', 'Basic Statistics'),
            ('distribution_analysis', 'Distribution Analysis'), 
            ('correlation_analysis', 'Correlation Analysis'),
            ('outlier_analysis', 'Outlier Analysis'),
            ('missing_value_analysis', 'Missing Value Analysis'),
            ('feature_importance', 'Feature Importance')
        ]
        
        total_steps = len(analysis_steps)
        
        for idx, (key, name) in enumerate(analysis_steps):
            try:
                self.logger.info(f"📊 Step {idx+1}/{total_steps}: {name}")
                
                if key == 'basic_statistics':
                    analysis_results[key] = self._enhanced_basic_statistics(df)
                elif key == 'distribution_analysis':
                    analysis_results[key] = self._enhanced_distribution_analysis(df)
                elif key == 'correlation_analysis':
                    analysis_results[key] = self._enhanced_correlation_analysis(df)
                elif key == 'outlier_analysis':
                    analysis_results[key] = self._robust_outlier_analysis(df)
                elif key == 'missing_value_analysis':
                    analysis_results[key] = self._advanced_missing_analysis(df)
                elif key == 'feature_importance':
                    analysis_results[key] = self._calculate_feature_importance(df)
                    
                self.logger.info(f"✅ {name} completed")
                
            except Exception as e:
                self.logger.warning(f"⚠️ {name} failed: {e}")
                analysis_results['analysis_errors'].append(f"{name}: {str(e)}")
        
        self.logger.info("✅ Statistical analysis completed")
        return analysis_results
    
    def _enhanced_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced basic statistics with error handling"""
        stats_dict = {}
        
        try:
            # Numerical features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Basic statistics
                numeric_stats = df[numeric_cols].describe()
                
                # Enhanced statistics with error handling
                additional_stats = {}
                for col in numeric_cols:
                    try:
                        data = df[col].dropna()
                        if len(data) > 0:
                            additional_stats[col] = {
                                'skewness': float(data.skew()),
                                'kurtosis': float(data.kurtosis()),
                                'excess_kurtosis': float(data.kurtosis() - 3),
                                'missing_count': int(df[col].isnull().sum()),
                                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                                'unique_count': int(df[col].nunique()),
                                'zero_count': int((df[col] == 0).sum()),
                                'negative_count': int((df[col] < 0).sum()),
                                'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else float('inf'),
                                'mad': float(stats.median_abs_deviation(data)),
                                'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                            }
                    except Exception as e:
                        additional_stats[col] = {'error': str(e)}
                
                stats_dict['numerical'] = {
                    'basic_stats': numeric_stats.to_dict(),
                    'enhanced_stats': additional_stats
                }
            
            # Categorical features
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_stats = {}
                
                for col in categorical_cols:
                    try:
                        value_counts = df[col].value_counts()
                        cat_stats[col] = {
                            'count': int(df[col].count()),
                            'unique': int(df[col].nunique()),
                            'top_value': str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                            'top_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            'top_freq_pct': float((value_counts.iloc[0] / len(df)) * 100) if len(value_counts) > 0 else 0,
                            'missing_count': int(df[col].isnull().sum()),
                            'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                            'entropy': self._calculate_entropy(value_counts),
                            'concentration_ratio': self._calculate_concentration_ratio(value_counts)
                        }
                    except Exception as e:
                        cat_stats[col] = {'error': str(e)}
                
                stats_dict['categorical'] = cat_stats
            
        except Exception as e:
            stats_dict['error'] = str(e)
        
        return stats_dict
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate Shannon entropy with error handling"""
        try:
            if len(value_counts) == 0:
                return 0.0
            
            probabilities = value_counts / value_counts.sum()
            # Add small epsilon to avoid log(0)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_concentration_ratio(self, value_counts: pd.Series) -> float:
        """Calculate concentration ratio with error handling"""
        try:
            if len(value_counts) == 0:
                return 0.0
            return float((value_counts.iloc[0] / value_counts.sum()) * 100)
        except:
            return 0.0
    
    def _enhanced_distribution_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced distribution analysis with multiple normality tests and progress tracking"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_results = {}
        
        if len(numeric_cols) == 0:
            return {'message': 'No numeric columns found'}
        
        # Limit columns for performance on large datasets
        if len(numeric_cols) > 20:
            self.logger.info(f"   📊 Limiting analysis to first 20 numeric columns (found {len(numeric_cols)})")
            numeric_cols = numeric_cols[:20]
        
        total_cols = len(numeric_cols)
        self.logger.info(f"   📊 Analyzing {total_cols} numeric columns...")
        
        for idx, col in enumerate(numeric_cols):
            try:
                if idx % 5 == 0:  # Progress update every 5 columns
                    self.logger.info(f"   📈 Progress: {idx+1}/{total_cols} columns")
                
                data = df[col].dropna()
                
                # Skip if insufficient data or all same values
                if len(data) < 10 or data.nunique() <= 1:
                    distribution_results[col] = {
                        'error': f'Insufficient variation (unique values: {data.nunique()}, count: {len(data)})'
                    }
                    continue
                
                # Sample large datasets for distribution analysis
                if len(data) > 10000:
                    data = data.sample(n=10000, random_state=self.config.random_state)
                
                results = {
                    'normality_tests': self._comprehensive_normality_tests(data),
                    'distribution_params': self._enhanced_distribution_params(data),
                    'distribution_recommendation': self._recommend_distribution(data)
                }
                distribution_results[col] = results
                
            except Exception as e:
                distribution_results[col] = {'error': str(e)}
        
        self.logger.info(f"   ✅ Distribution analysis completed for {len(distribution_results)} columns")
        return distribution_results
    
    def _comprehensive_normality_tests(self, data: pd.Series) -> Dict:
        """Comprehensive normality testing with multiple methods"""
        results = {}
        
        try:
            if len(data) < 3:
                return {'error': 'Insufficient data for normality tests'}
            
            # Shapiro-Wilk test (best for n < 5000)
            if len(data) <= 5000:
                try:
                    stat, p_value = shapiro(data)
                    results['shapiro_wilk'] = {
                        'statistic': float(stat), 
                        'p_value': float(p_value), 
                        'is_normal': p_value > 0.05
                    }
                except Exception as e:
                    results['shapiro_wilk'] = {'error': str(e)}
            
            # Jarque-Bera test (good for large samples)
            try:
                stat, p_value = jarque_bera(data)
                results['jarque_bera'] = {
                    'statistic': float(stat), 
                    'p_value': float(p_value), 
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                results['jarque_bera'] = {'error': str(e)}
            
            # D'Agostino's normality test
            try:
                stat, p_value = normaltest(data)
                results['dagostino'] = {
                    'statistic': float(stat), 
                    'p_value': float(p_value), 
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                results['dagostino'] = {'error': str(e)}
            
            # Anderson-Darling test
            try:
                result = anderson(data, dist='norm')
                results['anderson_darling'] = {
                    'statistic': float(result.statistic),
                    'critical_values': [float(x) for x in result.critical_values],
                    'significance_levels': [float(x) for x in result.significance_level],
                    'is_normal': result.statistic < result.critical_values[2]  # 5% significance
                }
            except Exception as e:
                results['anderson_darling'] = {'error': str(e)}
            
            # Kolmogorov-Smirnov test
            try:
                stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
                results['kolmogorov_smirnov'] = {
                    'statistic': float(stat), 
                    'p_value': float(p_value), 
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                results['kolmogorov_smirnov'] = {'error': str(e)}
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _enhanced_distribution_params(self, data: pd.Series) -> Dict:
        """Enhanced distribution parameters with robust statistics"""
        params = {}
        
        try:
            # Basic parameters
            params['mean'] = float(data.mean())
            params['std'] = float(data.std())
            params['median'] = float(data.median())
            params['mode'] = float(data.mode().iloc[0]) if not data.mode().empty else None
            
            # Robust statistics
            params['mad'] = float(stats.median_abs_deviation(data))
            params['iqr'] = float(data.quantile(0.75) - data.quantile(0.25))
            params['trimmed_mean'] = float(stats.trim_mean(data, 0.1))
            
            # Distribution shape
            params['skewness'] = float(data.skew())
            params['kurtosis'] = float(data.kurtosis())
            params['excess_kurtosis'] = float(data.kurtosis() - 3)
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                params[f'percentile_{p}'] = float(data.quantile(p/100))
                
        except Exception as e:
            params['error'] = str(e)
        
        return params
    
    def _recommend_distribution(self, data: pd.Series) -> str:
        """Recommend appropriate distribution based on characteristics"""
        try:
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
        except:
            return "analysis_failed"
    
    def _enhanced_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced correlation analysis with multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'warning': 'Insufficient numeric columns for correlation analysis'}
        
        results = {}
        
        try:
            # Pearson correlation (linear relationships)
            pearson_corr = df[numeric_cols].corr(method='pearson')
            results['pearson'] = pearson_corr.to_dict()
            
            # Spearman correlation (monotonic relationships)
            spearman_corr = df[numeric_cols].corr(method='spearman')
            results['spearman'] = spearman_corr.to_dict()
            
            # Kendall correlation (robust to outliers)
            kendall_corr = df[numeric_cols].corr(method='kendall')
            results['kendall'] = kendall_corr.to_dict()
            
            # High correlation pairs
            high_corr_pairs = self._find_high_correlation_pairs(pearson_corr)
            results['high_correlation_pairs'] = high_corr_pairs
            
            # Correlation stability analysis
            results['correlation_stability'] = self._analyze_correlation_stability(
                pearson_corr, spearman_corr, kendall_corr
            )
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value) and abs(corr_value) > self.config.correlation_threshold:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'abs_correlation': float(abs(corr_value))
                        })
            
            return sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
        except:
            return []
    
    def _analyze_correlation_stability(self, pearson: pd.DataFrame, spearman: pd.DataFrame, 
                                     kendall: pd.DataFrame) -> Dict:
        """Analyze stability of correlations across different methods"""
        stability = {}
        
        try:
            for col1 in pearson.columns:
                for col2 in pearson.columns:
                    if col1 != col2:
                        p_corr = pearson.loc[col1, col2]
                        s_corr = spearman.loc[col1, col2]
                        k_corr = kendall.loc[col1, col2]
                        
                        # Skip if any correlation is NaN
                        if any(np.isnan([p_corr, s_corr, k_corr])):
                            continue
                        
                        # Calculate stability as standard deviation of correlations
                        corr_std = np.std([p_corr, s_corr, k_corr])
                        
                        if abs(p_corr) > 0.3:  # Only analyze meaningful correlations
                            stability[f"{col1}-{col2}"] = {
                                'pearson': float(p_corr),
                                'spearman': float(s_corr),
                                'kendall': float(k_corr),
                                'stability_score': float(1 - corr_std),  # Higher score = more stable
                                'relationship_type': self._classify_relationship(p_corr, s_corr)
                            }
        except Exception as e:
            stability['error'] = str(e)
        
        return stability
    
    def _classify_relationship(self, pearson: float, spearman: float) -> str:
        """Classify the type of relationship based on correlation coefficients"""
        try:
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
        except:
            return "unknown"
    
    def _robust_outlier_analysis(self, df: pd.DataFrame) -> Dict:
        """Robust multi-method outlier detection"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_cols:
            try:
                data = df[col].dropna()
                if len(data) > 10 and data.nunique() > 1:
                    outlier_results[col] = {
                        'iqr_outliers': self._detect_outliers_iqr(data),
                        'zscore_outliers': self._detect_outliers_zscore(data),
                        'modified_zscore_outliers': self._detect_outliers_modified_zscore(data),
                        'outlier_consensus': self._calculate_outlier_consensus(data)
                    }
                    
                    # Add sklearn-based methods if available
                    if OPTIONAL_LIBS.get('sklearn_ensemble', False):
                        try:
                            from sklearn.ensemble import IsolationForest
                            outlier_results[col]['isolation_forest_outliers'] = self._detect_outliers_isolation_forest(data)
                        except:
                            pass
                    
                    if OPTIONAL_LIBS.get('sklearn_covariance', False):
                        try:
                            from sklearn.covariance import EllipticEnvelope
                            outlier_results[col]['elliptic_envelope_outliers'] = self._detect_outliers_elliptic_envelope(data)
                        except:
                            pass
                            
            except Exception as e:
                outlier_results[col] = {'error': str(e)}
        
        return outlier_results
    
    def _detect_outliers_iqr(self, data: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > self.config.outlier_threshold
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_modified_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Modified Z-score (more robust)"""
        try:
            median = np.median(data)
            mad = stats.median_abs_deviation(data)
            
            if mad == 0:
                return []
            
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > 3.5
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_isolation_forest(self, data: pd.Series) -> List[int]:
        """Detect outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _detect_outliers_elliptic_envelope(self, data: pd.Series) -> List[int]:
        """Detect outliers using Elliptic Envelope"""
        try:
            from sklearn.covariance import EllipticEnvelope
            clf = EllipticEnvelope(contamination=0.1, random_state=self.config.random_state)
            outlier_labels = clf.fit_predict(data.values.reshape(-1, 1))
            outlier_mask = outlier_labels == -1
            return data[outlier_mask].index.tolist()
        except:
            return []
    
    def _calculate_outlier_consensus(self, data: pd.Series) -> Dict:
        """Calculate consensus outliers across multiple methods"""
        try:
            methods = {
                'iqr': set(self._detect_outliers_iqr(data)),
                'zscore': set(self._detect_outliers_zscore(data)),
                'modified_zscore': set(self._detect_outliers_modified_zscore(data))
            }
            
            # Add sklearn methods if available
            if OPTIONAL_LIBS.get('sklearn_ensemble', False):
                methods['isolation_forest'] = set(self._detect_outliers_isolation_forest(data))
            
            if OPTIONAL_LIBS.get('sklearn_covariance', False):
                methods['elliptic_envelope'] = set(self._detect_outliers_elliptic_envelope(data))
            
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
        except:
            return {'error': 'Consensus calculation failed'}
    
    def _advanced_missing_analysis(self, df: pd.DataFrame) -> Dict:
        """Advanced missing value analysis with pattern detection"""
        missing_info = {}
        
        try:
            # Basic missing value statistics
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            missing_info['summary'] = {
                'missing_count': missing_counts.to_dict(),
                'missing_percentage': missing_percentages.to_dict()
            }
            
            # Missing value patterns and mechanisms
            missing_info['patterns'] = self._analyze_missing_patterns(df)
            missing_info['mechanisms'] = self._analyze_missing_mechanisms(df)
            
        except Exception as e:
            missing_info['error'] = str(e)
        
        return missing_info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in missing data"""
        try:
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
                            if not np.isnan(corr_value) and abs(corr_value) > 0.5:
                                high_missing_corr.append({
                                    'col1': missing_corr.columns[i],
                                    'col2': missing_corr.columns[j],
                                    'correlation': float(corr_value)
                                })
                    
                    patterns['missing_correlations'] = high_missing_corr
            
            return patterns
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_missing_mechanisms(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data mechanisms (MCAR, MAR, MNAR)"""
        mechanisms = {}
        
        try:
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            
            for col in cols_with_missing:
                try:
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
                                        'correlation': float(corr)
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
                except Exception as e:
                    mechanisms[col] = {'error': str(e)}
            
        except Exception as e:
            mechanisms['error'] = str(e)
        
        return mechanisms
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict:
        """Calculate feature importance using multiple methods"""
        importance_results = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {'message': 'Insufficient numeric columns for feature importance analysis'}
            
            # Use variance as a simple importance measure
            variances = df[numeric_cols].var().sort_values(ascending=False)
            importance_results['variance_importance'] = variances.to_dict()
            
            # Use coefficient of variation for normalized importance
            cv_importance = (df[numeric_cols].std() / df[numeric_cols].mean()).sort_values(ascending=False)
            importance_results['cv_importance'] = cv_importance.dropna().to_dict()
            
        except Exception as e:
            importance_results['error'] = str(e)
        
        return importance_results


class VisualizationEngine:
    """
    Basic visualization engine with error handling - Phase 1 implementation
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def create_comprehensive_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, str]:
        """Create basic visualizations with error handling"""
        self.logger.info("📊 Creating visualizations")
        
        viz_paths = {}
        
        try:
            # 1. Basic overview plots
            viz_paths['overview'] = self._create_basic_overview(df)
            
            # 2. Distribution plots
            viz_paths['distributions'] = self._create_distribution_plots(df)
            
            # 3. Correlation plots (if plotly available)
            if OPTIONAL_LIBS.get('plotly_express', False):
                viz_paths['correlations'] = self._create_correlation_plots(df)
            
            # 4. Missing value plots
            viz_paths['missing_values'] = self._create_missing_plots(df)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some visualizations failed: {e}")
        
        return viz_paths
    
    def _create_basic_overview(self, df: pd.DataFrame) -> str:
        """Create basic overview plots using matplotlib"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            
            # Data types
            dtype_counts = df.dtypes.value_counts()
            axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Data Types Distribution')
            
            # Missing values
            missing_counts = df.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            if len(missing_cols) > 0:
                missing_cols.head(10).plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Missing Values by Column')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
                axes[0, 1].set_title('Missing Values')
            
            # Numeric features distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                axes[1, 0].hist(len(numeric_cols), bins=20)
                axes[1, 0].set_title(f'Numeric Features: {len(numeric_cols)}')
            
            # Categorical features
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                axes[1, 1].hist(len(cat_cols), bins=20)
                axes[1, 1].set_title(f'Categorical Features: {len(cat_cols)}')
            
            plt.tight_layout()
            
            output_path = self.output_dir / "basic_overview.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Basic overview saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Basic overview creation failed: {e}")
            return ""
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> str:
        """Create distribution plots for numeric features"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns for distribution plots")
                return ""
            
            # Limit columns for performance
            cols_to_plot = list(numeric_cols)[:9]  # Max 9 for 3x3 grid
            
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, col in enumerate(cols_to_plot):
                row = idx // n_cols
                col_pos = idx % n_cols
                
                ax = axes[row, col_pos] if n_rows > 1 else axes[col_pos]
                
                data = df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{col}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
            
            # Hide empty subplots
            for idx in range(len(cols_to_plot), n_rows * n_cols):
                row = idx // n_cols
                col_pos = idx % n_cols
                ax = axes[row, col_pos] if n_rows > 1 else axes[col_pos]
                ax.set_visible(False)
            
            plt.tight_layout()
            
            output_path = self.output_dir / "distribution_plots.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Distribution plots saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Distribution plots creation failed: {e}")
            return ""
    
    def _create_correlation_plots(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap using plotly if available"""
        try:
            if not OPTIONAL_LIBS.get('plotly_graph', False):
                return self._create_correlation_matplotlib(df)
            
            import plotly.graph_objects as go
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                self.logger.warning("Insufficient numeric columns for correlation plot")
                return ""
            
            # Limit size for visualization
            if len(numeric_cols) > 25:
                numeric_cols = numeric_cols[:25]
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                width=800,
                height=800
            )
            
            output_path = self.output_dir / "correlation_matrix.html"
            fig.write_html(output_path)
            
            self.logger.info(f"📊 Correlation plot saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation plot creation failed: {e}")
            return ""
    
    def _create_correlation_matplotlib(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap using matplotlib as fallback"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return ""
            
            # Limit size
            if len(numeric_cols) > 15:
                numeric_cols = numeric_cols[:15]
            
            corr_matrix = df[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            output_path = self.output_dir / "correlation_matrix.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matplotlib correlation plot failed: {e}")
            return ""
    
    def _create_missing_plots(self, df: pd.DataFrame) -> str:
        """Create missing value visualization"""
        try:
            missing_counts = df.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            
            if len(cols_with_missing) == 0:
                self.logger.info("No missing values to visualize")
                return ""
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Missing counts
            cols_with_missing.head(15).plot(kind='bar', ax=axes[0])
            axes[0].set_title('Missing Value Counts')
            axes[0].set_xlabel('Columns')
            axes[0].set_ylabel('Missing Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Missing percentages
            missing_pct = (cols_with_missing / len(df)) * 100
            missing_pct.head(15).plot(kind='bar', ax=axes[1], color='red', alpha=0.7)
            axes[1].set_title('Missing Value Percentages')
            axes[1].set_xlabel('Columns')
            axes[1].set_ylabel('Missing %')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            output_path = self.output_dir / "missing_values.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Missing value plots saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Missing value plots creation failed: {e}")
            return ""


class DimensionalityAnalyzer:
    """
    Basic dimensionality analysis - Phase 1 implementation
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_dimensionality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic dimensionality analysis with error handling"""
        self.logger.info("📐 Analyzing data dimensionality")
        
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 3:
            return {'warning': 'Insufficient numeric features for dimensionality analysis'}
        
        results = {
            'basic_analysis': self._basic_dimensionality_analysis(numeric_df),
            'analysis_errors': []
        }
        
        # Add PCA if sklearn available
        if OPTIONAL_LIBS.get('sklearn_decomp', False):
            try:
                results['pca_analysis'] = self._basic_pca_analysis(numeric_df)
            except Exception as e:
                results['analysis_errors'].append(f"PCA analysis failed: {str(e)}")
        
        return results
    
    def _basic_dimensionality_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic dimensionality metrics"""
        try:
            return {
                'n_features': df.shape[1],
                'n_samples': df.shape[0],
                'feature_to_sample_ratio': df.shape[1] / df.shape[0],
                'curse_of_dimensionality_risk': 'high' if df.shape[1] / df.shape[0] > 0.1 else 'low'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _basic_pca_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic PCA analysis if sklearn available"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Perform PCA
            n_components = min(self.config.pca_components, df.shape[1])
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            pca.fit(df_scaled)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            return {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'n_components_90_variance': int(np.argmax(cumulative_variance >= 0.9) + 1),
                'n_components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
                'dimensionality_reduction_potential': float(1 - (np.argmax(cumulative_variance >= 0.9) + 1) / len(df.columns))
            }
            
        except Exception as e:
            return {'error': str(e)}


class ManufacturingDomainAnalyzer:
    """
    Basic manufacturing domain analysis - Phase 1 implementation
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        
    def analyze_manufacturing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic manufacturing pattern analysis"""
        self.logger.info("🏭 Analyzing manufacturing domain patterns")
        
        results = {
            'quality_patterns': self._basic_quality_analysis(df),
            'equipment_patterns': self._basic_equipment_analysis(df),
            'sensor_health': self._basic_sensor_health(df),
            'analysis_errors': []
        }
        
        return results
    
    def _basic_quality_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic quality pattern detection"""
        try:
            quality_indicators = ['quality', 'yield', 'defect', 'pass', 'fail', 'scrap', 'rework']
            quality_cols = [col for col in df.columns if any(ind in col.lower() for ind in quality_indicators)]
            
            if not quality_cols:
                return {'message': 'No quality-related columns detected'}
            
            quality_analysis = {}
            for col in quality_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    quality_analysis[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'quality_assessment': 'Stable' if df[col].std() / df[col].mean() < 0.1 else 'Variable'
                    }
                else:
                    quality_analysis[col] = {
                        'value_counts': df[col].value_counts().to_dict(),
                        'unique_values': df[col].nunique()
                    }
            
            return quality_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _basic_equipment_analysis(self, df: pd.DataFrame) -> Dict:
        """Basic equipment pattern detection"""
        try:
            equipment_indicators = ['machine', 'equipment', 'tool', 'line', 'station']
            equipment_cols = [col for col in df.columns if any(ind in col.lower() for ind in equipment_indicators)]
            
            if not equipment_cols:
                return {'message': 'No equipment-related columns detected'}
            
            equipment_analysis = {}
            for col in equipment_cols:
                equipment_analysis[col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'distribution': df[col].value_counts().head().to_dict()
                }
            
            return equipment_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _basic_sensor_health(self, df: pd.DataFrame) -> Dict:
        """Basic sensor health assessment"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            sensor_health = {}
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    variance = data.var()
                    unique_ratio = data.nunique() / len(data)
                    
                    sensor_health[col] = {
                        'variance': float(variance),
                        'unique_ratio': float(unique_ratio),
                        'health_assessment': (
                            'Healthy' if variance > 0 and unique_ratio > 0.1 else
                            'Stuck/Suspect' if variance == 0 else
                            'Monitor'
                        )
                    }
            
            return sensor_health
            
        except Exception as e:
            return {'error': str(e)}


class ReportGenerator:
    """
    Basic report generator - Phase 1 implementation
    """
    
    def __init__(self, config: EDAConfig, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        
    def generate_comprehensive_report(self, df: pd.DataFrame, analysis_results: Dict, viz_paths: Dict) -> str:
        """Generate basic comprehensive report"""
        self.logger.info("📄 Generating comprehensive report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"eda_report_{timestamp}.html"
        summary_path = self.output_dir / f"eda_summary_{timestamp}.json"
        
        try:
            # Save JSON summary
            with open(summary_path, 'w') as f:
                json.dump(analysis_results, f, indent=4, default=str)
            
            # Generate HTML report
            html_content = self._generate_basic_html_report(df, analysis_results, viz_paths, str(summary_path))
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"📄 Report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            raise
    
    def _generate_basic_html_report(self, df: pd.DataFrame, results: Dict, viz_paths: Dict, json_path: str) -> str:
        """Generate basic HTML report content"""
        
        # Basic dataset info
        n_rows, n_cols = df.shape
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_data_pct = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100
        
        # Generate visualization links
        viz_links = ""
        for viz_type, path in viz_paths.items():
            if path:
                viz_links += f'<li><a href="{path}" target="_blank">{viz_type.replace("_", " ").title()}</a></li>'
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>EDA Report - Phase 1</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .info-box {{ background: #f4f4f4; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .section {{ margin-bottom: 30px; }}
                .warning {{ color: #d9534f; }}
                .success {{ color: #5cb85c; }}
            </style>
        </head>
        <body>
            <h1>🔬 Scientific EDA Report - Phase 1</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>📊 Dataset Overview</h2>
                <div class="info-box">
                    <p><strong>Shape:</strong> {n_rows:,} rows × {n_cols} columns</p>
                    <p><strong>Numeric Features:</strong> {numeric_cols}</p>
                    <p><strong>Categorical Features:</strong> {categorical_cols}</p>
                    <p><strong>Missing Data:</strong> {missing_data_pct:.1f}%</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Analysis Summary</h2>
                <div class="info-box">
                    <p><strong>Statistical Analysis:</strong> {'✅ Completed' if 'statistical_analysis' in results else '❌ Failed'}</p>
                    <p><strong>Time Series Detection:</strong> {'✅ Detected' if results.get('temporal_analysis', {}).get('time_series_detected', False) else '❌ Not Detected'}</p>
                    <p><strong>Manufacturing Patterns:</strong> {'✅ Analyzed' if 'manufacturing_analysis' in results else '❌ Not Available'}</p>
                    <p><strong>ML Readiness:</strong> {results.get('ml_readiness', {}).get('overall_score', 'N/A')}/100</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Visualizations</h2>
                <ul>
                    {viz_links}
                </ul>
            </div>
            
            <div class="section">
                <h2>💾 Data Files</h2>
                <p><strong>JSON Summary:</strong> <a href="{json_path}" download>Download Analysis Results</a></p>
            </div>
            
            <div class="section">
                <h2>🎯 Next Steps</h2>
                <ol>
                    <li>Review generated visualizations</li>
                    <li>Address any data quality issues</li>
                    <li>Proceed with advanced analysis (Phase 2)</li>
                    <li>Implement ML pipeline recommendations</li>
                </ol>
            </div>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p><em>Generated by Scientific EDA Pipeline v2.0 - Phase 1</em></p>
            </footer>
        </body>
        </html>
        """
        
        return html_template


# Test function for Phase 1
def test_phase1_functionality():
    """Test Phase 1 core functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 1 CORE FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test configuration
        print("\n1. Testing Configuration...")
        config = EDAConfig()
        print(f"✅ Config created: output_dir={config.output_dir}")
        
        # Test profiler
        print("\n2. Testing DataProfiler...")
        profiler = DataProfiler(config)
        print("✅ DataProfiler initialized")
        
        # Test loader
        print("\n3. Testing DataLoader...")
        loader = DataLoader(config, profiler.logger)
        print("✅ DataLoader initialized")
        
        # Create sample data for testing
        print("\n4. Creating sample data for testing...")
        sample_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.randint(0, 2, 1000),
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
        })
        
        # Save sample data
        sample_path = Path(config.output_dir) / "sample_data.csv"
        sample_data.to_csv(sample_path, index=False)
        print(f"✅ Sample data saved to: {sample_path}")
        
        # Test file profiling
        print("\n5. Testing file profiling...")
        metadata = profiler.profile_file_metadata(str(sample_path))
        print(f"✅ File metadata: {metadata['file_size_mb']:.2f} MB")
        
        # Test data loading
        print("\n6. Testing data loading...")
        loaded_df = loader.load_data_smart(str(sample_path), metadata)
        print(f"✅ Data loaded: {loaded_df.shape}")
        
        # Test time series detection
        print("\n7. Testing TimeSeriesDetector...")
        time_detector = TimeSeriesDetector(config, profiler.logger)
        temporal_results = time_detector.detect_temporal_features(loaded_df)
        print(f"✅ Temporal analysis: {temporal_results['time_series_detected']}")
        
        # Test statistical analyzer
        print("\n8. Testing StatisticalAnalyzer...")
        stats_analyzer = StatisticalAnalyzer(config, profiler.logger)
        stats_results = stats_analyzer.comprehensive_statistical_analysis(loaded_df)
        print(f"✅ Statistical analysis completed")
        
        # Test visualization engine
        print("\n9. Testing VisualizationEngine...")
        viz_engine = VisualizationEngine(config, profiler.logger)
        viz_paths = viz_engine.create_comprehensive_visualizations(loaded_df, {'statistical_analysis': stats_results})
        print(f"✅ Visualizations created: {len(viz_paths)} plots")
        
        # Test dimensionality analyzer
        print("\n10. Testing DimensionalityAnalyzer...")
        dim_analyzer = DimensionalityAnalyzer(config, profiler.logger)
        dim_results = dim_analyzer.analyze_dimensionality(loaded_df)
        print(f"✅ Dimensionality analysis completed")
        
        # Test manufacturing analyzer
        print("\n11. Testing ManufacturingDomainAnalyzer...")
        mfg_analyzer = ManufacturingDomainAnalyzer(config, profiler.logger)
        mfg_results = mfg_analyzer.analyze_manufacturing_patterns(loaded_df)
        print(f"✅ Manufacturing analysis completed")
        
        # Test report generator
        print("\n12. Testing ReportGenerator...")
        report_generator = ReportGenerator(config, profiler.logger)
        
        all_results = {
            'df_shape': loaded_df.shape,
            'temporal_analysis': temporal_results,
            'statistical_analysis': stats_results,
            'dimensionality_analysis': dim_results,
            'manufacturing_analysis': mfg_results
        }
        
        report_path = report_generator.generate_comprehensive_report(loaded_df, all_results, viz_paths)
        print(f"✅ Report generated: {report_path}")
        
        print("\n" + "="*60)
        print("🎉 PHASE 1 TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    print("Starting Phase 1 utilities test...")
    success = test_phase1_functionality()
    
    if success:
        print("\n🚀 Phase 1 utilities are working correctly!")
        print("Ready for integration with main EDA pipeline.")
    else:
        print("\n🔧 Fix Phase 1 issues before proceeding")