"""
Enhanced Pipeline Utilities
Comprehensive utility functions for the enhanced manufacturing data preprocessing pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union, Dict, List, Tuple
import json
import yaml
import pickle
import joblib
from contextlib import contextmanager
import psutil
import os
import warnings

class EnhancedPipelineUtils:
    """Enhanced utilities for pipeline operations"""
    
    @staticmethod
    def setup_logging(output_dir: Path) -> logging.Logger:
        """
        Setup comprehensive logging configuration with single log file
        
        Args:
            output_dir: Directory where log files will be saved
            
        Returns:
            Configured logger instance with comprehensive logging
        """
        
        # Create single comprehensive log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_log_file = output_dir / f"enhanced_pipeline_comprehensive_{timestamp}.log"
        
        # Create custom formatter for comprehensive logging
        comprehensive_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Create main logger
        logger = logging.getLogger('enhanced_pipeline')
        logger.setLevel(logging.DEBUG)  # Capture everything
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Single comprehensive file handler for all messages
        file_handler = logging.FileHandler(comprehensive_log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(comprehensive_formatter)
        
        # Enhanced console handler with same formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Show  and above in console
        
        # Try to use colored formatter, fallback to standard
        try:
            from colorlog import ColoredFormatter
            console_formatter = ColoredFormatter(
                '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green', 
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
        except ImportError:
            # Fallback to standard formatter
            console_handler.setFormatter(comprehensive_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # NOW logger is created, so we can log startup information
        logger.info(f"Enhanced comprehensive logging initialized")
        logger.info(f"Comprehensive log file: {comprehensive_log_file}")
        logger.debug(f"Logging all terminal output and debug information to single file")
        
        # Log system information for comprehensive tracking
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.debug(f"System Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
            logger.debug(f"CPU: {psutil.cpu_count()} cores")
            logger.debug("📊 Feature count consistency will be monitored throughout pipeline")
            logger.info(f"💾 System Memory Status: {memory.available / (1024**3):.1f}GB available")
        except Exception:
            logger.warning("💾 Memory monitoring unavailable")
        
        return logger
    
    @staticmethod
    def save_dataset(df: pd.DataFrame, file_path: Path, format: str = 'parquet', 
                    compression: Optional[str] = None, **kwargs) -> bool:
        """
        Enhanced dataset saving with multiple formats and compression options
        
        Args:
            df: DataFrame to save
            file_path: Path where file will be saved
            format: File format ('parquet', 'csv', 'pickle', 'feather', 'hdf5')
            compression: Compression method ('gzip', 'bz2', 'xz', 'snappy', etc.)
            **kwargs: Additional arguments for specific formats
            
        Returns:
            True if successful, False otherwise
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup existing file if it exists
            if file_path.exists():
                backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}")
                file_path.rename(backup_path)
                logger.info(f"Existing file backed up to: {backup_path}")
            
            # Save based on format
            start_time = datetime.now()
            
            if format.lower() == 'parquet':
                df.to_parquet(
                    file_path, 
                    index=kwargs.get('index', False),
                    compression=compression or 'snappy',
                    engine=kwargs.get('engine', 'pyarrow')
                )
                
            elif format.lower() == 'csv':
                df.to_csv(
                    file_path, 
                    index=kwargs.get('index', False),
                    compression=compression,
                    encoding=kwargs.get('encoding', 'utf-8')
                )
                
            elif format.lower() == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f, protocol=kwargs.get('protocol', pickle.HIGHEST_PROTOCOL))
                    
            elif format.lower() == 'feather':
                df.to_feather(file_path, compression=compression or 'uncompressed')
                
            elif format.lower() == 'hdf5' or format.lower() == 'h5':
                df.to_hdf(
                    file_path, 
                    key=kwargs.get('key', 'data'),
                    mode=kwargs.get('mode', 'w'),
                    complevel=kwargs.get('complevel', 9),
                    complib=kwargs.get('complib', 'zlib')
                )
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate and log performance metrics
            save_time = (datetime.now() - start_time).total_seconds()
            file_size = file_path.stat().st_size / (1024**2)  # MB
            
            logger.info(f"Dataset saved successfully:")
            logger.info(f"  Format: {format.upper()}")
            logger.info(f"  Size: {file_size:.2f} MB")
            logger.info(f"  Time: {save_time:.2f} seconds")
            logger.info(f"  Path: {file_path}")
            
            if compression:
                logger.info(f"  Compression: {compression}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset to {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def load_dataset(file_path: Path, format: str = 'auto', **kwargs) -> Optional[pd.DataFrame]:
        """
        Enhanced dataset loading with automatic format detection and error handling
        
        Args:
            file_path: Path to the file to load
            format: File format ('auto', 'parquet', 'csv', 'pickle', 'feather', 'hdf5')
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Loaded DataFrame or None if failed
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Auto-detect format if requested
            if format == 'auto':
                format = file_path.suffix.lower().lstrip('.')
                if format in ['h5', 'hdf']:
                    format = 'hdf5'
            
            start_time = datetime.now()
            
            # Load based on format
            if format == 'parquet':
                df = pd.read_parquet(file_path, engine=kwargs.get('engine', 'pyarrow'))
                
            elif format == 'csv':
                df = pd.read_csv(
                    file_path,
                    encoding=kwargs.get('encoding', 'utf-8'),
                    low_memory=kwargs.get('low_memory', False)
                )
                
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    
            elif format == 'feather':
                df = pd.read_feather(file_path)
                
            elif format == 'hdf5':
                df = pd.read_hdf(file_path, key=kwargs.get('key', 'data'))
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate and log performance metrics
            load_time = (datetime.now() - start_time).total_seconds()
            file_size = file_path.stat().st_size / (1024**2)  # MB
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Format: {format.upper()}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  File size: {file_size:.2f} MB")
            logger.info(f"  Load time: {load_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def create_plots_dir(output_dir: Path) -> Path:
        """
        Create enhanced plots directory with subdirectories
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Path to main plots directory
        """
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_column_access(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
        """
        Safely access column from DataFrame with enhanced error handling
        
        Args:
            df: DataFrame to access
            column: Column name to access
            default_value: Default value if column doesn't exist
            
        Returns:
            Series if column exists, Series with default values otherwise
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        if column in df.columns:
            return df[column]
        else:
            # Try case-insensitive matching
            matching_cols = [col for col in df.columns if col.lower() == column.lower()]
            if matching_cols:
                logger.warning(f"Column '{column}' not found, using '{matching_cols[0]}' (case mismatch)")
                return df[matching_cols[0]]
            
            # Try partial matching
            partial_matches = [col for col in df.columns if column.lower() in col.lower()]
            if partial_matches:
                logger.warning(f"Column '{column}' not found, potential matches: {partial_matches}")
            
            logger.warning(f"Column '{column}' not found in DataFrame. Using default value.")
            return pd.Series([default_value] * len(df), name=column, index=df.index)
    
    @staticmethod
    def format_memory_size(bytes_size: float) -> str:
        """
        Format memory size in human readable format with enhanced precision
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted string with appropriate unit
        """
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        size = float(bytes_size)
        
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == 'B':
                    return f"{size:.0f} {unit}"
                elif size >= 100:
                    return f"{size:.1f} {unit}"
                else:
                    return f"{size:.2f} {unit}"
            size /= 1024.0
        
        return f"{bytes_size} B"  # Fallback
    
    @staticmethod
    def format_number(num: Union[int, float], precision: int = 1) -> str:
        """
        Format large numbers with appropriate suffixes and enhanced precision
        
        Args:
            num: Number to format
            precision: Decimal precision for formatted numbers
            
        Returns:
            Formatted string with appropriate suffix
        """
        
        if abs(num) < 1000:
            return f"{num:,.0f}" if isinstance(num, int) else f"{num:,.{precision}f}"
        
        units = ['', 'K', 'M', 'B', 'T']
        size = float(abs(num))
        sign = '-' if num < 0 else ''
        
        for i, unit in enumerate(units):
            if size < 1000.0 or unit == units[-1]:
                if unit == '':
                    return f"{sign}{size:,.0f}"
                else:
                    return f"{sign}{size:.{precision}f}{unit}"
            size /= 1000.0
        
        return str(num)  # Fallback
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1, 
                          required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Enhanced DataFrame validation with detailed reporting
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required
            min_cols: Minimum number of columns required
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results and details
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic existence check
        if df is None:
            validation_result['valid'] = False
            validation_result['errors'].append("DataFrame is None")
            return validation_result
        
        # Shape validation
        if len(df) < min_rows:
            validation_result['valid'] = False
            validation_result['errors'].append(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        
        if len(df.columns) < min_cols:
            validation_result['valid'] = False
            validation_result['errors'].append(f"DataFrame has {len(df.columns)} columns, minimum {min_cols} required")
        
        # Required columns check
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Data quality checks
        if validation_result['valid']:
            # Check for completely empty DataFrame
            if df.empty:
                validation_result['warnings'].append("DataFrame is empty")
            
            # Check for all-null columns
            all_null_cols = df.columns[df.isnull().all()].tolist()
            if all_null_cols:
                validation_result['warnings'].append(f"Columns with all null values: {all_null_cols}")
            
            # Check for duplicate columns
            if df.columns.duplicated().any():
                duplicate_cols = df.columns[df.columns.duplicated()].tolist()
                validation_result['warnings'].append(f"Duplicate column names: {duplicate_cols}")
            
            # Gather additional info
            validation_result['info'] = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'null_percentage': (df.isnull().sum().sum() / df.size) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'dtypes_summary': df.dtypes.value_counts().to_dict()
            }
        
        # Log results
        if not validation_result['valid']:
            for error in validation_result['errors']:
                logger.error(f"DataFrame validation error: {error}")
        
        for warning in validation_result['warnings']:
            logger.warning(f"DataFrame validation warning: {warning}")
        
        if validation_result['valid'] and not validation_result['warnings']:
            logger.info("DataFrame validation passed successfully")
        
        return validation_result
    
    @staticmethod
    def get_comprehensive_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive DataFrame information with enhanced metrics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with comprehensive DataFrame information
        """
        
        if df is None or df.empty:
            return {'error': 'DataFrame is empty or None'}
        
        # Basic information
        memory_usage = df.memory_usage(deep=True).sum()
        
        # Column type analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()
        
        # Data quality metrics
        null_counts = df.isnull().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Memory usage by column type
        memory_by_type = {}
        for dtype_category, columns in [
            ('numeric', numeric_columns),
            ('categorical', categorical_columns), 
            ('datetime', datetime_columns),
            ('boolean', boolean_columns)
        ]:
            if columns:
                memory_by_type[dtype_category] = df[columns].memory_usage(deep=True).sum() / (1024**2)
        
        # Cardinality analysis
        cardinality_stats = {
            'low_cardinality': len([col for col in df.columns if df[col].nunique() <= 10]),
            'medium_cardinality': len([col for col in df.columns if 10 < df[col].nunique() <= 100]),
            'high_cardinality': len([col for col in df.columns if df[col].nunique() > 100])
        }
        
        # Missing data analysis
        missing_data_analysis = {
            'columns_with_missing': (null_counts > 0).sum(),
            'columns_mostly_missing': (null_counts > len(df) * 0.5).sum(),
            'total_missing_values': null_counts.sum(),
            'missing_percentage': (null_counts.sum() / df.size) * 100
        }
        
        return {
            # Basic info
            'shape': df.shape,
            'memory_usage_mb': memory_usage / (1024**2),
            'memory_usage_formatted': EnhancedPipelineUtils.format_memory_size(memory_usage),
            
            # Column types
            'column_types': {
                'numeric': len(numeric_columns),
                'categorical': len(categorical_columns),
                'datetime': len(datetime_columns),
                'boolean': len(boolean_columns)
            },
            'column_lists': {
                'numeric': numeric_columns,
                'categorical': categorical_columns,
                'datetime': datetime_columns,
                'boolean': boolean_columns
            },
            
            # Data quality
            'data_quality': {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(df)) * 100,
                **missing_data_analysis
            },
            
            # Memory analysis
            'memory_by_type': memory_by_type,
            
            # Cardinality
            'cardinality_distribution': cardinality_stats,
            
            # Advanced metrics
            'dtypes_summary': df.dtypes.value_counts().to_dict(),
            'index_type': type(df.index).__name__,
            'has_multiindex': isinstance(df.index, pd.MultiIndex),
            
            # Performance metrics
            'estimated_processing_time': EnhancedPipelineUtils._estimate_processing_time(df),
        }
    
    @staticmethod
    def _estimate_processing_time(df: pd.DataFrame) -> Dict[str, str]:
        """Estimate processing time for common operations"""
        
        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Rough estimates based on DataFrame size
        estimates = {
            'correlation_analysis': f"{max(1, int(n_cols**2 / 1000))} seconds",
            'feature_importance': f"{max(5, int((n_rows * n_cols) / 100000))} seconds", 
            'encoding_operations': f"{max(2, int(n_cols / 10))} seconds",
            'balancing_operations': f"{max(3, int(n_rows / 50000))} seconds"
        }
        
        return estimates
    
    @staticmethod
    def print_enhanced_dataframe_summary(df: pd.DataFrame, title: str = "Enhanced DataFrame Summary"):
        """
        Print comprehensive DataFrame summary with enhanced formatting
        
        Args:
            df: DataFrame to summarize
            title: Title for the summary
        """
        
        info = EnhancedPipelineUtils.get_comprehensive_dataframe_info(df)
        
        if 'error' in info:
            print(f"\n❌ {title}: {info['error']}")
            return
        
        print(f"\n{'='*155}")
        print(f"📊 {title}")
        print(f"{'='*155}")
        
        # Basic information
        print(f"📏 Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        print(f"💾 Memory: {info['memory_usage_formatted']}")
        
        # Column type distribution
        print(f"\n🏷️  Column Types:")
        for dtype, count in info['column_types'].items():
            if count > 0:
                percentage = (count / info['shape'][1]) * 100
                print(f"   • {dtype.title()}: {count} columns ({percentage:.1f}%)")
        
        # Data quality
        quality = info['data_quality']
        print(f"\n🔍 Data Quality:")
        print(f"   • Missing values: {quality['total_missing_values']:,} ({quality['missing_percentage']:.2f}%)")
        print(f"   • Columns with missing data: {quality['columns_with_missing']}")
        print(f"   • Duplicate rows: {quality['duplicate_rows']:,} ({quality['duplicate_percentage']:.2f}%)")
        
        # Cardinality distribution
        cardinality = info['cardinality_distribution']
        print(f"\n📊 Cardinality Distribution:")
        print(f"   • Low (≤10 unique): {cardinality['low_cardinality']} columns")
        print(f"   • Medium (11-100 unique): {cardinality['medium_cardinality']} columns") 
        print(f"   • High (>100 unique): {cardinality['high_cardinality']} columns")
        
        # Memory usage by type
        if info['memory_by_type']:
            print(f"\n💾 Memory by Type:")
            for dtype, memory_mb in info['memory_by_type'].items():
                print(f"   • {dtype.title()}: {memory_mb:.2f} MB")
        
        # Processing time estimates
        print(f"\n⏱️  Estimated Processing Times:")
        for operation, time_est in info['estimated_processing_time'].items():
            print(f"   • {operation.replace('_', ' ').title()}: {time_est}")
        
        print(f"{'='*155}")
    
    @staticmethod
    def create_backup(df: pd.DataFrame, backup_dir: Path, name: str, 
                     format: str = 'parquet', compression: str = 'snappy') -> bool:
        """
        Create enhanced backup of DataFrame with metadata
        
        Args:
            df: DataFrame to backup
            backup_dir: Directory for backups
            name: Backup name
            format: File format for backup
            compression: Compression method
            
        Returns:
            True if successful, False otherwise
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create backup file
            backup_file = backup_dir / f"backup_{name}_{timestamp}.{format}"
            
            success = EnhancedPipelineUtils.save_dataset(
                df, backup_file, format=format, compression=compression
            )
            
            if success:
                # Create metadata file
                metadata = {
                    'backup_name': name,
                    'timestamp': timestamp,
                    'original_shape': df.shape,
                    'format': format,
                    'compression': compression,
                    'file_size_mb': backup_file.stat().st_size / (1024**2),
                    'dataframe_info': EnhancedPipelineUtils.get_comprehensive_dataframe_info(df)
                }
                
                metadata_file = backup_dir / f"backup_{name}_{timestamp}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"Backup created successfully: {backup_file}")
                logger.info(f"Metadata saved: {metadata_file}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame, strategy: str = 'standard') -> pd.DataFrame:
        """
        Enhanced column name cleaning with multiple strategies
        
        Args:
            df: DataFrame with columns to clean
            strategy: Cleaning strategy ('standard', 'snake_case', 'camel_case', 'minimal')
            
        Returns:
            DataFrame with cleaned column names
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        df = df.copy()
        original_names = df.columns.tolist()
        
        if strategy == 'standard':
            # Standard cleaning: lowercase, underscores, alphanumeric
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
            df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
            df.columns = df.columns.str.lower()  # Convert to lowercase
            df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)  # Remove special chars
            df.columns = df.columns.str.replace(r'_+', '_', regex=True)  # Multiple underscores to single
            df.columns = df.columns.str.strip('_')  # Remove leading/trailing underscores
            
        elif strategy == 'snake_case':
            # Snake case with better handling of camelCase
            import re
            new_columns = []
            for col in df.columns:
                col = col.strip()
                # Convert camelCase to snake_case
                col = re.sub('([a-z0-9])([A-Z])', r'\1_\2', col)
                col = col.lower()
                col = re.sub(r'[^a-z0-9_]', '_', col)
                col = re.sub(r'_+', '_', col)
                col = col.strip('_')
                new_columns.append(col)
            df.columns = new_columns
            
        elif strategy == 'camel_case':
            # Convert to camelCase
            new_columns = []
            for col in df.columns:
                col = col.strip().lower()
                words = re.split(r'[^a-z0-9]+', col)
                camel = words[0] + ''.join(word.capitalize() for word in words[1:] if word)
                new_columns.append(camel)
            df.columns = new_columns
            
        elif strategy == 'minimal':
            # Minimal cleaning - just remove problematic characters
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace(r'["\'\n\r\t]', '', regex=True)
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        duplicated_mask = cols.duplicated(keep=False)
        
        if duplicated_mask.any():
            for dup in cols[duplicated_mask].unique():
                dup_indices = cols[cols == dup].index.values
                for i, idx in enumerate(dup_indices):
                    if i > 0:  # Keep first occurrence unchanged
                        cols.iloc[idx] = f"{dup}_{i}"
            
            df.columns = cols
            logger.warning(f"Fixed {duplicated_mask.sum()} duplicate column names")
        
        # Log changes
        changes = [(orig, new) for orig, new in zip(original_names, df.columns) if orig != new]
        if changes:
            logger.info(f"Column names cleaned using '{strategy}' strategy:")
            for orig, new in changes[:5]:  # Show first 5 changes
                logger.info(f"  '{orig}' → '{new}'")
            if len(changes) > 5:
                logger.info(f"  ... and {len(changes) - 5} more changes")
        
        return df
    
    @staticmethod
    def detect_encoding_issues(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Enhanced encoding issues detection with categorization
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with categorized encoding issues
        """
        
        issues = {
            'mixed_types': [],
            'high_cardinality': [],
            'constant_columns': [],
            'high_missing': [],
            'potential_identifiers': [],
            'date_like_strings': [],
            'numeric_as_strings': []
        }
        
        for col in df.columns:
            series = df[col]
            
            # Mixed types detection
            if series.dtype == 'object':
                sample = series.dropna().head(1000)
                if len(sample) > 0:
                    # Check for mixed numeric/string
                    numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                    if 0 < numeric_count < len(sample):
                        issues['mixed_types'].append(col)
                    
                    # Check for numeric strings
                    elif numeric_count == len(sample):
                        issues['numeric_as_strings'].append(col)
                    
                    # Check for date-like strings
                    else:
                        date_like_patterns = [
                            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                        ]
                        sample_str = sample.astype(str)
                        for pattern in date_like_patterns:
                            if sample_str.str.contains(pattern, regex=True).any():
                                issues['date_like_strings'].append(col)
                                break
            
            # High cardinality detection (adjusted threshold)
            cardinality_ratio = series.nunique() / len(series)
            if cardinality_ratio > 0.95 and series.nunique() > 1000:
                issues['high_cardinality'].append(col)
            
            # Potential identifier detection
            if (cardinality_ratio > 0.9 and 
                ('id' in col.lower() or 'key' in col.lower() or 'code' in col.lower())):
                issues['potential_identifiers'].append(col)
            
            # Constant columns
            if series.nunique() <= 1:
                issues['constant_columns'].append(col)
            
            # High missing values
            missing_ratio = series.isnull().sum() / len(series)
            if missing_ratio > 0.8:
                issues['high_missing'].append(col)
        
        return issues
    
    @staticmethod
    @contextmanager
    def performance_monitor(operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Context manager for monitoring operation performance
        
        Args:
            operation_name: Name of the operation being monitored
            logger: Logger instance to use
        """
        
        if logger is None:
            logger = logging.getLogger('enhanced_pipeline')
        
        # Get initial system metrics
        process = psutil.Process()
        start_time = datetime.now()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        start_cpu_percent = process.cpu_percent()
        
        logger.debug(f"Initial memory usage: {start_memory:.2f} MB")
        
        try:
            yield
            
        finally:
            # Calculate final metrics
            end_time = datetime.now()
            end_memory = process.memory_info().rss / (1024**2)  # MB
            duration = (end_time - start_time).total_seconds()
            
            # Get CPU usage (average over the operation)
            cpu_percent = process.cpu_percent()
            
            logger.info(f"✅ Completed operation: {operation_name}")
            logger.info(f"   Duration: {duration:.2f} seconds")
            logger.info(f"   Memory change: {end_memory - start_memory:+.2f} MB")
            logger.info(f"   Peak memory: {end_memory:.2f} MB")
            
            if duration > 1:  # Only log CPU for operations > 1 second
                logger.debug(f"   CPU usage: {cpu_percent:.1f}%")
    
    @staticmethod
    def save_pipeline_config(config: Dict[str, Any], output_dir: Path) -> bool:
        """
        Save pipeline configuration with timestamp and validation
        
        Args:
            config: Configuration dictionary to save
            output_dir: Directory to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        
        logger = logging.getLogger('enhanced_pipeline')
        
        try:
            timestamp = datetime.now().isoformat()
            config_with_metadata = {
                'pipeline_config': config,
                'metadata': {
                    'created_at': timestamp,
                    'pipeline_version': '2.1',
                    'config_version': '1.0'
                }
            }
            
            # Save as both JSON and YAML for flexibility
            json_file = output_dir / f'pipeline_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            yaml_file = output_dir / f'pipeline_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
            
            # Save JSON
            with open(json_file, 'w') as f:
                json.dump(config_with_metadata, f, indent=2, default=str)
            
            # Save YAML (if available)
            try:
                with open(yaml_file, 'w') as f:
                    yaml.dump(config_with_metadata, f, default_flow_style=False, indent=2)
            except Exception:
                logger.warning("YAML library not available, skipping YAML config save")
            
            logger.info(f"Pipeline configuration saved: {json_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pipeline configuration: {str(e)}")
            return False
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information for pipeline optimization
        
        Returns:
            Dictionary with system information
        """
        
        try:
            memory = psutil.virtual_memory()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
            
            disk_info = psutil.disk_usage('/')
            
            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'recommendation': (
                        'sufficient' if memory.available > 2 * (1024**3) else
                        'limited' if memory.available > 1 * (1024**3) else 'critical'
                    )
                },
                'cpu': cpu_info,
                'disk': {
                    'total_gb': disk_info.total / (1024**3),
                    'free_gb': disk_info.free / (1024**3),
                    'used_percent': (disk_info.used / disk_info.total) * 100
                },
                'python_info': {
                    'version': pd.__version__,
                    'pandas_version': pd.__version__,
                    'numpy_version': np.__version__
                }
            }
            
        except Exception as e:
            return {'error': f'Failed to get system info: {str(e)}'}

# In pipeline_utils.py - Add this progress tracking class
class MeaningfulProgress:
    """Progress tracker with meaningful updates"""
    
    def __init__(self, total_steps: int, description: str):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step_name: str):
        """Update progress with meaningful step description"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100
        
        print(f"🔄 {self.description}: {step_name} ({self.current_step}/{self.total_steps}) - {progress_pct:.1f}% - {elapsed:.1f}s elapsed")
    
    def complete(self):
        """Mark as complete"""
        total_time = time.time() - self.start_time
        print(f"✅ {self.description} completed in {total_time:.1f}s")

# Usage example in methods:
def _apply_fixed_smote_balancing(self, df: pd.DataFrame, target_col: str, 
                                analysis: Dict, original_counts: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Fixed SMOTE with meaningful progress"""
    
    progress = MeaningfulProgress(4, "SMOTE Balancing")
    
    progress.update("Preparing data and encoding categorical features")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # ... encoding logic ...
    
    progress.update("Handling missing values and infinities")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    progress.update("Applying SMOTE algorithm")
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    progress.update("Combining results and finalizing")
    balanced_df = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name=target_col)
    ], axis=1)
    
    progress.complete()
    
    # ... rest of method
# Convenience functions for backward compatibility
def setup_logging(output_dir: Path) -> logging.Logger:
    """Convenience function for setting up logging"""
    return EnhancedPipelineUtils.setup_logging(output_dir)

def save_dataset(df: pd.DataFrame, file_path: Path, format: str = 'parquet') -> bool:
    """Convenience function for saving datasets"""
    return EnhancedPipelineUtils.save_dataset(df, file_path, format)

def create_plots_dir(output_dir: Path) -> Path:
    """Convenience function for creating plots directory"""
    return EnhancedPipelineUtils.create_plots_dir(output_dir)

def safe_column_access(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
    """Convenience function for safe column access"""
    return EnhancedPipelineUtils.safe_column_access(df, column, default_value)

def format_memory_size(bytes_size: float) -> str:
    """Convenience function for formatting memory size"""
    return EnhancedPipelineUtils.format_memory_size(bytes_size)

def format_number(num: Union[int, float]) -> str:
    """Convenience function for formatting numbers"""
    return EnhancedPipelineUtils.format_number(num)

def print_dataframe_summary(df: pd.DataFrame, title: str = "DataFrame Summary"):
    """Convenience function for printing DataFrame summary"""
    return EnhancedPipelineUtils.print_enhanced_dataframe_summary(df, title)