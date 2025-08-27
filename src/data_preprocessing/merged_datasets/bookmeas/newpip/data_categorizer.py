"""
Safe Manufacturing Data Categorizer
REMOVED optimization to avoid duplication - only categorization now
"""

import pandas as pd
import numpy as np
import gc
import psutil
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

class ManufacturingDataCategorizer:
    """
    Manufacturing data categorization ONLY - optimization moved to main pipeline
    """
    
    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.categorization_stats = {}
        
        # Define exact categories as per manufacturing domain requirements
        self.numeric_keywords = [
            'measure_value', 'measure_step_number', 'lower_limit', 'upper_limit'
        ]
        
        self.datetime_keywords = [
            'updated_at', 'created_at', 'book_stamp'
        ]
        
        self.categorical_keywords = [
            'booking_id', 'serial_number_id', 'catalog_id', 'teststep_id', 
            'workstep_id', 'workorder_id', 'recipe_revision_id', 'station_id', 
            'erp_group_id', 'object_id', 'product_variant_id', 'workplan_id', 
            'station_diag_id', 'product_id', 'line_id', 'lot_id', 'serial_number', 
            'measurement_name', 'measurement_unit', 'workorder_number', 
            'station_number', 'workstep_number_mes', 'workstep_number_erp', 
            'part_number', 'part_group', 'measurement_type', 'panel_position_number', 
            'workorder_type', 'workstep_number_alt', 'lot_number', 'station_desc', 
            'workstep_desc', 'erp_group_desc', 'workorder_desc', 'part_desc', 
            'station_diag_desc', 'line_desc', 'plant_desc'
        ]
        
        self.quality_keywords = [
            'sequence_number', 'book_state', 'station_diag_number', 
            'has_failures', 'measure_fail_code'
        ]
    
    def categorize_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize DataFrame columns based on manufacturing domain knowledge
        Returns categories exactly as specified in requirements
        """
        
        categories = {
            'numeric': [],
            'datetime': [],
            'categorical_identifiers_descriptions': [],
            'quality_indicators_boolean': []
        }
        
        for col in df.columns:
            category = self._categorize_column(col, df[col])
            categories[category].append(col)
        
        # Log categorization results
        self._log_categorization_results(categories)
        
        return categories
    
    def _categorize_column(self, col_name: str, series: pd.Series) -> str:
        """Categorize a single column using exact matching and type detection"""
        
        # Exact match for numeric fields
        if col_name in self.numeric_keywords:
            return 'numeric'
        
        # Exact match for datetime fields  
        if col_name in self.datetime_keywords:
            return 'datetime'
            
        # Exact match for categorical fields
        if col_name in self.categorical_keywords:
            return 'categorical_identifiers_descriptions'
            
        # Exact match for quality indicators
        if col_name in self.quality_keywords:
            return 'quality_indicators_boolean'
        
        # Fallback to type-based detection
        return self._detect_by_type(series)
    
    def _detect_by_type(self, series: pd.Series) -> str:
        """Fallback type detection for uncategorized columns"""
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check for datetime types
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check for boolean types or binary values
        if pd.api.types.is_bool_dtype(series) or self._is_binary_like(series):
            return 'quality_indicators_boolean'
        
        # Default to categorical
        return 'categorical_identifiers_descriptions'
    
    def _is_binary_like(self, series: pd.Series) -> bool:
        """Check if series contains binary-like values"""
        
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2:
            binary_values = {0, 1, '0', '1', 'True', 'False', 'true', 'false', 
                           'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}
            return all(val in binary_values for val in unique_vals)
        return False
    
    def _log_categorization_results(self, categories: Dict[str, List[str]]) -> None:
        """Log categorization results in a tabular format"""

        self.logger.info("📋 Data categorization results:")

        for category, columns in categories.items():
            self.logger.info(f"\n📂 {category} ({len(columns)} columns)")

            if not columns:
                self.logger.info("   [No columns]")
                continue

            # Format columns into rows of fixed width
            col_width = 25   # adjust width for readability
            cols_per_row = 4 # how many columns to show per row

            for i in range(0, len(columns), cols_per_row):
                row = columns[i:i+cols_per_row]
                formatted_row = "".join(col.ljust(col_width) for col in row)
                self.logger.info("   " + formatted_row)

    def check_system_memory(self):
        """Check system memory status"""
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / (1024**2),
            'available_mb': mem.available / (1024**2),
            'used_mb': mem.used / (1024**2),
            'percentage': mem.percent
        }


# Factory function for easy integration
def create_manufacturing_data_categorizer(config=None, logger=None):
    """Factory function to create ManufacturingDataCategorizer"""
    return ManufacturingDataCategorizer(config, logger)