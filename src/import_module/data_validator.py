"""
Data Validator
==============

Validates CSV files and DataFrames before import to ensure data quality
and compatibility with the RAN data schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging


class DataValidator:
    """
    Validates RAN performance data before database import.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define RAN-specific validation rules
        self.ran_columns_patterns = [
            'cell_id', 'site_id', 'sector_id', 'enb_id', 'gnb_id',
            'rsrp', 'rsrq', 'sinr', 'throughput', 'latency',
            'bler', 'cqi', 'rank_indicator', 'pmi',
            'timestamp', 'date', 'hour', 'minute'
        ]
    
    def validate_csv_file(self, csv_path: str) -> bool:
        """
        Validate that the CSV file exists and is readable.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            file_path = Path(csv_path)
            
            # Check if file exists
            if not file_path.exists():
                self.logger.error(f"CSV file does not exist: {csv_path}")
                return False
            
            # Check file size
            if file_path.stat().st_size == 0:
                self.logger.error(f"CSV file is empty: {csv_path}")
                return False
            
            # Try to read first few rows
            pd.read_csv(csv_path, nrows=5)
            
            self.logger.info(f"CSV file validation passed: {csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV file validation failed: {e}")
            return False
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate the structure and content of the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                self.logger.error("DataFrame is empty")
                return False
            
            # Check for duplicate columns
            if len(df.columns) != len(set(df.columns)):
                self.logger.warning("DataFrame has duplicate column names")
            
            # Validate RAN-specific patterns
            self._validate_ran_patterns(df)
            
            # Check data types
            self._validate_data_types(df)
            
            # Check for missing values
            self._check_missing_values(df)
            
            self.logger.info("DataFrame validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"DataFrame validation failed: {e}")
            return False
    
    def _validate_ran_patterns(self, df: pd.DataFrame) -> None:
        """
        Validate RAN-specific column patterns and data ranges.
        """
        columns = [col.lower() for col in df.columns]
        
        # Check for essential RAN columns
        essential_patterns = ['cell', 'site', 'timestamp', 'date']
        found_patterns = []
        
        for pattern in essential_patterns:
            if any(pattern in col for col in columns):
                found_patterns.append(pattern)
        
        if len(found_patterns) < 2:
            self.logger.warning(f"Few RAN-specific columns found: {found_patterns}")
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """
        Validate and suggest appropriate data types for columns.
        """
        for col in df.columns:
            col_lower = col.lower()
            
            # Check timestamp columns
            if any(pattern in col_lower for pattern in ['timestamp', 'date', 'time']):
                try:
                    pd.to_datetime(df[col].head())
                    self.logger.info(f"Column {col} appears to be datetime")
                except:
                    self.logger.warning(f"Column {col} might be datetime but conversion failed")
            
            # Check numeric KPI columns
            if any(pattern in col_lower for pattern in ['rsrp', 'rsrq', 'sinr', 'throughput', 'latency']):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.warning(f"Column {col} should be numeric but isn't")
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """
        Check for missing values and report statistics.
        """
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        for col, missing_count in missing_stats.items():
            if missing_count > 0:
                self.logger.info(f"Column {col}: {missing_count} missing values ({missing_percent[col]:.2f}%)")
    
    def get_validation_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive validation report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict: Validation report with statistics and recommendations
        """
        # Implementation placeholder for detailed validation report
        pass
