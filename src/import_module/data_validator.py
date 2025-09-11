"""
RAN Data Validator
==================

Validates RAN CSV files and DataFrames before import to ensure data quality
and compatibility with the RAN data schema. Handles both 2G and 4G specific validations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import re
from datetime import datetime


class DataValidator:
    """
    Validates RAN performance data before database import.
    Provides specific validation for 2G and 4G network performance metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Required columns for all RAN data
        self.required_columns = ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']
        
        # RAN 2G specific patterns and metrics
        self.ran_2g_metrics = {
            'call_metrics': ['ccalls', 'ccongs', 'cndrop', 'cnrelcong'],
            'channel_metrics': ['cs12dlack', 'cs12ulack', 'cs14dlack'],
            'traffic_metrics': ['tfcassall', 'tfncedrop', 'tftralacc', 'thcassall'],
            'handover_metrics': ['sumeohatt', 'sumeohsucc', 'sumohoatt', 'sumohosucc'],
            'quality_metrics': ['msestdltbf', 'msestultbf', 'tavaacc', 'tavascan']
        }
        
        # RAN 4G specific patterns and metrics
        self.ran_4g_metrics = {
            'connection_metrics': ['pmrrcconnestab*', 'pms1sigconnestab*'],
            'bearer_metrics': ['pmerabestab*', 'pmpdcp*'],
            'handover_metrics': ['pmhoexe*'],
            'resource_metrics': ['pmprb*'],
            'release_metrics': ['pmerabrel*'],
            'throughput_metrics': ['pmuethp*']
        }
        
        # Valid regions (can be extended based on actual data)
        self.valid_regions = ['Sumbagteng', 'Sumbagtim', 'Sumbagsel', 'Sumbagut']
        
    def validate_ran_csv_file(self, csv_path: str, ran_tech: str) -> bool:
        """
        Validate that the RAN CSV file exists, is readable, and has expected structure.
        
        Args:
            csv_path: Path to the CSV file
            ran_tech: RAN technology (2G or 4G)
            
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
            
            # Check file size is reasonable (not too large)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1000:  # 1GB limit
                self.logger.warning(f"Large CSV file ({file_size_mb:.1f} MB): {csv_path}")
            
            # Try to read header and validate structure
            try:
                df_header = pd.read_csv(csv_path, nrows=0)
                if not self._validate_ran_columns(df_header.columns.tolist(), ran_tech):
                    return False
                    
                # Try to read first few rows
                df_sample = pd.read_csv(csv_path, nrows=5)
                if not self._validate_ran_data_sample(df_sample, ran_tech):
                    return False
                    
            except pd.errors.EmptyDataError:
                self.logger.error(f"CSV file has no data: {csv_path}")
                return False
            except pd.errors.ParserError as e:
                self.logger.error(f"CSV parsing error: {e}")
                return False
            
            self.logger.info(f"RAN {ran_tech} CSV file validation passed: {csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV file validation failed: {e}")
            return False
    
    def validate_ran_dataframe(self, df: pd.DataFrame, ran_tech: str) -> bool:
        """
        Validate the structure and content of the RAN DataFrame.
        
        Args:
            df: DataFrame to validate
            ran_tech: RAN technology (2G or 4G)
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                self.logger.error("DataFrame is empty")
                return False
            
            # Check for required columns
            if not self._validate_required_columns(df.columns.tolist()):
                return False
            
            # Validate RAN-specific columns
            if not self._validate_ran_columns(df.columns.tolist(), ran_tech):
                return False
            
            # Check for duplicate rows based on ID
            if 'id' in df.columns:
                duplicates = df['id'].duplicated().sum()
                if duplicates > 0:
                    self.logger.warning(f"Found {duplicates} duplicate IDs in DataFrame")
            
            # Validate data quality
            if not self._validate_data_quality(df, ran_tech):
                return False
            
            # Validate timestamp format
            if 'last_update' in df.columns:
                if not self._validate_timestamps(df['last_update']):
                    return False
            
            # Validate geographical data
            if not self._validate_geographical_data(df):
                return False
            
            self.logger.info(f"RAN {ran_tech} DataFrame validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"DataFrame validation failed: {e}")
            return False
    
    def _validate_required_columns(self, columns: List[str]) -> bool:
        """Validate that all required columns are present."""
        missing_columns = set(self.required_columns) - set(columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def _validate_ran_columns(self, columns: List[str], ran_tech: str) -> bool:
        """Validate RAN-specific columns based on technology."""
        columns_lower = [col.lower() for col in columns]
        
        if ran_tech == "2G":
            # Check for at least some 2G metrics
            found_metrics = 0
            for category, metrics in self.ran_2g_metrics.items():
                for metric in metrics:
                    if any(metric.lower() in col for col in columns_lower):
                        found_metrics += 1
                        break
            
            if found_metrics < 2:
                self.logger.error(f"Insufficient 2G metrics found in columns")
                return False
                
        elif ran_tech == "4G":
            # Check for at least some 4G metrics
            found_metrics = 0
            for category, patterns in self.ran_4g_metrics.items():
                for pattern in patterns:
                    pattern_clean = pattern.replace('*', '')
                    if any(pattern_clean.lower() in col for col in columns_lower):
                        found_metrics += 1
                        break
            
            if found_metrics < 2:
                self.logger.error(f"Insufficient 4G metrics found in columns")
                return False
        
        return True
    
    def _validate_ran_data_sample(self, df: pd.DataFrame, ran_tech: str) -> bool:
        """Validate a sample of RAN data."""
        try:
            # Check ID column has valid values
            if 'id' in df.columns:
                if df['id'].isnull().any():
                    self.logger.error("Found null values in ID column")
                    return False
            
            # Check region values
            if 'region' in df.columns:
                invalid_regions = df[~df['region'].isin(self.valid_regions)]['region'].unique()
                if len(invalid_regions) > 0:
                    self.logger.warning(f"Found non-standard regions: {invalid_regions}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data sample validation failed: {e}")
            return False
    
    def _validate_data_quality(self, df: pd.DataFrame, ran_tech: str) -> bool:
        """Validate data quality metrics."""
        try:
            # Check for excessive null values
            null_percentages = df.isnull().sum() / len(df) * 100
            high_null_cols = null_percentages[null_percentages > 50].index.tolist()
            
            if high_null_cols:
                self.logger.warning(f"Columns with >50% null values: {high_null_cols}")
            
            # Check for unrealistic values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'id':  # Skip ID column
                    # Check for negative values where they shouldn't exist
                    if col.lower() in ['count', 'success', 'attempt', 'volume']:
                        negative_count = (df[col] < 0).sum()
                        if negative_count > 0:
                            self.logger.warning(f"Found {negative_count} negative values in {col}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return False
    
    def _validate_timestamps(self, timestamp_series: pd.Series) -> bool:
        """Validate timestamp format and values."""
        try:
            # Try to convert to datetime
            converted = pd.to_datetime(timestamp_series, errors='coerce')
            null_count = converted.isnull().sum()
            
            if null_count > len(timestamp_series) * 0.1:  # More than 10% invalid
                self.logger.error(f"Too many invalid timestamps: {null_count}/{len(timestamp_series)}")
                return False
            
            # Check timestamp range (should be recent)
            min_date = converted.min()
            max_date = converted.max()
            
            if pd.notnull(min_date) and min_date < pd.Timestamp('2020-01-01'):
                self.logger.warning(f"Very old timestamp found: {min_date}")
            
            if pd.notnull(max_date) and max_date > pd.Timestamp.now() + pd.Timedelta(days=1):
                self.logger.warning(f"Future timestamp found: {max_date}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Timestamp validation failed: {e}")
            return False
    
    def _validate_geographical_data(self, df: pd.DataFrame) -> bool:
        """Validate geographical data consistency."""
        try:
            geo_cols = ['region', 'kabupaten', 'siteid']
            available_geo_cols = [col for col in geo_cols if col in df.columns]
            
            if not available_geo_cols:
                self.logger.warning("No geographical columns found")
                return True
            
            # Check for null values in geographical columns
            for col in available_geo_cols:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.logger.warning(f"Found {null_count} null values in {col}")
            
            # Check siteid format (should be alphanumeric)
            if 'siteid' in df.columns:
                invalid_siteids = df[~df['siteid'].astype(str).str.match(r'^[A-Za-z0-9]+$')]['siteid']
                if len(invalid_siteids) > 0:
                    self.logger.warning(f"Found {len(invalid_siteids)} invalid siteid formats")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Geographical data validation failed: {e}")
            return False
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
