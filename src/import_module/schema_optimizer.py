"""
RAN Schema Optimizer
====================

Optimizes database schema for RAN performance data storage and querying.
Handles data type optimization, index creation, and performance tuning
specifically for 2G and 4G network performance metrics.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class SchemaOptimizer:
    """
    Optimizes database schema for efficient storage and querying of RAN data.
    Provides technology-specific optimizations for 2G and 4G data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define optimal data types for common RAN columns
        self.common_type_mapping = {
            'id': 'INTEGER',
            'region': 'TEXT',
            'kabupaten': 'TEXT',
            'siteid': 'TEXT',
            'eniqhost': 'TEXT',
            'last_update': 'TIMESTAMP'
        }
        
        # RAN 2G specific optimizations
        self.ran_2g_optimizations = {
            'integer_columns': ['ccalls', 'ccongs', 'cndrop', 'cnrelcong', 'pmcount_2g'],
            'float_columns': ['tavaacc', 'tavascan'],
            'bigint_columns': ['cs12dlack', 'cs12ulack', 'cs14dlack', 'mc19dlack', 'mc19ulack'],
            'count_columns': ['numtchoffps', 'numtchoffsdcch', 'numtrxoffps']
        }
        
        # RAN 4G specific optimizations
        self.ran_4g_optimizations = {
            'integer_columns': ['pm_count', 'pmcelldowntimeauto', 'pmcelldowntimeman'],
            'float_columns': ['pm_rrcconnuser'],
            'bigint_columns': ['pmpdcpvoldldrb', 'pmpdcpvoldldrblasttti', 'pmpdcpvoluldrb',
                             'pmprbavaildl', 'pmprbavailul', 'pmprbuseddldtch', 'pmprbuseduldtch'],
            'count_columns': ['pmerabestabattadded', 'pmerabestabattinit', 'pmrrcconnestabatt']
        }
        
        # Define columns that should be indexed for performance
        self.index_columns = [
            'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update'
        ]
    
    def optimize_ran_dataframe(self, df: pd.DataFrame, ran_tech: str) -> pd.DataFrame:
        """
        Optimize DataFrame data types for efficient storage based on RAN technology.
        
        Args:
            df: Input DataFrame
            ran_tech: RAN technology (2G or 4G)
            
        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        try:
            optimized_df = df.copy()
            
            # Apply common optimizations first
            optimized_df = self._apply_common_optimizations(optimized_df)
            
            # Apply technology-specific optimizations
            if ran_tech == "2G":
                optimized_df = self._apply_2g_optimizations(optimized_df)
            elif ran_tech == "4G":
                optimized_df = self._apply_4g_optimizations(optimized_df)
            
            # Apply general column optimizations
            for col in df.columns:
                if col not in self.common_type_mapping:
                    optimized_df[col] = self._optimize_column_by_content(df[col], col)
            
            # Memory usage comparison
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            self.logger.info(f"DataFrame optimization completed: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                           f"({(original_memory - optimized_memory) / original_memory * 100:.1f}% reduction)")
            
            return optimized_df
            
        except Exception as e:
            self.logger.error(f"DataFrame optimization failed: {e}")
            return df
    
    def _apply_common_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply optimizations for common RAN columns."""
        for col, dtype in self.common_type_mapping.items():
            if col in df.columns:
                if dtype == 'INTEGER':
                    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
                elif dtype == 'TEXT':
                    df[col] = df[col].astype('string')
                elif dtype == 'TIMESTAMP':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _apply_2g_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 2G-specific optimizations."""
        optimizations = self.ran_2g_optimizations
        
        # Integer columns (small values)
        for col in optimizations['integer_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # Float columns (percentage/ratio values)
        for col in optimizations['float_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        # Big integer columns (counters, volumes)
        for col in optimizations['bigint_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Count columns (non-negative integers)
        for col in optimizations['count_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='unsigned')
        
        return df
    
    def _apply_4g_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 4G-specific optimizations."""
        optimizations = self.ran_4g_optimizations
        
        # Integer columns (small values)
        for col in optimizations['integer_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # Float columns (averages, ratios)
        for col in optimizations['float_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        # Big integer columns (volumes, byte counts)
        for col in optimizations['bigint_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Count columns (attempts, successes)
        for col in optimizations['count_columns']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='unsigned')
        
        return df
    
    def _optimize_column_by_content(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Optimize column based on its content and naming pattern.
        
        Args:
            series: Pandas series to optimize
            col_name: Column name for pattern matching
            
        Returns:
            pd.Series: Optimized series
        """
        col_lower = col_name.lower()
        
        # Numeric optimization
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's a counter/attempt/success metric
            if any(pattern in col_lower for pattern in ['count', 'att', 'succ', 'drop', 'fail']):
                return pd.to_numeric(series, errors='coerce', downcast='unsigned')
            
            # Check if it's a percentage/ratio
            elif any(pattern in col_lower for pattern in ['rate', 'ratio', 'percent', 'bler']):
                return pd.to_numeric(series, errors='coerce', downcast='float')
            
            # Check if it's a volume/throughput metric
            elif any(pattern in col_lower for pattern in ['vol', 'byte', 'bit', 'throughput']):
                return pd.to_numeric(series, errors='coerce')  # Keep as int64/float64 for large values
            
            # Default numeric optimization
            else:
                if series.dtype in ['int64', 'int32']:
                    return pd.to_numeric(series, errors='coerce', downcast='integer')
                else:
                    return pd.to_numeric(series, errors='coerce', downcast='float')
        
        # String optimization
        elif series.dtype == 'object':
            # Check if it looks like categorical data
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # Less than 10% unique values
                return series.astype('category')
            else:
                return series.astype('string')
        
        return series
        min_val = series.min()
        max_val = series.max()
        
        if min_val >= 0:
            if max_val < 256:
                return series.astype('uint8')
            elif max_val < 65536:
                return series.astype('uint16')
            elif max_val < 4294967296:
                return series.astype('uint32')
        else:
            if min_val >= -128 and max_val < 128:
                return series.astype('int8')
            elif min_val >= -32768 and max_val < 32768:
                return series.astype('int16')
            elif min_val >= -2147483648 and max_val < 2147483648:
                return series.astype('int32')
        
        return series
    
    def _optimize_float_column(self, series: pd.Series) -> pd.Series:
        """
        Optimize float column to float32 if precision allows.
        """
        # Try converting to float32 and check if values are preserved
        float32_series = series.astype('float32')
        if np.allclose(series.dropna(), float32_series.dropna(), equal_nan=True):
            return float32_series
        return series
    
    def _optimize_string_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Optimize string columns by converting to category if appropriate.
        """
        unique_ratio = series.nunique() / len(series)
        
        # Convert to category if low cardinality
        if unique_ratio < 0.5 and series.nunique() < 1000:
            return series.astype('category')
        
        return series
    
    def _optimize_datetime_column(self, series: pd.Series) -> pd.Series:
        """
        Convert and optimize datetime columns.
        """
        try:
            return pd.to_datetime(series)
        except:
            self.logger.warning(f"Could not convert column to datetime")
            return series
    
    def create_indexes(self, conn: sqlite3.Connection, table_name: str, columns: List[str]) -> None:
        """
        Create performance indexes on important columns.
        
        Args:
            conn: SQLite connection
            table_name: Name of the table
            columns: List of column names
        """
        try:
            cursor = conn.cursor()
            
            for col in columns:
                col_lower = col.lower()
                
                # Create index on important columns
                if any(pattern in col_lower for pattern in self.index_columns):
                    index_name = f"idx_{table_name}_{col.replace(' ', '_')}"
                    create_index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col})"
                    cursor.execute(create_index_sql)
                    self.logger.info(f"Created index: {index_name}")
            
            # Create composite indexes for common query patterns
            self._create_composite_indexes(cursor, table_name, columns)
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    def _create_composite_indexes(self, cursor: sqlite3.Cursor, table_name: str, columns: List[str]) -> None:
        """
        Create composite indexes for common RAN query patterns.
        """
        columns_lower = [col.lower() for col in columns]
        
        # Common composite indexes for RAN data
        composite_patterns = [
            (['timestamp', 'cell_id'], 'time_cell'),
            (['date', 'site_id'], 'date_site'),
            (['cell_id', 'timestamp'], 'cell_time')
        ]
        
        for pattern_cols, suffix in composite_patterns:
            available_cols = []
            for pattern_col in pattern_cols:
                for actual_col in columns:
                    if pattern_col in actual_col.lower():
                        available_cols.append(actual_col)
                        break
            
            if len(available_cols) >= 2:
                index_name = f"idx_{table_name}_{suffix}"
                cols_str = ', '.join(available_cols[:2])
                create_index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({cols_str})"
                try:
                    cursor.execute(create_index_sql)
                    self.logger.info(f"Created composite index: {index_name}")
                except Exception as e:
                    self.logger.warning(f"Could not create composite index {index_name}: {e}")
    
    def analyze_table_statistics(self, conn: sqlite3.Connection, table_name: str) -> Dict:
        """
        Analyze table statistics for optimization recommendations.
        
        Args:
            conn: SQLite connection
            table_name: Name of the table
            
        Returns:
            Dict: Table statistics and optimization recommendations
        """
        # Implementation placeholder for table analysis
        pass
