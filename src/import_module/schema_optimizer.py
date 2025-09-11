"""
Schema Optimizer
================

Optimizes database schema for RAN performance data storage and querying.
Handles data type optimization, index creation, and performance tuning.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class SchemaOptimizer:
    """
    Optimizes database schema for efficient storage and querying of RAN data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define optimal data types for RAN data
        self.ran_type_mapping = {
            'cell_id': 'TEXT',
            'site_id': 'TEXT', 
            'sector_id': 'INTEGER',
            'enb_id': 'INTEGER',
            'gnb_id': 'INTEGER',
            'rsrp': 'REAL',
            'rsrq': 'REAL',
            'sinr': 'REAL',
            'throughput': 'REAL',
            'latency': 'REAL',
            'bler': 'REAL',
            'cqi': 'INTEGER',
            'timestamp': 'DATETIME',
            'date': 'DATE'
        }
        
        # Define columns that should be indexed for performance
        self.index_columns = [
            'cell_id', 'site_id', 'timestamp', 'date', 'hour'
        ]
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for efficient storage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        try:
            optimized_df = df.copy()
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Optimize integer columns
                if df[col].dtype in ['int64', 'int32']:
                    optimized_df[col] = self._optimize_integer_column(df[col])
                
                # Optimize float columns  
                elif df[col].dtype in ['float64', 'float32']:
                    optimized_df[col] = self._optimize_float_column(df[col])
                
                # Optimize string columns
                elif df[col].dtype == 'object':
                    optimized_df[col] = self._optimize_string_column(df[col], col_lower)
                
                # Handle datetime columns
                if any(pattern in col_lower for pattern in ['timestamp', 'date', 'time']):
                    optimized_df[col] = self._optimize_datetime_column(df[col])
            
            self.logger.info("DataFrame optimization completed")
            return optimized_df
            
        except Exception as e:
            self.logger.error(f"DataFrame optimization failed: {e}")
            return df
    
    def _optimize_integer_column(self, series: pd.Series) -> pd.Series:
        """
        Optimize integer column to smallest possible integer type.
        """
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
