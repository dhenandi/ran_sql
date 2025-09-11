"""
RAN CSV Importer
================

Specialized CSV importer for Radio Access Network (RAN) performance data.
Handles 2G and 4G network performance metrics with proper schema mapping.
"""

import sqlite3
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from .data_validator import DataValidator
from .schema_optimizer import SchemaOptimizer


class RANCSVImporter:
    """
    Specialized importer for RAN CSV files (2G and 4G network performance data).
    """
    
    # Common columns across both 2G and 4G files
    COMMON_COLUMNS = ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']
    
    # RAN technology specific column patterns
    RAN_2G_PATTERNS = ['ccalls', 'ccongs', 'cndrop', 'cnrelcong', 'cs12', 'cs14', 
                       'iaulrel', 'ldis', 'mc19', 'msest', 'numtch', 'othulrel',
                       'preempt', 'sumeo', 'sumo', 'tava', 'tfc', 'tfnce', 'tftra',
                       'thc', 'thnce', 'thtra', 'tnuch', 'pmcount_2g', 'numtrx',
                       'bcch', 'thnrel', 'tass', 'tfnrel']
    
    RAN_4G_PATTERNS = ['pm_count', 'pmcell', 'pmera', 'pmho', 'pmpdcp', 'pmprb',
                       'pmrrc', 'pms1sig', 'pmuethp', 'pm_rrcconn']
    
    def __init__(self, db_path: str, log_level: str = "INFO"):
        """
        Initialize the RAN CSV importer.
        
        Args:
            db_path: Path to the SQLite database file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.db_path = Path(db_path)
        self.validator = DataValidator()
        self.optimizer = SchemaOptimizer()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def detect_ran_technology(self, csv_path: str) -> str:
        """
        Detect whether the CSV contains 2G or 4G data based on column names.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            str: '2G', '4G', or 'UNKNOWN'
        """
        try:
            # Read just the header
            df_header = pd.read_csv(csv_path, nrows=0)
            columns = set(df_header.columns.str.lower())
            
            # Count matches for each technology
            ran_2g_matches = sum(1 for pattern in self.RAN_2G_PATTERNS 
                               if any(pattern in col for col in columns))
            ran_4g_matches = sum(1 for pattern in self.RAN_4G_PATTERNS 
                               if any(pattern in col for col in columns))
            
            if ran_2g_matches > ran_4g_matches:
                return "2G"
            elif ran_4g_matches > ran_2g_matches:
                return "4G"
            else:
                return "UNKNOWN"
                
        except Exception as e:
            self.logger.error(f"Error detecting RAN technology: {e}")
            return "UNKNOWN"
    
    def import_ran_csv(self, csv_path: str, table_name: Optional[str] = None) -> bool:
        """
        Import a RAN CSV file with automatic technology detection and schema optimization.
        
        Args:
            csv_path: Path to the CSV file
            table_name: Optional table name (auto-generated if not provided)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                self.logger.error(f"CSV file not found: {csv_path}")
                return False
            
            # Detect RAN technology
            ran_tech = self.detect_ran_technology(csv_path)
            self.logger.info(f"Detected RAN technology: {ran_tech}")
            
            # Generate table name if not provided
            if table_name is None:
                table_name = f"ran_{ran_tech.lower()}_{csv_file.stem.lower()}"
            
            self.logger.info(f"Starting import of {csv_path} to table {table_name}")
            
            # Read CSV file with proper data types
            df = self._read_ran_csv(csv_path, ran_tech)
            self.logger.info(f"Loaded {len(df)} rows from {ran_tech} CSV")
            
            # Optimize schema for RAN data
            optimized_df = self.optimizer.optimize_ran_dataframe(df, ran_tech)
            
            # Create connection and import data
            with sqlite3.connect(self.db_path) as conn:
                # Create table with proper schema
                self._create_ran_table(conn, table_name, optimized_df, ran_tech)
                
                # Insert data in chunks for large datasets
                chunk_size = 10000
                for i in range(0, len(optimized_df), chunk_size):
                    chunk = optimized_df.iloc[i:i+chunk_size]
                    chunk.to_sql(table_name, conn, if_exists='append', index=False)
                    self.logger.info(f"Inserted chunk {i//chunk_size + 1}: {len(chunk)} rows")
                
                # Create indexes for better query performance
                self._create_ran_indexes(conn, table_name, ran_tech)
            
            self.logger.info(f"Successfully imported {table_name} with {len(optimized_df)} rows")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing RAN CSV: {e}")
            return False
    
    def _read_ran_csv(self, csv_path: str, ran_tech: str) -> pd.DataFrame:
        """
        Read RAN CSV with appropriate data type handling.
        """
        try:
            # Read CSV with minimal type inference to avoid issues
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Convert timestamp column
            if 'last_update' in df.columns:
                df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading RAN CSV: {e}")
            raise
    
    def _create_ran_table(self, conn: sqlite3.Connection, table_name: str, 
                         df: pd.DataFrame, ran_tech: str) -> None:
        """
        Create a properly structured table for RAN data.
        """
        # Generate CREATE TABLE statement
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        
        column_defs = []
        for col in df.columns:
            col_type = self._get_sqlite_type(df[col].dtype)
            if col == 'id':
                column_defs.append(f"  {col} {col_type} PRIMARY KEY")
            else:
                column_defs.append(f"  {col} {col_type}")
        
        create_sql += ",\n".join(column_defs)
        create_sql += "\n);"
        
        self.logger.debug(f"Creating table with SQL: {create_sql}")
        conn.execute(create_sql)
        conn.commit()
    
    def _create_ran_indexes(self, conn: sqlite3.Connection, table_name: str, ran_tech: str) -> None:
        """
        Create indexes optimized for RAN data queries.
        """
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_region ON {table_name}(region)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_siteid ON {table_name}(siteid)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_last_update ON {table_name}(last_update)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
                self.logger.debug(f"Created index: {index_sql}")
            except Exception as e:
                self.logger.warning(f"Could not create index: {e}")
        
        conn.commit()
    
    def _get_sqlite_type(self, pandas_dtype) -> str:
        """
        Map pandas dtypes to SQLite types.
        """
        dtype_str = str(pandas_dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'REAL'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            return 'TEXT'
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get information about a table in the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                return {
                    'table_name': table_name,
                    'columns': [{'name': col[1], 'type': col[2]} for col in columns],
                    'row_count': row_count
                }
                
        except Exception as e:
            self.logger.error(f"Error getting table info: {e}")
            return {}
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                return tables
                
        except Exception as e:
            self.logger.error(f"Error listing tables: {e}")
            return []
    
    def import_all_ran_files(self, data_dir: str) -> Dict[str, bool]:
        """
        Import all RAN CSV files from a directory.
        """
        results = {}
        data_path = Path(data_dir)
        
        if not data_path.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return results
        
        # Find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files to import")
        
        for csv_file in csv_files:
            try:
                self.logger.info(f"Processing: {csv_file.name}")
                success = self.import_ran_csv(str(csv_file))
                results[csv_file.name] = success
                
                if success:
                    self.logger.info(f"✅ Successfully imported {csv_file.name}")
                else:
                    self.logger.error(f"❌ Failed to import {csv_file.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing {csv_file.name}: {e}")
                results[csv_file.name] = False
        
        return results
