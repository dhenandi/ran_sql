"""
Database Analyzer
=================

Analyzes SQLite database structure to understand RAN data schema,
relationships, and generate metadata for model training.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging


class DatabaseAnalyzer:
    """
    Analyzes database structure and generates schema documentation
    for RAN performance data.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the database analyzer.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.schema_info = {}
        
        # RAN domain knowledge
        self.ran_entity_types = {
            'network_elements': ['cell', 'site', 'sector', 'enb', 'gnb', 'base_station'],
            'kpis': ['rsrp', 'rsrq', 'sinr', 'throughput', 'latency', 'bler', 'cqi'],
            'temporal': ['timestamp', 'date', 'time', 'hour', 'minute', 'day'],
            'identifiers': ['id', 'identifier', 'code', 'name'],
            'measurements': ['value', 'measurement', 'metric', 'counter']
        }
    
    def analyze_database(self) -> Dict[str, Any]:
        """
        Perform comprehensive database analysis.
        
        Returns:
            Dict: Complete database analysis including tables, columns, relationships
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                self.schema_info = {
                    'tables': self._get_table_info(conn),
                    'columns': self._get_column_info(conn),
                    'relationships': self._identify_relationships(conn),
                    'data_samples': self._get_data_samples(conn),
                    'statistics': self._get_table_statistics(conn),
                    'ran_entities': self._map_ran_entities(conn)
                }
            
            self.logger.info("Database analysis completed successfully")
            return self.schema_info
            
        except Exception as e:
            self.logger.error(f"Database analysis failed: {e}")
            return {}
    
    def _get_table_info(self, conn: sqlite3.Connection) -> List[Dict]:
        """
        Get information about all tables in the database.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        table_info = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            table_info.append({
                'name': table_name,
                'columns': [col[1] for col in columns],
                'column_types': {col[1]: col[2] for col in columns},
                'row_count': row_count
            })
        
        return table_info
    
    def _get_column_info(self, conn: sqlite3.Connection) -> Dict[str, List[Dict]]:
        """
        Get detailed information about columns in each table.
        """
        column_info = {}
        
        for table in self.schema_info.get('tables', []):
            table_name = table['name']
            column_info[table_name] = []
            
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # Get column statistics
                stats = self._get_column_statistics(conn, table_name, col_name, col_type)
                
                column_info[table_name].append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': bool(col[3]),
                    'primary_key': bool(col[5]),
                    'statistics': stats,
                    'ran_entity_type': self._classify_ran_entity(col_name)
                })
        
        return column_info
    
    def _get_column_statistics(self, conn: sqlite3.Connection, table_name: str, 
                              col_name: str, col_type: str) -> Dict:
        """
        Get statistical information about a column.
        """
        cursor = conn.cursor()
        stats = {}
        
        try:
            # Basic statistics
            cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
            stats['unique_count'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL")
            stats['null_count'] = cursor.fetchone()[0]
            
            # Type-specific statistics
            if col_type.upper() in ['INTEGER', 'REAL', 'NUMERIC']:
                cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table_name}")
                min_val, max_val, avg_val = cursor.fetchone()
                stats.update({
                    'min': min_val,
                    'max': max_val,
                    'avg': avg_val
                })
            
            # Sample values
            cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10")
            stats['sample_values'] = [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.warning(f"Could not get statistics for {table_name}.{col_name}: {e}")
        
        return stats
    
    def _identify_relationships(self, conn: sqlite3.Connection) -> List[Dict]:
        """
        Identify potential relationships between tables based on column names and data.
        """
        relationships = []
        tables = self.schema_info.get('tables', [])
        
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                # Find potential foreign key relationships
                common_columns = set(table1['columns']) & set(table2['columns'])
                
                for col in common_columns:
                    # Skip obvious non-FK columns
                    if col.lower() in ['timestamp', 'date', 'value', 'measurement']:
                        continue
                    
                    # Check if values overlap significantly
                    if self._check_column_overlap(conn, table1['name'], table2['name'], col):
                        relationships.append({
                            'table1': table1['name'],
                            'table2': table2['name'],
                            'column': col,
                            'type': 'potential_foreign_key',
                            'confidence': 0.8  # This could be calculated based on overlap
                        })
        
        return relationships
    
    def _check_column_overlap(self, conn: sqlite3.Connection, 
                             table1: str, table2: str, column: str) -> bool:
        """
        Check if there's significant overlap in values between two columns.
        """
        try:
            cursor = conn.cursor()
            
            # Get distinct values from both tables
            cursor.execute(f"SELECT DISTINCT {column} FROM {table1} LIMIT 1000")
            values1 = set(row[0] for row in cursor.fetchall())
            
            cursor.execute(f"SELECT DISTINCT {column} FROM {table2} LIMIT 1000")
            values2 = set(row[0] for row in cursor.fetchall())
            
            # Calculate overlap ratio
            if len(values1) == 0 or len(values2) == 0:
                return False
            
            overlap = len(values1 & values2)
            overlap_ratio = overlap / min(len(values1), len(values2))
            
            return overlap_ratio > 0.3  # 30% overlap threshold
            
        except Exception:
            return False
    
    def _get_data_samples(self, conn: sqlite3.Connection, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Get sample data from each table for analysis.
        """
        samples = {}
        
        for table in self.schema_info.get('tables', []):
            table_name = table['name']
            
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
                samples[table_name] = df.to_dict('records')
            except Exception as e:
                self.logger.warning(f"Could not get samples from {table_name}: {e}")
                samples[table_name] = []
        
        return samples
    
    def _get_table_statistics(self, conn: sqlite3.Connection) -> Dict[str, Dict]:
        """
        Get overall statistics for each table.
        """
        statistics = {}
        
        for table in self.schema_info.get('tables', []):
            table_name = table['name']
            
            try:
                cursor = conn.cursor()
                
                # Row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Column count
                column_count = len(table['columns'])
                
                statistics[table_name] = {
                    'row_count': row_count,
                    'column_count': column_count,
                    'estimated_size_mb': (row_count * column_count * 8) / (1024 * 1024)  # Rough estimate
                }
                
            except Exception as e:
                self.logger.warning(f"Could not get statistics for {table_name}: {e}")
        
        return statistics
    
    def _map_ran_entities(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """
        Map columns to RAN entity types based on naming patterns and content.
        """
        ran_mapping = {entity_type: [] for entity_type in self.ran_entity_types}
        
        for table in self.schema_info.get('tables', []):
            for col_name in table['columns']:
                entity_type = self._classify_ran_entity(col_name)
                if entity_type and entity_type in ran_mapping:
                    full_name = f"{table['name']}.{col_name}"
                    if full_name not in ran_mapping[entity_type]:
                        ran_mapping[entity_type].append(full_name)
        
        return ran_mapping
    
    def _classify_ran_entity(self, column_name: str) -> Optional[str]:
        """
        Classify a column name into RAN entity types.
        """
        col_lower = column_name.lower()
        
        for entity_type, patterns in self.ran_entity_types.items():
            for pattern in patterns:
                if pattern in col_lower:
                    return entity_type
        
        return None
    
    def export_schema(self, output_path: str, format: str = 'json') -> bool:
        """
        Export schema analysis to file.
        
        Args:
            output_path: Path to save the schema analysis
            format: Output format ('json', 'yaml', 'csv')
            
        Returns:
            bool: True if successful
        """
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.schema_info, f, indent=2, default=str)
            
            self.logger.info(f"Schema exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export schema: {e}")
            return False
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database schema for quick reference.
        """
        if not self.schema_info:
            return {}
        
        tables = self.schema_info.get('tables', [])
        total_rows = sum(table['row_count'] for table in tables)
        total_columns = sum(len(table['columns']) for table in tables)
        
        return {
            'table_count': len(tables),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'table_names': [table['name'] for table in tables],
            'ran_entity_distribution': {
                entity_type: len(columns) 
                for entity_type, columns in self.schema_info.get('ran_entities', {}).items()
            }
        }
