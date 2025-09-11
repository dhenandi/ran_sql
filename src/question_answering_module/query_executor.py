"""
Query Executor
==============

Safely executes SQL queries against the database.
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import time


class QueryExecutor:
    """
    Executes SQL queries safely with timeout and result limits.
    """
    
    def __init__(self, db_path: str, timeout: int = 30, max_results: int = 1000):
        """
        Initialize the query executor.
        
        Args:
            db_path: Path to the SQLite database
            timeout: Query timeout in seconds
            max_results: Maximum number of results to return
        """
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.max_results = max_results
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, sql: str) -> Optional[List[Dict]]:
        """
        Execute SQL query and return results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Optional[List[Dict]]: Query results or None if failed
        """
        if not self.db_path.exists():
            self.logger.error(f"Database not found: {self.db_path}")
            return None
        
        try:
            start_time = time.time()
            
            with sqlite3.connect(str(self.db_path), timeout=self.timeout) as conn:
                conn.row_factory = sqlite3.Row  # Enable column name access
                cursor = conn.cursor()
                
                # Add LIMIT if not present and query is SELECT
                limited_sql = self._add_limit_if_needed(sql)
                
                self.logger.info(f"Executing SQL: {limited_sql}")
                
                cursor.execute(limited_sql)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = [dict(row) for row in rows]
                
                execution_time = time.time() - start_time
                self.logger.info(f"Query executed in {execution_time:.3f}s, {len(results)} rows returned")
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during query execution: {e}")
            return None
    
    def _add_limit_if_needed(self, sql: str) -> str:
        """
        Add LIMIT clause to SELECT queries if not present.
        
        Args:
            sql: Original SQL query
            
        Returns:
            str: SQL query with LIMIT clause if needed
        """
        sql_upper = sql.upper().strip()
        
        # Only add LIMIT to SELECT queries
        if not sql_upper.startswith('SELECT'):
            return sql
        
        # Check if LIMIT already exists
        if 'LIMIT' in sql_upper:
            return sql
        
        # Add LIMIT clause
        return f"{sql.rstrip(';')} LIMIT {self.max_results}"
    
    def execute_query_with_stats(self, sql: str) -> Dict[str, Any]:
        """
        Execute query and return results with execution statistics.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dict: Results with execution metadata
        """
        start_time = time.time()
        results = self.execute_query(sql)
        execution_time = time.time() - start_time
        
        return {
            'results': results,
            'success': results is not None,
            'execution_time': execution_time,
            'row_count': len(results) if results else 0,
            'limited': self.max_results if results and len(results) == self.max_results else None
        }
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            with sqlite3.connect(str(self.db_path), timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            self.logger.info("Database connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """
        Get information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Optional[Dict]: Table information
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                if not columns:
                    return None
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                return {
                    'table_name': table_name,
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'nullable': not col[3],
                            'primary_key': bool(col[5])
                        }
                        for col in columns
                    ],
                    'row_count': row_count
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_name}: {e}")
            return None
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database.
        
        Returns:
            List[str]: Table names
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to list tables: {e}")
            return []
