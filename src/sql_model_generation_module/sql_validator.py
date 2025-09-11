"""
SQL Validator
=============

Validates and optimizes generated SQL queries.
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Any


class SQLValidator:
    """
    Validates SQL queries for correctness and safety.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the SQL validator.
        
        Args:
            db_path: Path to database for validation
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def validate_sql(self, sql: str, schema_info: Dict = None) -> Dict[str, Any]:
        """
        Validate SQL query comprehensively.
        
        Args:
            sql: SQL query to validate
            schema_info: Database schema information
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Syntax validation
        syntax_results = self._validate_syntax(sql)
        results['errors'].extend(syntax_results.get('errors', []))
        
        # Security validation
        security_results = self._validate_security(sql)
        results['errors'].extend(security_results.get('errors', []))
        results['warnings'].extend(security_results.get('warnings', []))
        
        # Schema validation
        if schema_info:
            schema_results = self._validate_schema(sql, schema_info)
            results['errors'].extend(schema_results.get('errors', []))
            results['warnings'].extend(schema_results.get('warnings', []))
        
        # Performance suggestions
        performance_results = self._analyze_performance(sql)
        results['suggestions'].extend(performance_results.get('suggestions', []))
        
        results['is_valid'] = len(results['errors']) == 0
        
        return results
    
    def _validate_syntax(self, sql: str) -> Dict[str, List[str]]:
        """
        Validate SQL syntax.
        
        Args:
            sql: SQL query
            
        Returns:
            Dict: Syntax validation results
        """
        errors = []
        
        if not sql or not sql.strip():
            errors.append("Empty SQL query")
            return {'errors': errors}
        
        # Try to parse with SQLite
        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            except sqlite3.Error as e:
                errors.append(f"SQL syntax error: {str(e)}")
        
        return {'errors': errors}
    
    def _validate_security(self, sql: str) -> Dict[str, List[str]]:
        """
        Validate SQL for security issues.
        
        Args:
            sql: SQL query
            
        Returns:
            Dict: Security validation results
        """
        errors = []
        warnings = []
        
        sql_upper = sql.upper()
        
        # Check for dangerous operations
        dangerous_ops = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for op in dangerous_ops:
            if op in sql_upper:
                errors.append(f"Dangerous operation not allowed: {op}")
        
        # Check for potential injection patterns
        injection_patterns = ['--', '/*', '*/', 'UNION', 'OR 1=1', 'OR 1 = 1']
        for pattern in injection_patterns:
            if pattern in sql_upper:
                warnings.append(f"Potential SQL injection pattern detected: {pattern}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_schema(self, sql: str, schema_info: Dict) -> Dict[str, List[str]]:
        """
        Validate SQL against database schema.
        
        Args:
            sql: SQL query
            schema_info: Database schema information
            
        Returns:
            Dict: Schema validation results
        """
        errors = []
        warnings = []
        
        # Extract table and column references from SQL
        # This is a simplified implementation
        tables = self._extract_table_references(sql)
        columns = self._extract_column_references(sql)
        
        # Validate table existence
        available_tables = [table['name'] for table in schema_info.get('tables', [])]
        for table in tables:
            if table not in available_tables:
                errors.append(f"Table '{table}' does not exist")
        
        # Validate column existence (simplified)
        # In a real implementation, this would be more sophisticated
        
        return {'errors': errors, 'warnings': warnings}
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """
        Extract table names referenced in SQL.
        
        Args:
            sql: SQL query
            
        Returns:
            List[str]: Table names
        """
        # Simplified extraction - would need more robust parsing
        import re
        
        # Look for FROM and JOIN clauses
        from_pattern = r'FROM\s+(\w+)'
        join_pattern = r'JOIN\s+(\w+)'
        
        tables = []
        tables.extend(re.findall(from_pattern, sql, re.IGNORECASE))
        tables.extend(re.findall(join_pattern, sql, re.IGNORECASE))
        
        return list(set(tables))
    
    def _extract_column_references(self, sql: str) -> List[str]:
        """
        Extract column names referenced in SQL.
        
        Args:
            sql: SQL query
            
        Returns:
            List[str]: Column names
        """
        # Simplified extraction
        import re
        
        # This would need much more sophisticated parsing in reality
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        columns = []
        if match:
            select_clause = match.group(1)
            # Split by comma and clean up
            column_parts = [col.strip() for col in select_clause.split(',')]
            for part in column_parts:
                if '(' not in part and part != '*':  # Skip functions and *
                    columns.append(part.split()[-1])  # Get the column name
        
        return columns
    
    def _analyze_performance(self, sql: str) -> Dict[str, List[str]]:
        """
        Analyze SQL for performance issues and suggestions.
        
        Args:
            sql: SQL query
            
        Returns:
            Dict: Performance analysis results
        """
        suggestions = []
        
        sql_upper = sql.upper()
        
        # Check for SELECT *
        if 'SELECT *' in sql_upper:
            suggestions.append("Consider selecting specific columns instead of using SELECT *")
        
        # Check for missing WHERE clause in large tables
        if 'WHERE' not in sql_upper and 'LIMIT' not in sql_upper:
            suggestions.append("Consider adding WHERE clause or LIMIT to avoid scanning entire table")
        
        # Check for inefficient LIKE patterns
        if 'LIKE \'%' in sql_upper:
            suggestions.append("Leading wildcard in LIKE clause may cause poor performance")
        
        return {'suggestions': suggestions}
    
    def optimize_sql(self, sql: str, schema_info: Dict = None) -> str:
        """
        Optimize SQL query for better performance.
        
        Args:
            sql: Original SQL query
            schema_info: Database schema information
            
        Returns:
            str: Optimized SQL query
        """
        # Implementation placeholder for SQL optimization
        # This would apply various optimization techniques
        return sql
