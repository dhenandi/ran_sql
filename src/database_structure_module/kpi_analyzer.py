"""
KPI Analyzer
============

Analyzes KPI definitions and formulas to extract column dependencies
and create mappings for training data generation.
"""

import pandas as pd
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path
import logging


class KPIAnalyzer:
    """
    Analyzes KPI definitions and extracts column dependencies.
    """
    
    def __init__(self, kpi_csv_path: str):
        """
        Initialize KPI analyzer with KPI definition file.
        
        Args:
            kpi_csv_path: Path to KPI definition CSV file
        """
        self.kpi_csv_path = Path(kpi_csv_path)
        self.logger = logging.getLogger(__name__)
        self.kpi_data = None
        self.column_mappings = {}
        
        self._load_kpi_definitions()
    
    def _load_kpi_definitions(self):
        """Load KPI definitions from CSV file."""
        try:
            self.kpi_data = pd.read_csv(
                self.kpi_csv_path, 
                sep=';',
                encoding='utf-8'
            )
            self.logger.info(f"Loaded {len(self.kpi_data)} KPI definitions")
        except Exception as e:
            self.logger.error(f"Failed to load KPI definitions: {e}")
            self.kpi_data = pd.DataFrame()
    
    def extract_columns_from_formula(self, formula: str) -> Set[str]:
        """
        Extract column names from SQL formula.
        
        Args:
            formula: SQL formula string
            
        Returns:
            Set of column names found in formula
        """
        columns = set()
        
        # Pattern to match column names in SQL formulas
        # Matches: sum(columnName), avg(columnName), columnName
        patterns = [
            r'sum\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',  # sum(column)
            r'avg\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',  # avg(column)
            r'count\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',  # count(column)
            r'max\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',  # max(column)
            r'min\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',  # min(column)
            r'coalesce\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)',  # coalesce(column)
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:[+\-*/=<>]|\))',  # column in expressions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, formula, re.IGNORECASE)
            columns.update(matches)
        
        # Remove SQL keywords
        sql_keywords = {
            'cast', 'as', 'dec', 'nullif', 'sum', 'avg', 'count', 'max', 'min',
            'coalesce', 'from', 'where', 'group', 'by', 'order', 'having', 'select'
        }
        columns = {col for col in columns if col.lower() not in sql_keywords}
        
        return columns
    
    def analyze_kpi_mappings(self) -> Dict[str, Dict]:
        """
        Analyze all KPIs and create comprehensive mappings.
        
        Returns:
            Dict with KPI analysis results
        """
        if self.kpi_data is None or len(self.kpi_data) == 0:
            return {}
        
        mappings = {
            'kpis_by_category': {},
            'kpis_by_technology': {},
            'kpi_to_columns': {},
            'column_to_kpis': {},
            'kpi_descriptions': {},
            'kpi_units': {}
        }
        
        for _, row in self.kpi_data.iterrows():
            kpi_name = row['KPI_NAME']
            category = row['KPI_CATEGORY']
            tech = row['KPI_TECH']
            formula = str(row['KPI_FORMULA'])
            description = row['KPI_DESCRIPTION']
            unit = row['KPI_UNIT']
            table = row['KPI_TABLE']
            
            # Group by category
            if category not in mappings['kpis_by_category']:
                mappings['kpis_by_category'][category] = []
            mappings['kpis_by_category'][category].append(kpi_name)
            
            # Group by technology
            if tech not in mappings['kpis_by_technology']:
                mappings['kpis_by_technology'][tech] = []
            mappings['kpis_by_technology'][tech].append(kpi_name)
            
            # Extract columns from formula
            columns = self.extract_columns_from_formula(formula)
            mappings['kpi_to_columns'][kpi_name] = {
                'columns': list(columns),
                'table': table,
                'formula': formula
            }
            
            # Reverse mapping: column to KPIs
            for col in columns:
                if col not in mappings['column_to_kpis']:
                    mappings['column_to_kpis'][col] = []
                mappings['column_to_kpis'][col].append(kpi_name)
            
            # Store descriptions and units
            mappings['kpi_descriptions'][kpi_name] = description
            mappings['kpi_units'][kpi_name] = unit
        
        self.column_mappings = mappings
        return mappings
    
    def get_kpi_by_category(self, category: str) -> List[str]:
        """Get all KPIs in a specific category."""
        if not self.column_mappings:
            self.analyze_kpi_mappings()
        return self.column_mappings.get('kpis_by_category', {}).get(category, [])
    
    def get_kpi_by_technology(self, technology: str) -> List[str]:
        """Get all KPIs for a specific technology (2G/4G)."""
        if not self.column_mappings:
            self.analyze_kpi_mappings()
        return self.column_mappings.get('kpis_by_technology', {}).get(technology, [])
    
    def get_columns_for_kpi(self, kpi_name: str) -> List[str]:
        """Get all columns required for calculating a specific KPI."""
        if not self.column_mappings:
            self.analyze_kpi_mappings()
        kpi_info = self.column_mappings.get('kpi_to_columns', {}).get(kpi_name, {})
        return kpi_info.get('columns', [])
    
    def get_kpis_using_column(self, column_name: str) -> List[str]:
        """Get all KPIs that use a specific column."""
        if not self.column_mappings:
            self.analyze_kpi_mappings()
        return self.column_mappings.get('column_to_kpis', {}).get(column_name, [])
    
    def generate_kpi_summary(self) -> pd.DataFrame:
        """Generate a summary dataframe of all KPIs."""
        if self.kpi_data is None or len(self.kpi_data) == 0:
            return pd.DataFrame()
        
        summary = self.kpi_data[[
            'KPI_CATEGORY', 'KPI_TECH', 'KPI_NAME', 'KPI_UNIT'
        ]].copy()
        
        # Add column count
        summary['NUM_COLUMNS'] = summary['KPI_NAME'].apply(
            lambda x: len(self.get_columns_for_kpi(x))
        )
        
        return summary
    
    def export_mappings(self, output_path: str):
        """Export KPI mappings to JSON file."""
        if not self.column_mappings:
            self.analyze_kpi_mappings()
        
        import json
        with open(output_path, 'w') as f:
            json.dump(self.column_mappings, f, indent=2)
        
        self.logger.info(f"KPI mappings exported to {output_path}")
