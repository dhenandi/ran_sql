"""
Schema Mapper
=============

Maps natural language terms to database schema elements for RAN data.
Creates mappings between user queries and database columns/tables.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging


class SchemaMapper:
    """
    Maps natural language terms to database schema elements.
    """
    
    def __init__(self, schema_info: Dict):
        """
        Initialize the schema mapper.
        
        Args:
            schema_info: Database schema information from DatabaseAnalyzer
        """
        self.schema_info = schema_info
        self.logger = logging.getLogger(__name__)
        
        # Initialize mapping dictionaries
        self.term_to_column = {}
        self.term_to_table = {}
        self.synonym_mappings = {}
        
        # Load RAN domain vocabulary
        self._load_ran_vocabulary()
        
        # Build initial mappings
        self._build_schema_mappings()
    
    def _load_ran_vocabulary(self):
        """
        Load RAN-specific vocabulary and synonyms.
        """
        self.ran_vocabulary = {
            # Network elements
            'cell': ['cell', 'cells', 'cell_id', 'cellular'],
            'site': ['site', 'sites', 'site_id', 'location', 'base_station'],
            'sector': ['sector', 'sectors', 'sector_id'],
            'enb': ['enb', 'enodeb', 'enb_id', 'lte_base_station'],
            'gnb': ['gnb', 'gnodeb', 'gnb_id', '5g_base_station'],
            
            # KPIs
            'rsrp': ['rsrp', 'reference_signal_received_power', 'signal_power'],
            'rsrq': ['rsrq', 'reference_signal_received_quality', 'signal_quality'],
            'sinr': ['sinr', 'signal_to_interference_noise_ratio', 'snr'],
            'throughput': ['throughput', 'data_rate', 'speed', 'bandwidth'],
            'latency': ['latency', 'delay', 'response_time'],
            'bler': ['bler', 'block_error_rate', 'error_rate'],
            'cqi': ['cqi', 'channel_quality_indicator', 'channel_quality'],
            
            # Temporal terms
            'time': ['time', 'timestamp', 'datetime', 'when'],
            'date': ['date', 'day', 'daily'],
            'hour': ['hour', 'hourly', 'hr'],
            'minute': ['minute', 'min'],
            
            # Aggregations
            'average': ['average', 'avg', 'mean'],
            'maximum': ['maximum', 'max', 'highest', 'peak'],
            'minimum': ['minimum', 'min', 'lowest', 'bottom'],
            'count': ['count', 'number', 'total'],
            'sum': ['sum', 'total', 'aggregate']
        }
    
    def _build_schema_mappings(self):
        """
        Build mappings between natural language terms and schema elements.
        """
        tables = self.schema_info.get('tables', [])
        
        for table in tables:
            table_name = table['name']
            
            # Map table names
            self._map_table_name(table_name)
            
            # Map column names
            for column in table['columns']:
                self._map_column_name(table_name, column)
    
    def _map_table_name(self, table_name: str):
        """
        Create mappings for table names.
        """
        # Direct mapping
        self.term_to_table[table_name.lower()] = table_name
        
        # Extract key terms from table name
        terms = re.split(r'[_\-\s]+', table_name.lower())
        for term in terms:
            if len(term) > 2:  # Skip very short terms
                if term not in self.term_to_table:
                    self.term_to_table[term] = []
                if table_name not in self.term_to_table[term]:
                    self.term_to_table[term].append(table_name)
    
    def _map_column_name(self, table_name: str, column_name: str):
        """
        Create mappings for column names.
        """
        full_column = f"{table_name}.{column_name}"
        
        # Direct mapping
        self.term_to_column[column_name.lower()] = full_column
        
        # Extract terms from column name
        terms = re.split(r'[_\-\s]+', column_name.lower())
        for term in terms:
            if len(term) > 2:
                if term not in self.term_to_column:
                    self.term_to_column[term] = []
                if full_column not in self.term_to_column[term]:
                    self.term_to_column[term].append(full_column)
        
        # Map RAN vocabulary terms
        for concept, synonyms in self.ran_vocabulary.items():
            for synonym in synonyms:
                if synonym in column_name.lower():
                    if concept not in self.term_to_column:
                        self.term_to_column[concept] = []
                    if full_column not in self.term_to_column[concept]:
                        self.term_to_column[concept].append(full_column)
    
    def map_query_terms(self, query: str) -> Dict[str, List[str]]:
        """
        Map terms in a natural language query to database elements.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict: Mapping of identified terms to database elements
        """
        query_lower = query.lower()
        mappings = {
            'tables': [],
            'columns': [],
            'aggregations': [],
            'filters': [],
            'temporal': []
        }
        
        # Tokenize query
        tokens = re.findall(r'\b\w+\b', query_lower)
        
        # Map tokens to schema elements
        for token in tokens:
            # Check for table mappings
            if token in self.term_to_table:
                tables = self.term_to_table[token]
                if isinstance(tables, list):
                    mappings['tables'].extend(tables)
                else:
                    mappings['tables'].append(tables)
            
            # Check for column mappings
            if token in self.term_to_column:
                columns = self.term_to_column[token]
                if isinstance(columns, list):
                    mappings['columns'].extend(columns)
                else:
                    mappings['columns'].append(columns)
            
            # Check for aggregation functions
            if token in ['avg', 'average', 'mean']:
                mappings['aggregations'].append('AVG')
            elif token in ['max', 'maximum', 'highest']:
                mappings['aggregations'].append('MAX')
            elif token in ['min', 'minimum', 'lowest']:
                mappings['aggregations'].append('MIN')
            elif token in ['count', 'number']:
                mappings['aggregations'].append('COUNT')
            elif token in ['sum', 'total']:
                mappings['aggregations'].append('SUM')
        
        # Remove duplicates
        for key in mappings:
            mappings[key] = list(set(mappings[key]))
        
        return mappings
    
    def find_relevant_columns(self, concepts: List[str], table_name: Optional[str] = None) -> List[str]:
        """
        Find database columns relevant to given concepts.
        
        Args:
            concepts: List of concept terms
            table_name: Optional table name to restrict search
            
        Returns:
            List[str]: List of relevant column names
        """
        relevant_columns = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Direct lookup
            if concept_lower in self.term_to_column:
                columns = self.term_to_column[concept_lower]
                if isinstance(columns, list):
                    relevant_columns.extend(columns)
                else:
                    relevant_columns.append(columns)
            
            # Fuzzy matching for RAN terms
            for ran_concept, synonyms in self.ran_vocabulary.items():
                if concept_lower in synonyms or any(syn in concept_lower for syn in synonyms):
                    if ran_concept in self.term_to_column:
                        columns = self.term_to_column[ran_concept]
                        if isinstance(columns, list):
                            relevant_columns.extend(columns)
                        else:
                            relevant_columns.append(columns)
        
        # Filter by table if specified
        if table_name:
            relevant_columns = [col for col in relevant_columns if col.startswith(f"{table_name}.")]
        
        return list(set(relevant_columns))
    
    def get_column_suggestions(self, partial_term: str, limit: int = 10) -> List[Dict]:
        """
        Get column suggestions based on partial term matching.
        
        Args:
            partial_term: Partial column name or concept
            limit: Maximum number of suggestions
            
        Returns:
            List[Dict]: List of suggestions with metadata
        """
        suggestions = []
        partial_lower = partial_term.lower()
        
        # Search in direct column mappings
        for term, columns in self.term_to_column.items():
            if partial_lower in term:
                if isinstance(columns, list):
                    for col in columns:
                        suggestions.append({
                            'column': col,
                            'match_term': term,
                            'match_type': 'partial',
                            'confidence': len(partial_lower) / len(term)
                        })
                else:
                    suggestions.append({
                        'column': columns,
                        'match_term': term,
                        'match_type': 'partial',
                        'confidence': len(partial_lower) / len(term)
                    })
        
        # Sort by confidence and limit
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions[:limit]
    
    def validate_mapping(self, term: str, suggested_column: str) -> float:
        """
        Validate the quality of a term-to-column mapping.
        
        Args:
            term: Natural language term
            suggested_column: Database column name
            
        Returns:
            float: Confidence score (0-1)
        """
        term_lower = term.lower()
        column_lower = suggested_column.lower()
        
        # Exact match
        if term_lower == column_lower.split('.')[-1]:
            return 1.0
        
        # Partial match
        if term_lower in column_lower:
            return 0.8
        
        # RAN vocabulary match
        for concept, synonyms in self.ran_vocabulary.items():
            if term_lower in synonyms:
                if concept in column_lower:
                    return 0.7
        
        # Fuzzy matching based on common substrings
        common_length = len(set(term_lower) & set(column_lower))
        max_length = max(len(term_lower), len(column_lower))
        
        if max_length > 0:
            return common_length / max_length
        
        return 0.0
    
    def export_mappings(self, output_path: str) -> bool:
        """
        Export schema mappings to file for reuse.
        
        Args:
            output_path: Path to save mappings
            
        Returns:
            bool: True if successful
        """
        try:
            mappings_data = {
                'term_to_column': self.term_to_column,
                'term_to_table': self.term_to_table,
                'ran_vocabulary': self.ran_vocabulary
            }
            
            with open(output_path, 'w') as f:
                json.dump(mappings_data, f, indent=2)
            
            self.logger.info(f"Schema mappings exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export mappings: {e}")
            return False
