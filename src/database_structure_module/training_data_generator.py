"""
Training Data Generator
=======================

Generates training datasets for NER and SQL generation models
based on database schema analysis and RAN domain knowledge.
"""

import json
import random
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import logging


class TrainingDataGenerator:
    """
    Generates training data for NER and SQL generation models.
    """
    
    def __init__(self, schema_info: Dict, db_path: str, kpi_analyzer=None):
        """
        Initialize the training data generator.
        
        Args:
            schema_info: Database schema information
            db_path: Path to the SQLite database
            kpi_analyzer: Optional KPIAnalyzer instance for KPI-aware generation
        """
        self.schema_info = schema_info
        self.db_path = Path(db_path)
        self.kpi_analyzer = kpi_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Load RAN-specific templates and patterns
        self._load_query_templates()
        self._load_entity_patterns()
        self._load_kpi_templates()
    
    def _load_query_templates(self):
        """
        Load natural language query templates for RAN data.
        """
        self.query_templates = {
            'single_kpi': [
                "What is the average {kpi} for {entity}?",
                "Show me the {kpi} values for {entity}",
                "Get the maximum {kpi} from {entity}",
                "What is the minimum {kpi} in {entity}?",
                "Display {kpi} measurements for {entity}"
            ],
            'temporal': [
                "What was the {kpi} on {date}?",
                "Show {kpi} for the last {time_period}",
                "Get {kpi} values between {start_time} and {end_time}",
                "What is the hourly {kpi} trend?",
                "Display daily {kpi} statistics"
            ],
            'comparison': [
                "Compare {kpi} between {entity1} and {entity2}",
                "Which {entity} has the highest {kpi}?",
                "Show {kpi} differences across {entity_group}",
                "Rank {entity_group} by {kpi}",
                "Find {entity} with {kpi} above {threshold}"
            ],
            'aggregation': [
                "Count the number of {entity} with {condition}",
                "Sum all {kpi} values for {entity}",
                "Calculate average {kpi} per {grouping_entity}",
                "Get total {measurement} across all {entity}",
                "Show {kpi} distribution by {category}"
            ],
            'filtering': [
                "Show {kpi} where {condition}",
                "Get {entity} with {kpi} greater than {value}",
                "Find {entity} where {kpi} is between {min_val} and {max_val}",
                "Display {entity} with poor {kpi}",
                "List {entity} having {condition}"
            ]
        }
    
    def _load_entity_patterns(self):
        """
        Load patterns for entity recognition in RAN domain.
        """
        self.entity_patterns = {
            'CELL_ID': [
                r'\b[A-Z0-9]{6,12}\b',  # Cell ID pattern
                r'\bcell[_\s]*\w+\b',
                r'\b\w*cell\w*\b'
            ],
            'SITE_ID': [
                r'\bsite[_\s]*\w+\b',
                r'\b\w*site\w*\b',
                r'\bBTS[_\s]*\w+\b'
            ],
            'KPI_NAME': [
                r'\brsrp\b', r'\brsrq\b', r'\bsinr\b',
                r'\bthroughput\b', r'\blatency\b', r'\bbler\b',
                r'\bcqi\b', r'\bpci\b'
            ],
            'DATE_TIME': [
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{2}/\d{2}/\d{4}\b',
                r'\b\d{2}:\d{2}:\d{2}\b',
                r'\byesterday\b', r'\btoday\b', r'\blast week\b'
            ],
            'NUMERIC_VALUE': [
                r'\b-?\d+\.?\d*\b',
                r'\b\d+%\b',
                r'\bgreater than \d+\b',
                r'\bless than \d+\b'
            ],
            'AGGREGATION': [
                r'\baverage\b', r'\bmean\b', r'\bmax\b', r'\bmaximum\b',
                r'\bmin\b', r'\bminimum\b', r'\bcount\b', r'\bsum\b'
            ]
        }
    
    def _load_kpi_templates(self):
        """
        Load KPI-specific query templates if KPI analyzer is available.
        """
        if not self.kpi_analyzer:
            return
        
        # Add KPI-specific templates
        self.kpi_query_templates = {
            'kpi_categorical': [
                "What is the {kpi} for {tech} network?",
                "Show me {kpi} statistics",
                "Calculate {kpi} across all sites",
                "Get {kpi} measurements for {region}",
                "Display {kpi} by {category}"
            ],
            'kpi_comparison': [
                "Compare {kpi} between 2G and 4G",
                "Which technology has better {kpi}?",
                "Show {kpi} differences across regions",
                "{kpi} performance comparison"
            ],
            'kpi_threshold': [
                "Find sites where {kpi} is below {threshold}",
                "Show cells with {kpi} exceeding {threshold}",
                "Alert when {kpi} drops below {threshold}",
                "Identify poor {kpi} performance"
            ]
        }
    
    def generate_ner_training_data(self, num_samples: int = 1000) -> List[Dict]:
        """
        Generate training data for Named Entity Recognition.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            List[Dict]: Training samples with entities labeled
        """
        training_data = []
        
        # Get available entities from schema
        entities = self._extract_schema_entities()
        
        for _ in range(num_samples):
            # Select random template and entity combination
            template_type = random.choice(list(self.query_templates.keys()))
            template = random.choice(self.query_templates[template_type])
            
            # Fill template with actual entities
            filled_query, entities_in_query = self._fill_template(template, entities)
            
            # Create NER annotations
            ner_annotations = self._create_ner_annotations(filled_query, entities_in_query)
            
            training_data.append({
                'text': filled_query,
                'entities': ner_annotations,
                'template_type': template_type
            })
        
        self.logger.info(f"Generated {len(training_data)} NER training samples")
        return training_data
    
    def generate_sql_training_data(self, num_samples: int = 1000) -> List[Dict]:
        """
        Generate training data for SQL generation.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            List[Dict]: Training samples with query-SQL pairs
        """
        training_data = []
        
        # Get available tables and columns
        tables = self.schema_info.get('tables', [])
        
        for _ in range(num_samples):
            # Select random table and template
            table = random.choice(tables)
            template_type = random.choice(list(self.query_templates.keys()))
            
            # Generate natural language query and corresponding SQL
            nl_query, sql_query = self._generate_query_sql_pair(table, template_type)
            
            if nl_query and sql_query:
                training_data.append({
                    'natural_language': nl_query,
                    'sql': sql_query,
                    'table': table['name'],
                    'template_type': template_type
                })
        
        self.logger.info(f"Generated {len(training_data)} SQL training samples")
        return training_data
    
    def _extract_schema_entities(self) -> Dict[str, List[str]]:
        """
        Extract entities from database schema for template filling.
        """
        entities = {
            'tables': [],
            'columns': [],
            'kpis': [],
            'identifiers': [],
            'temporal_columns': []
        }
        
        ran_entities = self.schema_info.get('ran_entities', {})
        
        # Extract from RAN entity mapping
        for entity_type, columns in ran_entities.items():
            if entity_type == 'kpis':
                entities['kpis'].extend([col.split('.')[-1] for col in columns])
            elif entity_type == 'identifiers':
                entities['identifiers'].extend([col.split('.')[-1] for col in columns])
            elif entity_type == 'temporal':
                entities['temporal_columns'].extend([col.split('.')[-1] for col in columns])
        
        # Extract table and column names
        for table in self.schema_info.get('tables', []):
            entities['tables'].append(table['name'])
            entities['columns'].extend(table['columns'])
        
        return entities
    
    def _fill_template(self, template: str, entities: Dict[str, List[str]]) -> Tuple[str, Dict]:
        """
        Fill a query template with actual entities from the schema.
        
        Args:
            template: Query template with placeholders
            entities: Available entities from schema
            
        Returns:
            Tuple: (filled_query, entities_used)
        """
        filled_query = template
        entities_used = {}
        
        # Define placeholder mappings
        placeholder_mappings = {
            '{kpi}': entities.get('kpis', []),
            '{entity}': entities.get('identifiers', []),
            '{table}': entities.get('tables', []),
            '{column}': entities.get('columns', []),
            '{date}': ['2023-01-01', 'yesterday', 'last week'],
            '{time_period}': ['hour', 'day', 'week', 'month'],
            '{value}': ['100', '50', '200', '75'],
            '{threshold}': ['10', '20', '50', '100']
        }
        
        # Fill placeholders
        for placeholder, options in placeholder_mappings.items():
            if placeholder in filled_query and options:
                selected_value = random.choice(options)
                filled_query = filled_query.replace(placeholder, selected_value)
                entities_used[placeholder] = selected_value
        
        return filled_query, entities_used
    
    def _create_ner_annotations(self, text: str, entities_used: Dict) -> List[Dict]:
        """
        Create NER annotations for a filled query.
        
        Args:
            text: The filled query text
            entities_used: Entities that were filled in the template
            
        Returns:
            List[Dict]: Entity annotations with start, end, and label
        """
        annotations = []
        
        # Map entity types to NER labels
        entity_type_mapping = {
            '{kpi}': 'KPI_NAME',
            '{entity}': 'IDENTIFIER',
            '{table}': 'TABLE_NAME',
            '{column}': 'COLUMN_NAME',
            '{date}': 'DATE_TIME',
            '{value}': 'NUMERIC_VALUE',
            '{threshold}': 'NUMERIC_VALUE'
        }
        
        for placeholder, value in entities_used.items():
            if placeholder in entity_type_mapping:
                # Find the position of the value in the text
                start_pos = text.find(value)
                if start_pos != -1:
                    end_pos = start_pos + len(value)
                    annotations.append({
                        'start': start_pos,
                        'end': end_pos,
                        'label': entity_type_mapping[placeholder],
                        'text': value
                    })
        
        return annotations
    
    def _generate_query_sql_pair(self, table: Dict, template_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a natural language query and corresponding SQL query.
        
        Args:
            table: Table information
            template_type: Type of query template
            
        Returns:
            Tuple: (natural_language_query, sql_query)
        """
        table_name = table['name']
        columns = table['columns']
        
        # Select appropriate columns based on template type
        if template_type == 'single_kpi':
            kpi_columns = [col for col in columns if any(kpi in col.lower() 
                          for kpi in ['rsrp', 'rsrq', 'sinr', 'throughput', 'latency'])]
            if not kpi_columns:
                return None, None
            
            kpi_col = random.choice(kpi_columns)
            nl_query = f"What is the average {kpi_col} in {table_name}?"
            sql_query = f"SELECT AVG({kpi_col}) FROM {table_name};"
            
        elif template_type == 'temporal':
            time_columns = [col for col in columns if any(time_word in col.lower() 
                           for time_word in ['time', 'date', 'hour'])]
            kpi_columns = [col for col in columns if any(kpi in col.lower() 
                          for kpi in ['rsrp', 'rsrq', 'sinr', 'throughput'])]
            
            if not time_columns or not kpi_columns:
                return None, None
            
            time_col = random.choice(time_columns)
            kpi_col = random.choice(kpi_columns)
            nl_query = f"Show {kpi_col} values for the last day in {table_name}"
            sql_query = f"SELECT {time_col}, {kpi_col} FROM {table_name} WHERE {time_col} >= date('now', '-1 day');"
            
        elif template_type == 'aggregation':
            id_columns = [col for col in columns if 'id' in col.lower()]
            if not id_columns:
                return None, None
            
            id_col = random.choice(id_columns)
            nl_query = f"Count the number of unique {id_col} in {table_name}"
            sql_query = f"SELECT COUNT(DISTINCT {id_col}) FROM {table_name};"
            
        else:
            # Default case
            if len(columns) < 2:
                return None, None
            
            col1, col2 = random.sample(columns, 2)
            nl_query = f"Show {col1} and {col2} from {table_name}"
            sql_query = f"SELECT {col1}, {col2} FROM {table_name};"
        
        return nl_query, sql_query
    
    def generate_validation_queries(self, db_connection: sqlite3.Connection, 
                                   num_queries: int = 50) -> List[Dict]:
        """
        Generate validation queries based on actual data in the database.
        
        Args:
            db_connection: SQLite database connection
            num_queries: Number of validation queries to generate
            
        Returns:
            List[Dict]: Validation queries with expected results
        """
        validation_queries = []
        
        try:
            for table in self.schema_info.get('tables', []):
                table_name = table['name']
                
                # Generate simple aggregation queries
                for column in table['columns']:
                    # Check if column is numeric
                    cursor = db_connection.cursor()
                    cursor.execute(f"SELECT typeof({column}) FROM {table_name} LIMIT 1")
                    col_type = cursor.fetchone()
                    
                    if col_type and col_type[0] in ['integer', 'real']:
                        # Generate avg, min, max queries
                        for agg_func in ['AVG', 'MIN', 'MAX']:
                            sql_query = f"SELECT {agg_func}({column}) FROM {table_name}"
                            nl_query = f"What is the {agg_func.lower()} {column} in {table_name}?"
                            
                            # Execute query to get expected result
                            cursor.execute(sql_query)
                            result = cursor.fetchone()[0]
                            
                            validation_queries.append({
                                'natural_language': nl_query,
                                'sql': sql_query,
                                'expected_result': result,
                                'table': table_name,
                                'column': column,
                                'query_type': 'aggregation'
                            })
                            
                            if len(validation_queries) >= num_queries:
                                break
                
                if len(validation_queries) >= num_queries:
                    break
        
        except Exception as e:
            self.logger.error(f"Error generating validation queries: {e}")
        
        return validation_queries[:num_queries]
    
    def export_training_data(self, data: List[Dict], output_path: str, 
                           format: str = 'json') -> bool:
        """
        Export training data to file.
        
        Args:
            data: Training data to export
            output_path: Path to save the data
            format: Output format ('json', 'csv', 'parquet')
            
        Returns:
            bool: True if successful
        """
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format.lower() == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
                
            elif format.lower() == 'parquet':
                df = pd.DataFrame(data)
                df.to_parquet(output_path, index=False)
            
            self.logger.info(f"Training data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export training data: {e}")
            return False
    
    def get_training_statistics(self, data: List[Dict]) -> Dict:
        """
        Get statistics about the generated training data.
        
        Args:
            data: Training data
            
        Returns:
            Dict: Statistics about the training data
        """
        if not data:
            return {}
        
        stats = {
            'total_samples': len(data),
            'avg_query_length': sum(len(item.get('text', item.get('natural_language', '')).split()) 
                                  for item in data) / len(data),
        }
        
        # Template type distribution
        if 'template_type' in data[0]:
            template_counts = {}
            for item in data:
                template_type = item.get('template_type', 'unknown')
                template_counts[template_type] = template_counts.get(template_type, 0) + 1
            stats['template_distribution'] = template_counts
        
        # Entity type distribution for NER data
        if 'entities' in data[0]:
            entity_counts = {}
            for item in data:
                for entity in item.get('entities', []):
                    label = entity.get('label', 'unknown')
                    entity_counts[label] = entity_counts.get(label, 0) + 1
            stats['entity_distribution'] = entity_counts
        
        return stats
