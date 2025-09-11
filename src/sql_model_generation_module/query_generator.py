"""
Query Generator
===============

Generates SQL queries from natural language and extracted entities.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class QueryGenerator:
    """
    Generates SQL queries using trained models or templates.
    """
    
    def __init__(self, model_path: str, model_type: str = "template"):
        """
        Initialize the query generator.
        
        Args:
            model_path: Path to the trained model or templates
            model_type: Type of model ("template", "transformers")
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        self.templates = {}
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model or templates."""
        if self.model_type == "template":
            self._load_templates()
        else:
            self._load_transformer_model()
    
    def _load_templates(self) -> None:
        """Load SQL templates from file."""
        try:
            if self.model_path.is_file():
                template_file = self.model_path
            else:
                template_file = self.model_path / "sql_templates.json"
            
            with open(template_file, 'r') as f:
                self.templates = json.load(f)
            
            self.logger.info(f"Loaded SQL templates from {template_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            self.templates = {}
    
    def _load_transformer_model(self) -> None:
        """Load transformer-based model."""
        # Implementation placeholder for transformer models
        self.logger.warning("Transformer model loading not implemented yet")
    
    def generate_sql(self, natural_language: str, entities: List[Dict], 
                    schema_info: Dict = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language and entities.
        
        Args:
            natural_language: User's natural language query
            entities: Extracted entities from NER
            schema_info: Database schema information
            
        Returns:
            Dict: Generated SQL with metadata
        """
        if self.model_type == "template":
            return self._generate_from_template(natural_language, entities, schema_info)
        else:
            return self._generate_from_transformer(natural_language, entities, schema_info)
    
    def _generate_from_template(self, nl_query: str, entities: List[Dict], 
                              schema_info: Dict = None) -> Dict[str, Any]:
        """
        Generate SQL using template matching.
        
        Args:
            nl_query: Natural language query
            entities: Extracted entities
            schema_info: Database schema information
            
        Returns:
            Dict: Generated SQL with metadata
        """
        # Classify query type
        query_type = self._classify_query_type(nl_query, entities)
        
        # Find best matching template
        best_template = self._find_best_template(query_type, nl_query, entities)
        
        if not best_template:
            return {
                'sql': None,
                'confidence': 0.0,
                'error': f"No suitable template found for query type: {query_type}",
                'query_type': query_type
            }
        
        # Fill template with entities
        filled_sql = self._fill_template(best_template, entities, schema_info)
        
        return {
            'sql': filled_sql,
            'confidence': 0.8,  # Template-based confidence
            'template': best_template,
            'query_type': query_type,
            'entities_used': entities
        }
    
    def _classify_query_type(self, nl_query: str, entities: List[Dict]) -> str:
        """
        Classify the type of query based on natural language and entities.
        
        Args:
            nl_query: Natural language query
            entities: Extracted entities
            
        Returns:
            str: Query type classification
        """
        nl_lower = nl_query.lower()
        
        # Check for aggregation keywords
        if any(word in nl_lower for word in ['average', 'avg', 'sum', 'count', 'max', 'min', 'total']):
            return 'aggregation'
        
        # Check for temporal keywords
        if any(word in nl_lower for word in ['yesterday', 'today', 'last', 'between', 'date', 'time']):
            return 'temporal'
        
        # Check for comparison keywords
        if any(word in nl_lower for word in ['greater', 'less', 'above', 'below', 'higher', 'lower']):
            return 'filtering'
        
        # Check for comparison entities
        if any(entity.get('label') == 'COMPARISON_OP' for entity in entities):
            return 'comparison'
        
        # Default to selection
        return 'selection'
    
    def _find_best_template(self, query_type: str, nl_query: str, 
                           entities: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching template for the query.
        
        Args:
            query_type: Type of query
            nl_query: Natural language query
            entities: Extracted entities
            
        Returns:
            Optional[Dict]: Best matching template
        """
        templates = self.templates.get(query_type, [])
        
        if not templates:
            self.logger.warning(f"No templates found for query type: {query_type}")
            return None
        
        # Simple matching - use first template for now
        # In a real implementation, this would use more sophisticated matching
        best_template = templates[0]
        
        self.logger.info(f"Selected template for {query_type}: {best_template.get('template', '')}")
        return best_template
    
    def _fill_template(self, template_info: Dict, entities: List[Dict], 
                      schema_info: Dict = None) -> str:
        """
        Fill SQL template with actual entity values.
        
        Args:
            template_info: Template information
            entities: Extracted entities
            schema_info: Database schema information
            
        Returns:
            str: Filled SQL query
        """
        sql_template = template_info.get('template', '')
        
        # Create entity mapping
        entity_map = {}
        for entity in entities:
            label = entity.get('label', '')
            text = entity.get('text', '')
            
            if label == 'TABLE_NAME':
                entity_map['<TABLE_NAME>'] = text
            elif label == 'COLUMN_NAME':
                entity_map['<COLUMN_NAME>'] = text
            elif label == 'KPI_NAME':
                entity_map['<KPI_NAME>'] = text
            elif label == 'CELL_ID':
                entity_map['<CELL_ID>'] = f"'{text}'"
            elif label == 'SITE_ID':
                entity_map['<SITE_ID>'] = f"'{text}'"
            elif label == 'NUMERIC_VALUE':
                entity_map['<NUMERIC_VALUE>'] = text
            elif label == 'DATE_TIME':
                entity_map['<DATE_TIME>'] = f"'{text}'"
        
        # Fill template
        filled_sql = sql_template
        for placeholder, value in entity_map.items():
            filled_sql = filled_sql.replace(placeholder, value)
        
        # Use schema info to resolve ambiguous references
        if schema_info:
            filled_sql = self._resolve_schema_references(filled_sql, schema_info)
        
        return filled_sql
    
    def _resolve_schema_references(self, sql: str, schema_info: Dict) -> str:
        """
        Resolve table and column references using schema information.
        
        Args:
            sql: SQL query with potential ambiguous references
            schema_info: Database schema information
            
        Returns:
            str: SQL with resolved references
        """
        # Implementation placeholder for schema resolution
        # This would map entity references to actual table.column names
        return sql
    
    def _generate_from_transformer(self, nl_query: str, entities: List[Dict], 
                                 schema_info: Dict = None) -> Dict[str, Any]:
        """
        Generate SQL using transformer model.
        
        Args:
            nl_query: Natural language query
            entities: Extracted entities
            schema_info: Database schema information
            
        Returns:
            Dict: Generated SQL with metadata
        """
        # Implementation placeholder for transformer-based generation
        return {
            'sql': None,
            'confidence': 0.0,
            'error': "Transformer-based generation not implemented yet"
        }
    
    def validate_generated_sql(self, sql: str, schema_info: Dict = None) -> Dict[str, Any]:
        """
        Validate the generated SQL query.
        
        Args:
            sql: Generated SQL query
            schema_info: Database schema information
            
        Returns:
            Dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic SQL syntax validation
        if not sql or not sql.strip():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Empty SQL query")
            return validation_results
        
        # Check for basic SQL structure
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith('SELECT'):
            validation_results['warnings'].append("Query does not start with SELECT")
        
        # Check for SQL injection patterns
        dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for pattern in dangerous_patterns:
            if pattern in sql_upper:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Potentially dangerous SQL operation: {pattern}")
        
        return validation_results
