"""
Template Manager
================

Manages SQL query templates for different query patterns.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class TemplateManager:
    """
    Manages SQL query templates and patterns.
    """
    
    def __init__(self, template_path: str = None):
        """
        Initialize the template manager.
        
        Args:
            template_path: Path to template file or directory
        """
        self.template_path = Path(template_path) if template_path else None
        self.logger = logging.getLogger(__name__)
        
        self.templates = {}
        self.patterns = {}
        
        if self.template_path:
            self._load_templates()
        else:
            self._initialize_default_templates()
    
    def _load_templates(self) -> None:
        """Load templates from file."""
        try:
            if self.template_path.is_file():
                with open(self.template_path, 'r') as f:
                    self.templates = json.load(f)
            else:
                # Load from directory
                for template_file in self.template_path.glob("*.json"):
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        self.templates.update(template_data)
            
            self.logger.info(f"Loaded templates from {self.template_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default RAN-specific SQL templates."""
        self.templates = {
            'aggregation': [
                {
                    'name': 'average_kpi',
                    'template': 'SELECT AVG(<KPI_NAME>) FROM <TABLE_NAME>',
                    'description': 'Calculate average value of a KPI',
                    'required_entities': ['KPI_NAME', 'TABLE_NAME'],
                    'optional_entities': ['CELL_ID', 'SITE_ID', 'DATE_TIME']
                },
                {
                    'name': 'max_kpi',
                    'template': 'SELECT MAX(<KPI_NAME>) FROM <TABLE_NAME>',
                    'description': 'Find maximum value of a KPI',
                    'required_entities': ['KPI_NAME', 'TABLE_NAME']
                },
                {
                    'name': 'count_cells',
                    'template': 'SELECT COUNT(DISTINCT <CELL_ID>) FROM <TABLE_NAME>',
                    'description': 'Count unique cells',
                    'required_entities': ['TABLE_NAME'],
                    'optional_entities': ['CELL_ID']
                }
            ],
            'selection': [
                {
                    'name': 'select_kpi',
                    'template': 'SELECT <KPI_NAME> FROM <TABLE_NAME>',
                    'description': 'Select KPI values',
                    'required_entities': ['KPI_NAME', 'TABLE_NAME']
                },
                {
                    'name': 'select_cell_data',
                    'template': 'SELECT * FROM <TABLE_NAME> WHERE <CELL_ID_COLUMN> = <CELL_ID>',
                    'description': 'Select all data for a specific cell',
                    'required_entities': ['TABLE_NAME', 'CELL_ID']
                }
            ],
            'filtering': [
                {
                    'name': 'filter_by_threshold',
                    'template': 'SELECT * FROM <TABLE_NAME> WHERE <KPI_NAME> > <NUMERIC_VALUE>',
                    'description': 'Filter by KPI threshold',
                    'required_entities': ['TABLE_NAME', 'KPI_NAME', 'NUMERIC_VALUE']
                },
                {
                    'name': 'filter_by_range',
                    'template': 'SELECT * FROM <TABLE_NAME> WHERE <KPI_NAME> BETWEEN <MIN_VALUE> AND <MAX_VALUE>',
                    'description': 'Filter by KPI range',
                    'required_entities': ['TABLE_NAME', 'KPI_NAME', 'MIN_VALUE', 'MAX_VALUE']
                }
            ],
            'temporal': [
                {
                    'name': 'filter_by_date',
                    'template': 'SELECT * FROM <TABLE_NAME> WHERE DATE(<DATE_COLUMN>) = <DATE_TIME>',
                    'description': 'Filter by specific date',
                    'required_entities': ['TABLE_NAME', 'DATE_TIME']
                },
                {
                    'name': 'hourly_average',
                    'template': 'SELECT HOUR(<TIMESTAMP_COLUMN>) as hour, AVG(<KPI_NAME>) FROM <TABLE_NAME> GROUP BY HOUR(<TIMESTAMP_COLUMN>)',
                    'description': 'Calculate hourly averages',
                    'required_entities': ['TABLE_NAME', 'KPI_NAME']
                }
            ],
            'comparison': [
                {
                    'name': 'compare_cells',
                    'template': 'SELECT <CELL_ID_COLUMN>, AVG(<KPI_NAME>) FROM <TABLE_NAME> GROUP BY <CELL_ID_COLUMN> ORDER BY AVG(<KPI_NAME>) DESC',
                    'description': 'Compare KPI values across cells',
                    'required_entities': ['TABLE_NAME', 'KPI_NAME']
                }
            ]
        }
        
        self.logger.info("Initialized default SQL templates")
    
    def get_templates_by_type(self, template_type: str) -> List[Dict]:
        """
        Get templates of a specific type.
        
        Args:
            template_type: Type of template (e.g., 'aggregation', 'selection')
            
        Returns:
            List[Dict]: Templates of the specified type
        """
        return self.templates.get(template_type, [])
    
    def find_matching_templates(self, entities: List[Dict], query_type: str = None) -> List[Dict]:
        """
        Find templates that match the available entities.
        
        Args:
            entities: Available entities
            query_type: Optional query type to filter templates
            
        Returns:
            List[Dict]: Matching templates
        """
        entity_labels = {entity.get('label', '') for entity in entities}
        matching_templates = []
        
        # Search in specific type or all types
        template_types = [query_type] if query_type else self.templates.keys()
        
        for ttype in template_types:
            for template in self.templates.get(ttype, []):
                required_entities = set(template.get('required_entities', []))
                
                # Check if all required entities are available
                if required_entities.issubset(entity_labels):
                    template_copy = template.copy()
                    template_copy['type'] = ttype
                    template_copy['match_score'] = self._calculate_match_score(
                        template, entities
                    )
                    matching_templates.append(template_copy)
        
        # Sort by match score
        matching_templates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return matching_templates
    
    def _calculate_match_score(self, template: Dict, entities: List[Dict]) -> float:
        """
        Calculate how well a template matches the available entities.
        
        Args:
            template: Template to evaluate
            entities: Available entities
            
        Returns:
            float: Match score (0-1)
        """
        required_entities = set(template.get('required_entities', []))
        optional_entities = set(template.get('optional_entities', []))
        entity_labels = {entity.get('label', '') for entity in entities}
        
        # Base score for having all required entities
        base_score = 1.0 if required_entities.issubset(entity_labels) else 0.0
        
        # Bonus for optional entities
        optional_matches = len(optional_entities & entity_labels)
        optional_bonus = optional_matches / len(optional_entities) if optional_entities else 0
        
        return base_score + (0.2 * optional_bonus)  # 20% bonus for optional matches
    
    def add_template(self, template_type: str, template: Dict) -> None:
        """
        Add a new template.
        
        Args:
            template_type: Type of template
            template: Template definition
        """
        if template_type not in self.templates:
            self.templates[template_type] = []
        
        self.templates[template_type].append(template)
        self.logger.info(f"Added new template: {template.get('name', 'unnamed')}")
    
    def save_templates(self, output_path: str) -> bool:
        """
        Save templates to file.
        
        Args:
            output_path: Path to save templates
            
        Returns:
            bool: True if successful
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.templates, f, indent=2)
            
            self.logger.info(f"Templates saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e}")
            return False
    
    def validate_template(self, template: Dict) -> Dict[str, Any]:
        """
        Validate a template definition.
        
        Args:
            template: Template to validate
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = ['name', 'template', 'description']
        for field in required_fields:
            if field not in template:
                results['errors'].append(f"Missing required field: {field}")
        
        # Check template syntax
        template_sql = template.get('template', '')
        if not template_sql:
            results['errors'].append("Empty template SQL")
        
        # Check for placeholder consistency
        placeholders = self._extract_placeholders(template_sql)
        required_entities = set(template.get('required_entities', []))
        
        for placeholder in placeholders:
            if placeholder not in required_entities and placeholder not in template.get('optional_entities', []):
                results['warnings'].append(f"Placeholder {placeholder} not in entity lists")
        
        results['is_valid'] = len(results['errors']) == 0
        
        return results
    
    def _extract_placeholders(self, template_sql: str) -> List[str]:
        """
        Extract placeholders from SQL template.
        
        Args:
            template_sql: SQL template with placeholders
            
        Returns:
            List[str]: Placeholder names
        """
        import re
        
        # Find all placeholders in format <PLACEHOLDER>
        placeholders = re.findall(r'<([^>]+)>', template_sql)
        return placeholders
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded templates.
        
        Returns:
            Dict: Template statistics
        """
        stats = {
            'total_templates': 0,
            'templates_by_type': {},
            'most_common_entities': {}
        }
        
        entity_counts = {}
        
        for template_type, templates in self.templates.items():
            stats['templates_by_type'][template_type] = len(templates)
            stats['total_templates'] += len(templates)
            
            # Count entity usage
            for template in templates:
                for entity in template.get('required_entities', []):
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
                for entity in template.get('optional_entities', []):
                    entity_counts[entity] = entity_counts.get(entity, 0) + 0.5
        
        # Sort entities by usage
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        stats['most_common_entities'] = dict(sorted_entities[:10])  # Top 10
        
        return stats
