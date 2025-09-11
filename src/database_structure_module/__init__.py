"""
Database Structure Module
=========================

This module analyzes database structure and generates training datasets
for model training. It understands RAN database schemas and creates
mappings between natural language entities and database elements.

Key Features:
- Database schema analysis and documentation
- Entity-relationship mapping
- Training data generation for NER and SQL models
- Schema visualization and export
"""

from .database_analyzer import DatabaseAnalyzer
from .schema_mapper import SchemaMapper
from .training_data_generator import TrainingDataGenerator

__all__ = ['DatabaseAnalyzer', 'SchemaMapper', 'TrainingDataGenerator']
