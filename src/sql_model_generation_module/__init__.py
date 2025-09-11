"""
SQL Model Generation Module
===========================

This module trains and manages models for generating SQL queries from
natural language questions and extracted entities. It uses the entities
identified by the NER module to construct accurate SQL queries.

Key Features:
- Text-to-SQL model training
- Query template generation
- SQL validation and optimization
- Integration with NER outputs
"""

from .sql_trainer import SQLTrainer
from .query_generator import QueryGenerator
from .sql_validator import SQLValidator
from .template_manager import TemplateManager

__all__ = ['SQLTrainer', 'QueryGenerator', 'SQLValidator', 'TemplateManager']
