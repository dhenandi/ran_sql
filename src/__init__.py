"""
RAN SQL Question Answering System
==================================

A modular system for translating natural language queries into SQL queries
to extract Radio Access Network (RAN) performance data.

Main Modules:
- import_module: CSV to SQLite database import functionality
- database_structure_module: Database schema analysis and understanding
- name_entity_recognition_training_module: NER model training for RAN entities
- sql_model_generation_module: SQL query generation model training
- question_answering_module: End-to-end QA pipeline with Streamlit UI
"""

__version__ = "1.0.0"
__author__ = "RAN SQL Team"

from .import_module import CSVImporter
from .database_structure_module import DatabaseAnalyzer
from .name_entity_recognition_training_module import NERTrainer
from .sql_model_generation_module import SQLModelTrainer
from .question_answering_module import QuestionAnsweringSystem
