"""
Question Answering Module
=========================

This module provides the end-to-end question answering pipeline that
integrates NER and SQL generation modules. It includes a Streamlit UI
for user interaction.

Key Features:
- End-to-end QA pipeline
- NER and SQL integration
- Streamlit web interface
- Query result visualization
- Performance monitoring
"""

from .qa_pipeline import QuestionAnsweringPipeline
from .streamlit_app import StreamlitApp
from .result_formatter import ResultFormatter
from .query_executor import QueryExecutor

__all__ = ['QuestionAnsweringPipeline', 'StreamlitApp', 'ResultFormatter', 'QueryExecutor']
