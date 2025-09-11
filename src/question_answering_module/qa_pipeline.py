"""
Question Answering Pipeline
===========================

Main pipeline that orchestrates the complete question answering process.
"""

import logging
from typing import Dict, List, Optional, Any, NamedTuple
from pathlib import Path

from ..name_entity_recognition_training_module import NERPipeline
from ..sql_model_generation_module import QueryGenerator, SQLValidator
from ..database_structure_module import DatabaseAnalyzer, SchemaMapper
from .query_executor import QueryExecutor
from .result_formatter import ResultFormatter


class QAResult(NamedTuple):
    """Result of question answering process."""
    success: bool
    natural_language_query: str
    entities: List[Dict]
    sql_query: Optional[str]
    query_results: Optional[List[Dict]]
    formatted_results: Optional[Dict]
    confidence: float
    processing_time: float
    errors: List[str]
    warnings: List[str]


class QuestionAnsweringPipeline:
    """
    Complete pipeline for processing natural language questions and generating answers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the QA pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ner_pipeline = None
        self.query_generator = None
        self.sql_validator = None
        self.query_executor = None
        self.result_formatter = ResultFormatter()
        
        # Schema information
        self.schema_info = {}
        self.schema_mapper = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize NER pipeline
            ner_config = self.config.get('ner', {})
            self.ner_pipeline = NERPipeline(ner_config)
            
            # Load NER model if path provided
            ner_model_path = self.config.get('ner_model_path')
            if ner_model_path and Path(ner_model_path).exists():
                self.ner_pipeline.load_model(ner_model_path)
                self.logger.info("NER model loaded successfully")
            
            # Initialize query generator
            sql_model_path = self.config.get('sql_model_path')
            if sql_model_path:
                self.query_generator = QueryGenerator(
                    sql_model_path,
                    self.config.get('sql_model_type', 'template')
                )
            
            # Initialize SQL validator
            db_path = self.config.get('database_path')
            if db_path:
                self.sql_validator = SQLValidator(db_path)
                self.query_executor = QueryExecutor(db_path)
            
            # Load schema information
            self._load_schema_info()
            
            self.logger.info("QA pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize QA pipeline: {e}")
            raise
    
    def _load_schema_info(self) -> None:
        """Load database schema information."""
        try:
            db_path = self.config.get('database_path')
            if not db_path or not Path(db_path).exists():
                self.logger.warning("Database not found, schema analysis skipped")
                return
            
            # Analyze database schema
            analyzer = DatabaseAnalyzer(db_path)
            self.schema_info = analyzer.analyze_database()
            
            # Initialize schema mapper
            self.schema_mapper = SchemaMapper(self.schema_info)
            
            self.logger.info("Schema information loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load schema info: {e}")
    
    def answer_question(self, question: str) -> QAResult:
        """
        Process a natural language question and return the answer.
        
        Args:
            question: Natural language question
            
        Returns:
            QAResult: Complete result of the QA process
        """
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Step 1: Extract entities using NER
            entities = self._extract_entities(question, errors)
            
            # Step 2: Map entities to schema elements
            if self.schema_mapper:
                mapped_entities = self._map_entities_to_schema(entities, question, warnings)
            else:
                mapped_entities = entities
                warnings.append("Schema mapping not available")
            
            # Step 3: Generate SQL query
            sql_result = self._generate_sql_query(question, mapped_entities, errors)
            
            if not sql_result or not sql_result.get('sql'):
                return QAResult(
                    success=False,
                    natural_language_query=question,
                    entities=entities,
                    sql_query=None,
                    query_results=None,
                    formatted_results=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    errors=errors,
                    warnings=warnings
                )
            
            # Step 4: Validate SQL query
            validation_result = self._validate_sql_query(sql_result['sql'], warnings)
            
            if not validation_result.get('is_valid', False):
                errors.extend(validation_result.get('errors', []))
                return QAResult(
                    success=False,
                    natural_language_query=question,
                    entities=entities,
                    sql_query=sql_result['sql'],
                    query_results=None,
                    formatted_results=None,
                    confidence=sql_result.get('confidence', 0.0),
                    processing_time=time.time() - start_time,
                    errors=errors,
                    warnings=warnings
                )
            
            # Step 5: Execute SQL query
            query_results = self._execute_sql_query(sql_result['sql'], errors)
            
            # Step 6: Format results
            formatted_results = self._format_results(
                question, sql_result['sql'], query_results, entities
            )
            
            processing_time = time.time() - start_time
            
            return QAResult(
                success=len(errors) == 0,
                natural_language_query=question,
                entities=entities,
                sql_query=sql_result['sql'],
                query_results=query_results,
                formatted_results=formatted_results,
                confidence=sql_result.get('confidence', 0.8),
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            errors.append(f"Processing error: {str(e)}")
            
            return QAResult(
                success=False,
                natural_language_query=question,
                entities=[],
                sql_query=None,
                query_results=None,
                formatted_results=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    def _extract_entities(self, question: str, errors: List[str]) -> List[Dict]:
        """Extract entities from the question."""
        try:
            if not self.ner_pipeline:
                errors.append("NER pipeline not initialized")
                return []
            
            entities = self.ner_pipeline.extract_entities(question)
            self.logger.info(f"Extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            errors.append(f"Entity extraction error: {str(e)}")
            return []
    
    def _map_entities_to_schema(self, entities: List[Dict], question: str, 
                               warnings: List[str]) -> List[Dict]:
        """Map extracted entities to database schema elements."""
        try:
            # Use schema mapper to enhance entities with database information
            query_mappings = self.schema_mapper.map_query_terms(question)
            
            # Enhance entities with schema mappings
            enhanced_entities = []
            for entity in entities:
                enhanced_entity = entity.copy()
                
                # Add schema mappings if available
                entity_text = entity.get('text', '').lower()
                entity_label = entity.get('label', '')
                
                # Map to database columns/tables
                if entity_label == 'KPI_NAME':
                    relevant_columns = self.schema_mapper.find_relevant_columns([entity_text])
                    if relevant_columns:
                        enhanced_entity['database_columns'] = relevant_columns
                
                enhanced_entities.append(enhanced_entity)
            
            return enhanced_entities
            
        except Exception as e:
            self.logger.warning(f"Schema mapping failed: {e}")
            warnings.append(f"Schema mapping warning: {str(e)}")
            return entities
    
    def _generate_sql_query(self, question: str, entities: List[Dict], 
                           errors: List[str]) -> Optional[Dict]:
        """Generate SQL query from question and entities."""
        try:
            if not self.query_generator:
                errors.append("Query generator not initialized")
                return None
            
            sql_result = self.query_generator.generate_sql(
                question, entities, self.schema_info
            )
            
            if sql_result.get('error'):
                errors.append(sql_result['error'])
                return None
            
            self.logger.info(f"Generated SQL: {sql_result.get('sql', '')}")
            return sql_result
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            errors.append(f"SQL generation error: {str(e)}")
            return None
    
    def _validate_sql_query(self, sql: str, warnings: List[str]) -> Dict:
        """Validate the generated SQL query."""
        try:
            if not self.sql_validator:
                warnings.append("SQL validator not available")
                return {'is_valid': True}  # Assume valid if no validator
            
            validation_result = self.sql_validator.validate_sql(sql, self.schema_info)
            
            # Add validation warnings to main warnings list
            warnings.extend(validation_result.get('warnings', []))
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"SQL validation failed: {e}")
            warnings.append(f"SQL validation warning: {str(e)}")
            return {'is_valid': True}  # Assume valid if validation fails
    
    def _execute_sql_query(self, sql: str, errors: List[str]) -> Optional[List[Dict]]:
        """Execute the SQL query and return results."""
        try:
            if not self.query_executor:
                errors.append("Query executor not initialized")
                return None
            
            results = self.query_executor.execute_query(sql)
            
            if results is None:
                errors.append("Query execution failed")
                return None
            
            self.logger.info(f"Query returned {len(results)} rows")
            return results
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            errors.append(f"Query execution error: {str(e)}")
            return None
    
    def _format_results(self, question: str, sql: str, results: List[Dict], 
                       entities: List[Dict]) -> Dict:
        """Format query results for presentation."""
        try:
            return self.result_formatter.format_results(
                question=question,
                sql_query=sql,
                raw_results=results,
                entities=entities,
                schema_info=self.schema_info
            )
            
        except Exception as e:
            self.logger.error(f"Result formatting failed: {e}")
            return {
                'error': f"Result formatting error: {str(e)}",
                'raw_results': results
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components."""
        return {
            'ner_pipeline': self.ner_pipeline is not None,
            'query_generator': self.query_generator is not None,
            'sql_validator': self.sql_validator is not None,
            'query_executor': self.query_executor is not None,
            'schema_info_loaded': bool(self.schema_info),
            'schema_mapper': self.schema_mapper is not None
        }
