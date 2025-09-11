# RAN SQL Question Answering System

A comprehensive modular system for translating natural language queries into SQL queries to extract Radio Access Network (RAN) performance data.

## System Overview

This system provides an end-to-end solution for analyzing RAN performance data through natural language questions. It consists of five main modules that work together to understand user queries, extract relevant entities, generate SQL queries, and present results through an intuitive web interface.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Question Answering Module                    │
│                     (Streamlit Web UI)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                 NER Training Module                             │
│              (Entity Recognition)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│              SQL Model Generation Module                       │
│                (Query Generation)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│             Database Structure Module                          │
│           (Schema Analysis & Mapping)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                   Import Module                                │
│               (CSV to SQLite Import)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Main Modules

### 1. Import Module (`src/import_module/`)
Handles the import of CSV files containing RAN performance data into SQLite databases.

**Key Components:**
- `csv_importer.py`: Main CSV import functionality
- `data_validator.py`: Validates CSV files and data quality
- `schema_optimizer.py`: Optimizes database schema for performance

**Features:**
- Automatic data type detection and optimization
- Data validation and quality checks
- Performance indexing for common query patterns
- Support for large CSV files with batch processing

### 2. Database Structure Module (`src/database_structure_module/`)
Analyzes database structure to understand RAN data schema and creates training datasets.

**Key Components:**
- `database_analyzer.py`: Analyzes database schema and relationships
- `schema_mapper.py`: Maps natural language terms to database elements
- `training_data_generator.py`: Generates training data for ML models

**Features:**
- Automatic schema discovery and documentation
- RAN domain-specific entity mapping
- Training data generation for NER and SQL models
- Relationship identification between tables

### 3. Name Entity Recognition Training Module (`src/name_entity_recognition_training_module/`)
Trains and fine-tunes NER models for RAN domain entities.

**Key Components:**
- `ner_trainer.py`: Trains NER models (spaCy and Transformers)
- `entity_extractor.py`: Extracts entities from user queries
- `model_evaluator.py`: Evaluates model performance
- `ner_pipeline.py`: End-to-end NER pipeline

**Supported Entity Types:**
- Network Elements: CELL_ID, SITE_ID, SECTOR_ID, ENB_ID, GNB_ID
- KPIs: RSRP, RSRQ, SINR, THROUGHPUT, LATENCY, BLER, CQI
- Temporal: TIMESTAMP, DATE_TIME
- Operations: AGGREGATION, COMPARISON_OP, NUMERIC_VALUE

### 4. SQL Model Generation Module (`src/sql_model_generation_module/`)
Trains models to generate SQL queries from extracted entities and natural language context.

**Key Components:**
- `sql_trainer.py`: Trains text-to-SQL models
- `query_generator.py`: Generates SQL queries from NER outputs
- `sql_validator.py`: Validates and optimizes generated SQL
- `template_manager.py`: Manages query templates for different patterns

**Features:**
- Template-based query generation
- Context-aware SQL construction
- Query validation and optimization
- Support for complex aggregations and joins

### 5. Question Answering Module (`src/question_answering_module/`)
Provides the end-to-end QA pipeline with web interface.

**Key Components:**
- `qa_pipeline.py`: Orchestrates the complete QA process
- `streamlit_app.py`: Web interface for user interaction
- `result_formatter.py`: Formats and presents query results
- `query_executor.py`: Executes SQL queries safely

**Features:**
- Interactive web interface with Streamlit
- Real-time query processing
- Result visualization and export
- Query history and performance monitoring

## Directory Structure

```
ran_sql/
├── src/
│   ├── __init__.py
│   ├── import_module/
│   │   ├── __init__.py
│   │   ├── csv_importer.py
│   │   ├── data_validator.py
│   │   └── schema_optimizer.py
│   ├── database_structure_module/
│   │   ├── __init__.py
│   │   ├── database_analyzer.py
│   │   ├── schema_mapper.py
│   │   └── training_data_generator.py
│   ├── name_entity_recognition_training_module/
│   │   ├── __init__.py
│   │   ├── ner_trainer.py
│   │   ├── entity_extractor.py
│   │   ├── model_evaluator.py
│   │   └── ner_pipeline.py
│   ├── sql_model_generation_module/
│   │   ├── __init__.py
│   │   ├── sql_trainer.py
│   │   ├── query_generator.py
│   │   ├── sql_validator.py
│   │   └── template_manager.py
│   └── question_answering_module/
│       ├── __init__.py
│       ├── qa_pipeline.py
│       ├── streamlit_app.py
│       ├── result_formatter.py
│       └── query_executor.py
├── data/
│   ├── raw/           # Original CSV files
│   ├── processed/     # Processed training data
│   └── databases/     # SQLite databases
├── models/
│   ├── ner/          # Trained NER models
│   └── sql_generation/ # Trained SQL generation models
├── config/
│   └── settings.py    # Configuration settings
├── tests/            # Unit and integration tests
├── scripts/          # Utility scripts
├── logs/            # Application logs
└── README.md
```

## Data Flow

1. **Data Import**: CSV files are imported into SQLite database with optimized schema
2. **Schema Analysis**: Database structure is analyzed to understand RAN data patterns
3. **Training Data Generation**: Synthetic training data is generated for ML models
4. **NER Model Training**: Models are trained to recognize RAN entities in queries
5. **SQL Model Training**: Models learn to generate SQL from entity-enriched queries
6. **Query Processing**: User queries are processed through the complete pipeline
7. **Result Presentation**: Query results are formatted and displayed in the web interface

## Key Features

- **Modular Design**: Each module can be developed and tested independently
- **RAN Domain Expertise**: Built-in knowledge of telecommunications terminology
- **Flexible Model Support**: Supports both spaCy and Transformers-based models
- **Performance Optimized**: Database and query optimization for large datasets
- **User-Friendly Interface**: Intuitive web interface for non-technical users
- **Extensible**: Easy to add new entity types, query patterns, and data sources

## Getting Started

1. **Install Dependencies**: Install required packages (see requirements.txt)
2. **Import Data**: Use the import module to load your CSV files
3. **Analyze Schema**: Run database analysis to understand your data structure
4. **Train Models**: Train NER and SQL generation models with your data
5. **Launch Interface**: Start the Streamlit app for interactive querying

## Configuration

The system uses environment-based configuration (Development, Production, Testing) defined in `config/settings.py`. Key settings include:

- Database paths and connection settings
- Model training parameters
- Logging configuration
- RAN-specific entity definitions
- Performance tuning parameters

## Example Usage

```python
from src import QuestionAnsweringSystem

# Initialize the system
qa_system = QuestionAnsweringSystem()

# Process a natural language query
query = "What is the average RSRP for cell ABC123 yesterday?"
result = qa_system.answer(query)

# Display results
print(result.sql_query)  # Generated SQL
print(result.data)       # Query results
print(result.visualization)  # Chart/graph
```

## Next Steps

Now that the basic structure is in place, you can proceed to implement the detailed functionality of each module. Start with:

1. Implementing the CSV import functionality
2. Setting up database schema analysis
3. Creating training data for your specific RAN datasets
4. Training NER models with RAN entities
5. Developing SQL generation capabilities
6. Building the web interface

Each module is designed to be independent, allowing for parallel development and testing.