# RAN SQL Question Answering System - AI Coding Guide

## System Architecture & Flow
This is a **modular pipeline system** for translating natural language queries into SQL for Radio Access Network (RAN) performance data. The system follows a strict data flow:
1. **CSV Import** → SQLite database with optimized schema
2. **Schema Analysis** → Training data generation for ML models  
3. **NER Training** → Entity extraction from user queries
4. **SQL Generation** → Query construction from extracted entities
5. **QA Pipeline** → End-to-end processing with Streamlit UI

## Key Modules & Boundaries

### Import Module (`src/import_module/`)
- **Purpose**: Convert RAN CSV files (2G/4G) to SQLite with performance optimization
- **Key Pattern**: Uses `RANCSVImporter` with chunked processing for large files
- **Critical**: Automatically detects RAN technology (2G vs 4G) by column patterns
- **Data Path**: `data/raw/*.csv` → `data/databases/ran_performance.db`

### Database Structure Module (`src/database_structure_module/`)
- **Purpose**: Analyze DB schema to generate training datasets for ML models
- **Key Pattern**: `DatabaseAnalyzer` creates schema mappings, `TrainingDataGenerator` produces NER/SQL training data
- **Critical**: Maps RAN domain terms (RSRP, SINR, etc.) to actual DB columns

### NER Training Module (`src/name_entity_recognition_training_module/`)
- **Purpose**: Train models to extract RAN entities from natural language
- **Supported Models**: Both spaCy (`en_core_web_sm`) and Transformers (`BERT-based`)
- **Entity Types**: `CELL_ID`, `KPI_NAME`, `TIMESTAMP`, `AGGREGATION`, etc. (see `config/settings.py`)
- **Pattern**: Training data → Model training → Entity extraction pipeline

### SQL Model Generation Module (`src/sql_model_generation_module/`)
- **Purpose**: Generate SQL queries from extracted entities + context
- **Key Pattern**: Template-based approach with `TemplateManager` + ML validation
- **Critical**: Uses entity-to-SQL mapping patterns specific to RAN data queries

### Question Answering Module (`src/question_answering_module/`)
- **Purpose**: Orchestrate complete pipeline with web interface
- **Entry Point**: `streamlit_app.py` launches the web interface
- **Pattern**: `QAPipeline` coordinates NER → SQL Generation → Execution → Formatting

## Configuration & Environment

### Settings System (`config/settings.py`)
- **Pattern**: Environment-based configs (Development/Production/Testing)
- **Key Paths**: All paths computed from `PROJECT_ROOT`, creates directories automatically
- **Critical Settings**: 
  - `RAN_ENTITY_LABELS`: Defines NER entity types
  - `RAN_KPIS`: Known RAN performance indicators
  - Model paths: `models/ner/` and `models/sql_generation/`

### Data Organization
```
data/
├── raw/           # Original RAN_2G.csv, RAN_4G.csv files
├── processed/     # Generated training datasets
└── databases/     # ran_performance.db (SQLite)
```

## Development Workflow

### Initial Setup Commands
```bash
# Install dependencies (includes spaCy, transformers, streamlit)
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create directory structure
python -c "from config.settings import get_config; get_config().create_directories()"
```

### Typical Development Sequence
1. **Place CSV files** in `data/raw/` (RAN_2G.csv, RAN_4G.csv)
2. **Import data**: `python scripts/import_data.py --all`
3. **Analyze schema**: Run database analysis to generate training data
4. **Train models**: NER first, then SQL generation models
5. **Test pipeline**: `streamlit run src/question_answering_module/streamlit_app.py`

## RAN Domain Specifics

### CSV Structure Expectations
- **Common columns**: `id`, `region`, `kabupaten`, `siteid`, `eniqhost`, `last_update`
- **2G patterns**: `ccalls`, `ccongs`, `cndrop`, `tava`, `pmcount_2g`, etc.
- **4G patterns**: `pm_count`, `pmcell`, `pmera`, `pmho`, `pmuethp`, etc.
- **Auto-detection**: System identifies technology by column patterns

### Entity Recognition Patterns
- **Network Elements**: Cell IDs, Site IDs, eNB IDs (4G), etc.
- **KPIs**: RSRP, RSRQ, SINR, throughput, latency, BLER, CQI
- **Temporal**: Timestamp patterns for time-based queries
- **Operations**: Aggregation functions (AVG, MAX, MIN, SUM)

## Testing & Debugging

### Test Files Available
- `test_import_module.py`: Tests CSV import with actual data
- `test_large_import.py`: Performance testing with large datasets
- `scripts/import_data.py`: Main import script with options (--all, --file, --list)

### Common Issues
- **Missing spaCy model**: Run `python -m spacy download en_core_web_sm`
- **Path issues**: All imports assume project root in Python path
- **Memory**: Large CSV files use chunked processing (configurable in settings)

## Model Training Patterns

### NER Training Flow
1. Generate training data from DB schema analysis
2. Train spaCy model with RAN-specific entities
3. Validate with test queries containing known entities
4. Store trained model in `models/ner/`

### SQL Generation Flow  
1. Create entity-to-SQL templates for common RAN queries
2. Train seq2seq model (BART-based) on generated examples
3. Validate generated SQL syntax and execution
4. Store in `models/sql_generation/`

## Current Project Status

✅ **COMPLETED - Step (i) Import Module**:
- RAN CSV files imported to SQLite database (`data/databases/ran_performance.db`)
- Database size: 3.0GB with 13+ million rows (7M+ from 2G, 6M+ from 4G)
- Tables created: `ran_2g_ran_2g`, `ran_4g_ran_4g`
- Performance indexes applied for common query patterns

🎯 **NEXT STEPS**:
- Step (ii): Database structure analysis and training data generation
- Step (iii): NER model training for RAN entity extraction  
- Step (iv): SQL generation model training
- Step (v): End-to-end QA pipeline integration

Focus on the **database structure analysis next** - analyze the imported schema to generate training datasets for ML models.