# Database Structure Module - Analysis Report

## ✅ Module Completion Status: COMPLETE

The Database Structure Module has been fully implemented and executed successfully!

---

## 📊 Overview

This module analyzes the RAN database structure, integrates KPI mappings, and generates training datasets for the NER and SQL generation models.

---

## 🎯 Components Implemented

### 1. **Database Analyzer** (`database_analyzer.py`)
- Analyzes SQLite database schema
- Extracts table and column information
- Identifies relationships between tables
- Maps RAN-specific entities from database structure
- Generates comprehensive schema documentation

### 2. **KPI Analyzer** (`kpi_analyzer.py`) ⭐ NEW
- Parses KPI definition CSV file
- Extracts column dependencies from SQL formulas
- Creates mappings between KPIs and database columns
- Groups KPIs by category and technology
- Generates KPI summary reports

### 3. **Schema Mapper** (`schema_mapper.py`)
- Maps natural language terms to database elements
- Creates vocabulary for RAN domain concepts
- Enables query term to schema element translation
- Supports synonym mappings for flexible query understanding

### 4. **Training Data Generator** (`training_data_generator.py`)
- Generates NER training datasets
- Creates SQL query generation training pairs
- Uses KPI mappings for domain-aware generation
- Produces training data in multiple formats (JSON, CSV)

---

## 📈 Analysis Results

### Database Statistics
- **Database Path**: `/workspaces/ran_sql/data/databases/ran_performance.db`
- **Database Size**: 3.2 GB
- **Total Tables**: 2
  - `ran_2g_ran_2g`: 7,021,280 rows, 49 columns
  - `ran_4g_ran_4g`: 6,249,598 rows, 47 columns
- **Total Records**: 13,270,878 rows
- **Total Columns**: 96

### KPI Analysis
- **Total KPIs Defined**: 29 KPIs
- **Categories**: 10 categories
  - Accessibility (3 KPIs)
  - Availability (4 KPIs)
  - Capacity (2 KPIs)
  - Completeness (2 KPIs)
  - Congestion (2 KPIs)
  - Energy Consumption (4 KPIs)
  - Integrity (2 KPIs)
  - Mobility (3 KPIs)
  - Productivity (4 KPIs)
  - Retainability (3 KPIs)
- **Technologies**: 2G (13 KPIs), 4G (16 KPIs)
- **Unique Columns Referenced**: 82 database columns

### Training Data Generated
- **NER Training Samples**: 2,000 samples
  - Entity types: NUMERIC_VALUE, DATE_TIME, KPI_NAME, etc.
  - Template types: single_kpi, temporal, aggregation, filtering, comparison
- **SQL Training Samples**: 910 query-SQL pairs
  - Query types: aggregation, filtering, comparison
  - Covers both 2G and 4G tables

---

## 📁 Generated Files

All files are located in `data/processed/`:

1. **`database_schema.json`** (6 KB)
   - Complete database schema analysis
   - Table and column metadata
   - Relationships and statistics

2. **`kpi_mappings.json`** (20 KB)
   - KPI to column mappings
   - Column to KPI reverse mappings
   - KPI descriptions and formulas
   - Category and technology groupings

3. **`kpi_summary.csv`** (1.3 KB)
   - Tabular summary of all KPIs
   - Categories, technologies, and units
   - Number of columns per KPI

4. **`ner_training_data_spacy.json`** (255 KB)
   - NER training data in spaCy format
   - Entity annotations with positions
   - Template type classifications

5. **`ner_training_samples.csv`** (97 KB)
   - Human-readable NER samples
   - Text and template type columns

6. **`sql_training_data.json`** (195 KB)
   - SQL training data with query-SQL pairs
   - Table and template type information

7. **`sql_training_samples.csv`** (117 KB)
   - Human-readable SQL training samples
   - Natural language queries with corresponding SQL

8. **`analysis_report.json`** (1.1 KB)
   - Summary of all analysis results
   - File locations and statistics

---

## 🔍 Sample Training Data

### NER Training Sample
```
Query: "Show me the throughput values for cell BTM680"
Entities:
  - "throughput" → KPI_NAME (12:22)
  - "BTM680" → CELL_ID (37:43)
```

### SQL Training Sample
```
Natural Language: "What is the average call drop rate in Sumbagteng region?"
SQL: "SELECT AVG(cndrop) FROM ran_2g_ran_2g WHERE region = 'Sumbagteng';"
```

---

## 🗺️ Entity Mapping Examples

### Geographic Entities
- `region`: REGION entity (e.g., "Sumbagteng", "Kalimantan")
- `kabupaten`: CITY entity (e.g., "KOTA BATAM")

### Network Entities
- `siteid`: SITE_ID entity (e.g., "BTM678", "BTM680")
- `eniqhost`: ENODEB_ID entity

### KPI Columns (2G)
- `ccalls`: Call attempts
- `ccongs`: Call congestion
- `cndrop`: Call drops
- `tava`: Traffic availability

### KPI Columns (4G)
- `pm_count`: Performance measurement count
- `pmcelldowntimeauto`: Cell downtime (automatic)
- `pmerabestabattadded`: E-RAB establishment attempts
- `pmrrcconnuser`: RRC connected users

---

## 🚀 Next Steps

### Step (iii): Name Entity Recognition Training Module
Now that training data is generated, proceed to:

1. **Train spaCy NER Model**
   ```bash
   python scripts/train_ner_model.py
   ```

2. **Train Transformer NER Model** (optional)
   ```bash
   python scripts/train_ner_transformer.py
   ```

3. **Evaluate NER Models**
   ```bash
   python scripts/evaluate_ner_model.py
   ```

### Step (iv): SQL Model Generation Module
After NER training:

1. **Train SQL Generation Model**
   ```bash
   python scripts/train_sql_model.py
   ```

2. **Validate SQL Generation**
   ```bash
   python scripts/validate_sql_generation.py
   ```

---

## 📊 Disk Space Optimization

**Files Cleaned Up:**
- ✅ Removed `RAN_2G.csv` (1.8 GB)
- ✅ Removed `RAN_4G.csv` (2.1 GB)
- ✅ Removed Python `__pycache__` directories (~180 KB)

**Space Saved**: ~3.9 GB

**Retained Files:**
- Database: `ran_performance.db` (3.2 GB) ✅
- KPI mapping: `ran_formula_kpi_detail.csv` (8 KB) ✅
- Generated training data (~800 KB) ✅

---

## 🎓 Key Insights

1. **Comprehensive KPI Coverage**: 29 KPIs covering 10 categories provide extensive domain knowledge for model training

2. **Rich Training Data**: 2,910 total training samples (2,000 NER + 910 SQL) with diverse query patterns

3. **Column Dependencies**: 82 unique database columns are referenced in KPI formulas, enabling intelligent entity extraction

4. **Multi-Technology Support**: Training data covers both 2G and 4G networks with technology-specific entities and KPIs

5. **Template Diversity**: 5 template types for NER and 3 query types for SQL ensure model generalization

---

## 📝 Usage Examples

### Load KPI Mappings
```python
from src.database_structure_module.kpi_analyzer import KPIAnalyzer

analyzer = KPIAnalyzer('data/raw/ran_formula_kpi_detail.csv')
mappings = analyzer.analyze_kpi_mappings()

# Get KPIs by category
accessibility_kpis = analyzer.get_kpi_by_category('Accessibility')
print(accessibility_kpis)  # ['Session Setup Success Rate (SSSR)', ...]

# Get columns for a KPI
columns = analyzer.get_columns_for_kpi('Cell Availability')
print(columns)  # ['pmCellDowntimeAuto', 'pmCellDowntimeMan', 'pm_count']
```

### Load Training Data
```python
import json

# Load NER training data
with open('data/processed/ner_training_data_spacy.json', 'r') as f:
    ner_data = json.load(f)

# Load SQL training data
with open('data/processed/sql_training_data.json', 'r') as f:
    sql_data = json.load(f)
```

---

## ✅ Completion Checklist

- [x] Database schema analysis
- [x] KPI mapping extraction and analysis
- [x] Schema to natural language mapping
- [x] NER training data generation (2,000 samples)
- [x] SQL training data generation (910 samples)
- [x] Output files in multiple formats (JSON, CSV)
- [x] Comprehensive documentation
- [x] Disk space optimization
- [x] Analysis report generation

---

## 📞 Module Status: READY FOR NER TRAINING

The Database Structure Module is complete and all training data is ready for use in the Name Entity Recognition Training Module (Step iii).
