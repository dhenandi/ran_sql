#!/usr/bin/env python3
"""
Database Structure Analysis Script
===================================

Analyzes the RAN database structure, integrates KPI mappings,
and generates training datasets for NER and SQL generation models.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database_structure_module.database_analyzer import DatabaseAnalyzer
from src.database_structure_module.schema_mapper import SchemaMapper
from src.database_structure_module.kpi_analyzer import KPIAnalyzer
from src.database_structure_module.training_data_generator import TrainingDataGenerator
from config.settings import get_config
import logging
import json
import pandas as pd


def setup_logging():
    """Setup logging configuration."""
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOGS_DIR / 'database_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    config = get_config()
    
    print("=" * 80)
    print("🔍 RAN DATABASE STRUCTURE ANALYSIS")
    print("=" * 80)
    print()
    
    # Step 1: Analyze Database Structure
    print("📊 Step 1: Analyzing Database Structure")
    print("-" * 80)
    
    db_path = config.DATABASE_PATH
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    analyzer = DatabaseAnalyzer(str(db_path))
    schema_info = analyzer.analyze_database()
    
    summary = analyzer.get_schema_summary()
    print(f"   Tables: {summary['table_count']}")
    print(f"   Total Rows: {summary['total_rows']:,}")
    print(f"   Total Columns: {summary['total_columns']}")
    print(f"   Table Names: {', '.join(summary['table_names'])}")
    print()
    
    # Export schema analysis
    schema_output = config.PROCESSED_DATA_DIR / 'database_schema.json'
    analyzer.export_schema(str(schema_output))
    print(f"   ✅ Schema exported to: {schema_output}")
    print()
    
    # Step 2: Analyze KPI Mappings
    print("📈 Step 2: Analyzing KPI Mappings")
    print("-" * 80)
    
    kpi_csv_path = config.RAW_DATA_DIR / 'ran_formula_kpi_detail.csv'
    if not kpi_csv_path.exists():
        logger.warning(f"KPI mapping file not found: {kpi_csv_path}")
        kpi_analyzer = None
    else:
        kpi_analyzer = KPIAnalyzer(str(kpi_csv_path))
        kpi_mappings = kpi_analyzer.analyze_kpi_mappings()
        
        print(f"   KPI Categories: {len(kpi_mappings['kpis_by_category'])}")
        for category, kpis in kpi_mappings['kpis_by_category'].items():
            print(f"      • {category}: {len(kpis)} KPIs")
        
        print(f"\n   Technologies: {len(kpi_mappings['kpis_by_technology'])}")
        for tech, kpis in kpi_mappings['kpis_by_technology'].items():
            print(f"      • {tech}: {len(kpis)} KPIs")
        
        print(f"\n   Total Unique Columns Referenced: {len(kpi_mappings['column_to_kpis'])}")
        
        # Export KPI mappings
        kpi_output = config.PROCESSED_DATA_DIR / 'kpi_mappings.json'
        kpi_analyzer.export_mappings(str(kpi_output))
        print(f"   ✅ KPI mappings exported to: {kpi_output}")
        
        # Generate KPI summary
        kpi_summary = kpi_analyzer.generate_kpi_summary()
        kpi_summary_path = config.PROCESSED_DATA_DIR / 'kpi_summary.csv'
        kpi_summary.to_csv(kpi_summary_path, index=False)
        print(f"   ✅ KPI summary exported to: {kpi_summary_path}")
    print()
    
    # Step 3: Create Schema Mappings
    print("🗺️  Step 3: Creating Schema Mappings")
    print("-" * 80)
    
    mapper = SchemaMapper(schema_info)
    
    # Test query mapping
    test_queries = [
        "What is the average call drop rate in Sumbagteng region?",
        "Show me cell availability for 4G network",
        "Get throughput statistics by site"
    ]
    
    for query in test_queries:
        mappings = mapper.map_query_terms(query)
        print(f"   Query: {query}")
        print(f"      Tables: {mappings.get('tables', [])[:3]}")
        print(f"      Columns: {len(mappings.get('columns', []))} identified")
        print()
    
    # Step 4: Generate Training Data
    print("🎓 Step 4: Generating Training Datasets")
    print("-" * 80)
    
    generator = TrainingDataGenerator(schema_info, str(db_path), kpi_analyzer)
    
    # Generate NER training data
    print("   Generating NER training data...")
    ner_data = generator.generate_ner_training_data(num_samples=2000)
    
    # Save in spaCy format
    ner_output_spacy = config.PROCESSED_DATA_DIR / 'ner_training_data_spacy.json'
    with open(ner_output_spacy, 'w') as f:
        json.dump(ner_data, f, indent=2)
    print(f"   ✅ Generated {len(ner_data)} NER training samples")
    print(f"      Saved to: {ner_output_spacy}")
    
    # Convert to DataFrame for easy viewing
    ner_df = pd.DataFrame(ner_data)
    ner_csv = config.PROCESSED_DATA_DIR / 'ner_training_samples.csv'
    ner_df[['text', 'template_type']].to_csv(ner_csv, index=False)
    print(f"      Sample CSV: {ner_csv}")
    print()
    
    # Generate SQL training data
    print("   Generating SQL training data...")
    sql_data = generator.generate_sql_training_data(num_samples=1500)
    
    sql_output = config.PROCESSED_DATA_DIR / 'sql_training_data.json'
    with open(sql_output, 'w') as f:
        json.dump(sql_data, f, indent=2)
    print(f"   ✅ Generated {len(sql_data)} SQL training samples")
    print(f"      Saved to: {sql_output}")
    
    # Save SQL samples as CSV
    sql_df = pd.DataFrame(sql_data)
    sql_csv = config.PROCESSED_DATA_DIR / 'sql_training_samples.csv'
    sql_df.to_csv(sql_csv, index=False)
    print(f"      Sample CSV: {sql_csv}")
    print()
    
    # Step 5: Generate Entity Statistics
    print("📊 Step 5: Entity Statistics")
    print("-" * 80)
    
    # Analyze entity distribution in training data
    entity_types = {}
    for sample in ner_data:
        for entity in sample.get('entities', []):
            entity_type = entity['label']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print("   Entity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"      • {entity_type}: {count}")
    print()
    
    # SQL query type distribution
    sql_types = sql_df['template_type'].value_counts()
    print("   SQL Query Type Distribution:")
    for query_type, count in sql_types.items():
        print(f"      • {query_type}: {count}")
    print()
    
    # Step 6: Generate Summary Report
    print("📝 Step 6: Generating Summary Report")
    print("-" * 80)
    
    report = {
        'database_analysis': {
            'path': str(db_path),
            'tables': summary['table_count'],
            'total_rows': summary['total_rows'],
            'total_columns': summary['total_columns']
        },
        'kpi_analysis': {
            'total_kpis': len(kpi_mappings.get('kpi_descriptions', {})) if kpi_analyzer else 0,
            'categories': len(kpi_mappings.get('kpis_by_category', {})) if kpi_analyzer else 0,
            'technologies': list(kpi_mappings.get('kpis_by_technology', {}).keys()) if kpi_analyzer else []
        },
        'training_data': {
            'ner_samples': len(ner_data),
            'sql_samples': len(sql_data),
            'entity_types': list(entity_types.keys()),
            'query_types': list(sql_types.index)
        },
        'output_files': {
            'schema': str(schema_output),
            'kpi_mappings': str(kpi_output) if kpi_analyzer else None,
            'kpi_summary': str(kpi_summary_path) if kpi_analyzer else None,
            'ner_training': str(ner_output_spacy),
            'sql_training': str(sql_output),
            'ner_csv': str(ner_csv),
            'sql_csv': str(sql_csv)
        }
    }
    
    report_path = config.PROCESSED_DATA_DIR / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✅ Summary report saved to: {report_path}")
    print()
    
    print("=" * 80)
    print("✅ DATABASE STRUCTURE ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("📁 Generated Files:")
    for key, path in report['output_files'].items():
        if path:
            print(f"   • {key}: {Path(path).name}")
    print()
    print("🎯 Next Step: Train NER models using the generated training data")
    print("   Command: python scripts/train_ner_model.py")
    print()


if __name__ == "__main__":
    main()
