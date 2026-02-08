#!/usr/bin/env python3
"""
Enhanced NER Training Data Generator
=====================================

Generates improved NER training data with real RAN entities.
"""

import sys
import json
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def get_database_entities(db_path: str) -> Dict[str, List[str]]:
    """Extract real entities from database for training."""
    entities = {
        'regions': [],
        'cities': [],
        'sites': [],
        'kpis_2g': [],
        'kpis_4g': []
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get distinct regions
        cursor.execute("SELECT DISTINCT region FROM ran_2g_ran_2g WHERE region IS NOT NULL LIMIT 10")
        entities['regions'] = [row[0] for row in cursor.fetchall()]
        
        # Get distinct cities
        cursor.execute("SELECT DISTINCT kabupaten FROM ran_2g_ran_2g WHERE kabupaten IS NOT NULL LIMIT 20")
        entities['cities'] = [row[0] for row in cursor.fetchall()]
        
        # Get sample site IDs
        cursor.execute("SELECT DISTINCT siteid FROM ran_2g_ran_2g WHERE siteid IS NOT NULL LIMIT 30")
        entities['sites'] = [row[0] for row in cursor.fetchall()]
        
        # Get 2G KPI columns
        cursor.execute("PRAGMA table_info(ran_2g_ran_2g)")
        cols_2g = [row[1] for row in cursor.fetchall()]
        entities['kpis_2g'] = [col for col in cols_2g if col not in ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']]
        
        # Get 4G KPI columns
        cursor.execute("PRAGMA table_info(ran_4g_ran_4g)")
        cols_4g = [row[1] for row in cursor.fetchall()]
        entities['kpis_4g'] = [col for col in cols_4g if col not in ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']]
        
    finally:
        conn.close()
    
    return entities


def generate_enhanced_training_data(entities: Dict, num_samples: int = 1500) -> List[Dict]:
    """Generate enhanced NER training data with real entities."""
    
    training_data = []
    
    # Query templates with entity placeholders
    templates = [
        # Single KPI queries
        {
            "text": "What is the average {kpi} in {region}?",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("region", "REGION")
            ]
        },
        {
            "text": "Show me {kpi} for site {site}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("site", "SITE_ID")
            ]
        },
        {
            "text": "Get maximum {kpi} from all cells",
            "entities": [
                ("kpi", "KPI_NAME")
            ]
        },
        {
            "text": "Display {kpi} values for {city}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("city", "LOCATION")
            ]
        },
        # Threshold queries
        {
            "text": "Find sites where {kpi} is above {value}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("value", "NUMERIC_VALUE")
            ]
        },
        {
            "text": "Show cells with {kpi} below {value} in {region}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("value", "NUMERIC_VALUE"),
                ("region", "REGION")
            ]
        },
        # Aggregation queries
        {
            "text": "Calculate average {kpi} per site",
            "entities": [
                ("kpi", "KPI_NAME")
            ]
        },
        {
            "text": "Sum all {kpi} values for {region} region",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("region", "REGION")
            ]
        },
        {
            "text": "Count total sites in {city}",
            "entities": [
                ("city", "LOCATION")
            ]
        },
        # Technology-specific
        {
            "text": "Get 4G {kpi} statistics",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("4G", "TECHNOLOGY")
            ]
        },
        {
            "text": "Show 2G network {kpi}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("2G", "TECHNOLOGY")
            ]
        },
        # Comparative queries
        {
            "text": "Compare {kpi} between {region} and {region2}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("region", "REGION"),
                ("region2", "REGION")
            ]
        },
        # Temporal queries
        {
            "text": "What was the {kpi} yesterday?",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("yesterday", "DATE_TIME")
            ]
        },
        {
            "text": "Show {kpi} for last week in {city}",
            "entities": [
                ("kpi", "KPI_NAME"),
                ("last week", "DATE_TIME"),
                ("city", "LOCATION")
            ]
        }
    ]
    
    # Generate samples
    for _ in range(num_samples):
        template = random.choice(templates)
        text = template["text"]
        entity_annotations = []
        
        # Fill placeholders
        replacements = {}
        for placeholder, label in template["entities"]:
            if placeholder == "kpi":
                # Choose between 2G and 4G KPIs
                if "4G" in text or random.random() > 0.5:
                    value = random.choice(entities['kpis_4g'])
                else:
                    value = random.choice(entities['kpis_2g'])
                replacements["{kpi}"] = value
                
            elif placeholder == "region" or placeholder == "region2":
                value = random.choice(entities['regions'])
                replacements[f"{{{placeholder}}}"] = value
                
            elif placeholder == "city":
                value = random.choice(entities['cities'])
                replacements["{city}"] = value
                
            elif placeholder == "site":
                value = random.choice(entities['sites'])
                replacements["{site}"] = value
                
            elif placeholder == "value":
                value = str(random.choice([10, 20, 50, 75, 100, -10, -20]))
                replacements["{value}"] = value
                
            elif placeholder in ["yesterday", "last week"]:
                value = placeholder
                # These are already in text, just mark for annotation
        
        # Apply replacements
        filled_text = text
        for placeholder_key, value in replacements.items():
            filled_text = filled_text.replace(placeholder_key, value)
        
        # Create entity annotations
        for placeholder, label in template["entities"]:
            # Find the actual value that was used
            if placeholder in ["yesterday", "last week"]:
                search_text = placeholder
            else:
                search_text = replacements.get(f"{{{placeholder}}}", "")
            
            if search_text and search_text in filled_text:
                start = filled_text.find(search_text)
                end = start + len(search_text)
                entity_annotations.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": search_text
                })
        
        training_data.append({
            "text": filled_text,
            "entities": entity_annotations,
            "template_type": "enhanced"
        })
    
    return training_data


def main():
    config = get_config()
    
    print("="*80)
    print("🔧 ENHANCED NER TRAINING DATA GENERATION")
    print("="*80)
    print()
    
    # Load real entities from database
    print("Step 1: Extracting Real Entities from Database")
    print("-"*80)
    
    db_path = config.DATABASE_PATH
    entities = get_database_entities(str(db_path))
    
    print(f"   Regions: {len(entities['regions'])}")
    print(f"   Cities: {len(entities['cities'])}")
    print(f"   Sites: {len(entities['sites'])}")
    print(f"   2G KPIs: {len(entities['kpis_2g'])}")
    print(f"   4G KPIs: {len(entities['kpis_4g'])}")
    print()
    
    # Generate enhanced training data
    print("Step 2: Generating Enhanced Training Data")
    print("-"*80)
    
    num_samples = 1500
    training_data = generate_enhanced_training_data(entities, num_samples)
    
    print(f"   Generated: {len(training_data)} samples")
    
    # Count entities
    entity_counts = {}
    for sample in training_data:
        for entity in sample['entities']:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print(f"   Entity distribution:")
    for label, count in sorted(entity_counts.items()):
        print(f"      • {label}: {count}")
    print()
    
    # Save to file
    print("Step 3: Saving Enhanced Training Data")
    print("-"*80)
    
    output_path = config.PROCESSED_DATA_DIR / 'ner_training_data_enhanced.json'
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print()
    
    # Show samples
    print("Step 4: Sample Training Data")
    print("-"*80)
    
    for i, sample in enumerate(random.sample(training_data, min(5, len(training_data))), 1):
        print(f"\n   Sample {i}:")
        print(f"      Text: {sample['text']}")
        print(f"      Entities:")
        for ent in sample['entities']:
            print(f"         - {ent['text']:20} → {ent['label']}")
    
    print()
    print("="*80)
    print("✅ ENHANCED TRAINING DATA GENERATED!")
    print("="*80)
    print()
    print("Next: Retrain NER model with enhanced data")
    print("   python scripts/train_ner_model_enhanced.py")
    print()


if __name__ == "__main__":
    main()
