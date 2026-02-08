#!/usr/bin/env python3
"""
Enhanced NER Training with Negative Examples
==============================================

Generates comprehensive training data including negative examples
and retrains the NER model for better generalization.
"""

import sys
import json
import random
from pathlib import Path
import sqlite3
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def get_database_entities():
    """Extract entities from database."""
    config = get_config()
    db_path = config.DATABASE_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get regions
    cursor.execute("SELECT DISTINCT region FROM ran_2g_ran_2g WHERE region IS NOT NULL LIMIT 10")
    regions = [row[0] for row in cursor.fetchall()]
    
    # Get cities
    cursor.execute("SELECT DISTINCT kabupaten FROM ran_2g_ran_2g WHERE kabupaten IS NOT NULL LIMIT 10")
    cities = [row[0] for row in cursor.fetchall()]
    
    # Get site IDs
    cursor.execute("SELECT DISTINCT siteid FROM ran_2g_ran_2g WHERE siteid IS NOT NULL LIMIT 30")
    sites = [row[0] for row in cursor.fetchall()]
    
    # Get 2G KPIs (column names)
    cursor.execute("PRAGMA table_info(ran_2g_ran_2g)")
    columns_2g = [row[1] for row in cursor.fetchall()]
    kpis_2g = [col for col in columns_2g if col not in ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']]
    
    # Get 4G KPIs
    cursor.execute("PRAGMA table_info(ran_4g_ran_4g)")
    columns_4g = [row[1] for row in cursor.fetchall()]
    kpis_4g = [col for col in columns_4g if col not in ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']]
    
    conn.close()
    
    return {
        'regions': regions,
        'cities': cities,
        'sites': sites,
        'kpis_2g': kpis_2g,
        'kpis_4g': kpis_4g,
        'all_kpis': list(set(kpis_2g + kpis_4g))
    }


def generate_negative_examples(entities, count=500):
    """Generate queries with NO entities (negative examples)."""
    
    negative_templates = [
        "Show me the data table",
        "Display information about the network",
        "What is the rate of improvement?",
        "Can you help me understand the metrics?",
        "I need to check the performance dashboard",
        "Please display the configuration settings",
        "Show network topology",
        "What are the key indicators?",
        "Display the summary report",
        "Get the latest updates",
        "Show system status",
        "What is the current situation?",
        "Display operational statistics",
        "How is the network performing?",
        "Show me the analysis results",
        "What trends do you see?",
        "Display historical data",
        "Show the comparison chart",
        "What insights can you provide?",
        "Display the monitoring dashboard",
        "Show quality metrics overview",
        "What statistics are available?",
        "Display aggregated results",
        "Show me the report summary",
        "What patterns are visible?",
        "Display city information",
        "Show region mapping",
        "What areas are covered?",
        "Display site locations",
        "Show infrastructure details",
        "What resources are allocated?",
        "Display capacity planning",
        "Show network architecture",
        "What technologies are deployed?",
        "Display vendor information",
        "Show equipment inventory",
        "What maintenance is scheduled?",
        "Display operational procedures",
        "Show best practices",
        "What recommendations do you have?",
    ]
    
    # Common non-entity words to include
    common_words = [
        "data", "table", "information", "metrics", "performance", 
        "dashboard", "configuration", "settings", "status", "situation",
        "statistics", "analysis", "results", "trends", "insights",
        "report", "summary", "overview", "patterns", "details",
        "city", "region", "areas", "locations", "infrastructure",
        "resources", "capacity", "architecture", "technologies",
        "equipment", "inventory", "procedures", "practices", "recommendations"
    ]
    
    samples = []
    
    # Add base templates
    for template in negative_templates[:count//2]:
        samples.append({
            'text': template,
            'entities': []
        })
    
    # Add variations with common words
    for _ in range(count - len(samples)):
        words = random.sample(common_words, random.randint(2, 4))
        queries = [
            f"Show me {words[0]} and {words[1]}",
            f"Display {words[0]} for {words[1]}",
            f"What is the {words[0]} of {words[1]}?",
            f"Get {words[0]} {words[1]} information",
            f"I need {words[0]} and {words[1]} details",
        ]
        samples.append({
            'text': random.choice(queries),
            'entities': []
        })
    
    return samples


def generate_diverse_positive_examples(entities, count=2000):
    """Generate positive examples with diverse phrasing."""
    
    samples = []
    
    # Template variations for each entity type
    kpi_templates = [
        "Show {kpi}",
        "What is {kpi}?",
        "Display {kpi} values",
        "Get {kpi} statistics",
        "Calculate {kpi}",
        "Find {kpi} metrics",
        "Analyze {kpi} data",
        "Retrieve {kpi} information",
        "Check {kpi} performance",
        "Monitor {kpi} trends",
        "I need {kpi} data",
        "Can you show me {kpi}?",
        "Please display {kpi}",
        "I want to see {kpi}",
        "Looking for {kpi} values",
    ]
    
    region_templates = [
        "{kpi} in {region}",
        "{kpi} for {region} region",
        "{kpi} from {region}",
        "{kpi} across {region}",
        "Show {kpi} in {region}",
        "Get {kpi} for {region}",
        "Display {kpi} from {region} area",
        "What is {kpi} in {region}?",
        "{region} {kpi} values",
        "{kpi} statistics for {region}",
    ]
    
    location_templates = [
        "{kpi} in {location}",
        "{kpi} for {location} city",
        "{kpi} from {location}",
        "Show {kpi} in {location}",
        "{location} {kpi} data",
        "Get {kpi} for {location}",
    ]
    
    site_templates = [
        "{kpi} for site {site}",
        "{kpi} at {site}",
        "Show {kpi} for {site}",
        "Get {site} {kpi}",
        "Site {site} {kpi}",
    ]
    
    threshold_templates = [
        "{kpi} above {threshold}",
        "{kpi} below {threshold}",
        "{kpi} greater than {threshold}",
        "{kpi} less than {threshold}",
        "{kpi} exceeding {threshold}",
        "Show {kpi} > {threshold}",
        "Find {kpi} < {threshold}",
        "Where {kpi} is above {threshold}",
    ]
    
    time_templates = [
        "{kpi} {time}",
        "{kpi} from {time}",
        "Show {kpi} {time}",
        "Get {kpi} for {time}",
        "{kpi} data {time}",
    ]
    
    time_expressions = [
        "yesterday", "today", "last week", "last month",
        "this week", "this month", "last 7 days", "past week"
    ]
    
    thresholds = ["50", "100", "200", "500", "1000"]
    
    # Generate samples
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        
        # Simple KPI queries
        template = random.choice(kpi_templates)
        text = template.format(kpi=kpi)
        start = text.find(kpi)
        samples.append({
            'text': text,
            'entities': [{'start': start, 'end': start + len(kpi), 'label': 'KPI_NAME'}]
        })
    
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        region = random.choice(entities['regions'])
        
        template = random.choice(region_templates)
        text = template.format(kpi=kpi, region=region)
        
        ents = []
        kpi_start = text.find(kpi)
        if kpi_start != -1:
            ents.append({'start': kpi_start, 'end': kpi_start + len(kpi), 'label': 'KPI_NAME'})
        
        region_start = text.find(region)
        if region_start != -1:
            ents.append({'start': region_start, 'end': region_start + len(region), 'label': 'REGION'})
        
        if ents:
            samples.append({'text': text, 'entities': ents})
    
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        location = random.choice(entities['cities'])
        
        template = random.choice(location_templates)
        text = template.format(kpi=kpi, location=location)
        
        ents = []
        kpi_start = text.find(kpi)
        if kpi_start != -1:
            ents.append({'start': kpi_start, 'end': kpi_start + len(kpi), 'label': 'KPI_NAME'})
        
        loc_start = text.find(location)
        if loc_start != -1:
            ents.append({'start': loc_start, 'end': loc_start + len(location), 'label': 'LOCATION'})
        
        if ents:
            samples.append({'text': text, 'entities': ents})
    
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        site = random.choice(entities['sites'])
        
        template = random.choice(site_templates)
        text = template.format(kpi=kpi, site=site)
        
        ents = []
        kpi_start = text.find(kpi)
        if kpi_start != -1:
            ents.append({'start': kpi_start, 'end': kpi_start + len(kpi), 'label': 'KPI_NAME'})
        
        site_start = text.find(site)
        if site_start != -1:
            ents.append({'start': site_start, 'end': site_start + len(site), 'label': 'SITE_ID'})
        
        if ents:
            samples.append({'text': text, 'entities': ents})
    
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        threshold = random.choice(thresholds)
        
        template = random.choice(threshold_templates)
        text = template.format(kpi=kpi, threshold=threshold)
        
        ents = []
        kpi_start = text.find(kpi)
        if kpi_start != -1:
            ents.append({'start': kpi_start, 'end': kpi_start + len(kpi), 'label': 'KPI_NAME'})
        
        threshold_start = text.find(threshold)
        if threshold_start != -1:
            ents.append({'start': threshold_start, 'end': threshold_start + len(threshold), 'label': 'NUMERIC_VALUE'})
        
        if ents:
            samples.append({'text': text, 'entities': ents})
    
    for _ in range(count // 6):
        kpi = random.choice(entities['all_kpis'])
        time_expr = random.choice(time_expressions)
        
        template = random.choice(time_templates)
        text = template.format(kpi=kpi, time=time_expr)
        
        ents = []
        kpi_start = text.find(kpi)
        if kpi_start != -1:
            ents.append({'start': kpi_start, 'end': kpi_start + len(kpi), 'label': 'KPI_NAME'})
        
        time_start = text.find(time_expr)
        if time_start != -1:
            ents.append({'start': time_start, 'end': time_start + len(time_expr), 'label': 'DATE_TIME'})
        
        if ents:
            samples.append({'text': text, 'entities': ents})
    
    return samples


def main():
    config = get_config()
    
    print("="*100)
    print("🔧 GENERATING ENHANCED TRAINING DATA WITH NEGATIVE EXAMPLES")
    print("="*100)
    print()
    
    # Extract entities
    print("📦 Extracting entities from database...")
    entities = get_database_entities()
    print(f"   Regions: {len(entities['regions'])}")
    print(f"   Cities: {len(entities['cities'])}")
    print(f"   Sites: {len(entities['sites'])}")
    print(f"   KPIs (2G): {len(entities['kpis_2g'])}")
    print(f"   KPIs (4G): {len(entities['kpis_4g'])}")
    print(f"   Total unique KPIs: {len(entities['all_kpis'])}")
    print()
    
    # Generate negative examples
    print("🚫 Generating negative examples (queries with NO entities)...")
    negative_samples = generate_negative_examples(entities, count=800)
    print(f"   Generated: {len(negative_samples)} negative samples")
    print()
    
    # Generate positive examples
    print("✅ Generating positive examples (diverse phrasing)...")
    positive_samples = generate_diverse_positive_examples(entities, count=2400)
    print(f"   Generated: {len(positive_samples)} positive samples")
    print()
    
    # Combine and shuffle
    all_samples = negative_samples + positive_samples
    random.shuffle(all_samples)
    
    print("📊 Final Dataset Statistics:")
    print(f"   Total samples: {len(all_samples)}")
    print(f"   Negative examples: {len(negative_samples)} ({len(negative_samples)/len(all_samples):.1%})")
    print(f"   Positive examples: {len(positive_samples)} ({len(positive_samples)/len(all_samples):.1%})")
    
    # Count entities
    entity_counts = {}
    for sample in all_samples:
        for entity in sample['entities']:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print(f"\n   Entity distribution:")
    for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {label}: {count}")
    print()
    
    # Save
    output_file = config.PROCESSED_DATA_DIR / 'ner_training_data_robust.json'
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    file_size = output_file.stat().st_size / 1024
    print(f"💾 Saved to: {output_file}")
    print(f"   File size: {file_size:.2f} KB")
    print()
    
    print("="*100)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*100)
    print()
    print("🎯 Next: Run training script")
    print("   python scripts/train_ner_robust.py")
    print()


if __name__ == "__main__":
    main()
