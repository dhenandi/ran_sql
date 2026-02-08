"""
Generate FINAL NER Training Data with Enhanced Negative Examples
Focus on edge cases that caused false positives in specificity validation
"""

import json
import random
import sqlite3
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_config

config = get_config()

def get_database_entities():
    """Extract real entities from database"""
    db_path = config.DATA_DIR / "databases" / "ran_performance.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get regions
    cursor.execute("SELECT DISTINCT region FROM ran_2g_ran_2g WHERE region IS NOT NULL LIMIT 5")
    regions = [row[0] for row in cursor.fetchall()]
    
    # Get locations (kabupaten)
    cursor.execute("SELECT DISTINCT kabupaten FROM ran_2g_ran_2g WHERE kabupaten IS NOT NULL LIMIT 10")
    locations = [row[0] for row in cursor.fetchall()]
    
    # Get site IDs
    cursor.execute("SELECT DISTINCT siteid FROM ran_2g_ran_2g WHERE siteid IS NOT NULL LIMIT 40")
    sites = [row[0] for row in cursor.fetchall()]
    
    # Known KPIs
    kpis = [
        'ccalls', 'ccongs', 'cndrop', 'tava', 'cqi', 'bler', 'sinr', 'rsrp', 'rsrq',
        'throughput', 'latency', 'packet_loss', 'availability', 'accessibility',
        'cssr', 'hosr', 'integrity', 'mobility', 'retainability', 'cdr',
        'pmcount_2g', 'pm_count', 'pmcell', 'pmera', 'pmho', 'pmuethp'
    ]
    
    conn.close()
    
    return {
        'regions': regions,
        'locations': locations,
        'sites': sites,
        'kpis': kpis
    }

def generate_enhanced_negative_examples(count=1000):
    """
    Generate diverse negative examples focusing on edge cases
    These queries should NOT trigger any entity detection
    """
    
    # Edge case templates that caused false positives
    edge_case_templates = [
        # Words similar to KPIs but not entities
        "What is the purpose of {word}?",
        "Show me all {word} in the system",
        "How many {word} are there?",
        "List the {word} available",
        "Display {word} information",
        "Get the {word} details",
        "I need the {word} report",
        "Can you show {word}?",
        "What about the {word}?",
        "Tell me about {word}",
        
        # Questions about database structure
        "What tables are in this database?",
        "Show me the schema structure",
        "How many columns exist?",
        "What fields are available?",
        "List all table names",
        "Display database info",
        "What is the data model?",
        "Show me the relationships",
        
        # Generic requests without entities
        "Give me the data",
        "Show me the records",
        "Display the information",
        "I need help with the system",
        "Can you explain this?",
        "What is this for?",
        "How does this work?",
        "Where can I find data?",
        
        # Temporal references that might confuse DATE_TIME
        "When was this last updated?",
        "How old is this data?",
        "What is the current status?",
        "Is this information up to date?",
        "Show me the latest version",
        
        # Counting/measurement queries (not KPIs)
        "How many records are there?",
        "What is the total count?",
        "How big is the database?",
        "What's the row count?",
        "How much data exists?",
    ]
    
    # Common words that should NOT be tagged (edge cases from false positives)
    confusing_words = [
        'purpose', 'database', 'tables', 'count', 'columns', 'names', 'cells',
        'data', 'records', 'information', 'system', 'structure', 'schema',
        'fields', 'values', 'metrics', 'measurements', 'calculations', 'methods',
        'results', 'outputs', 'inputs', 'parameters', 'attributes', 'properties',
        'rates', 'trends', 'patterns', 'statistics', 'summaries', 'reports'
    ]
    
    negative_examples = []
    
    # Generate from edge case templates
    for _ in range(count // 2):
        template = random.choice(edge_case_templates)
        if '{word}' in template:
            word = random.choice(confusing_words)
            query = template.format(word=word)
        else:
            query = template
        
        negative_examples.append({
            'text': query,
            'entities': []  # No entities
        })
    
    # Generate completely generic questions
    generic_questions = [
        "What is this?", "How does it work?", "Can you help me?",
        "Show me everything", "I need information", "What can you do?",
        "Explain this to me", "Where do I start?", "What's available?",
        "How do I use this?", "What are my options?", "Can you assist?",
        "I'm looking for help", "What should I know?", "Tell me more",
        "Is there documentation?", "How can I learn?", "What's possible?",
        "Can you guide me?", "I need assistance", "Help me understand"
    ]
    
    for _ in range(count // 4):
        query = random.choice(generic_questions)
        negative_examples.append({
            'text': query,
            'entities': []
        })
    
    # Generate queries about analysis/interpretation (not specific data queries)
    analysis_questions = [
        "What's the best way to analyze this?",
        "How should I interpret the results?",
        "What methodology should I use?",
        "Can you recommend an approach?",
        "What are the best practices?",
        "How do I optimize queries?",
        "What's the most efficient method?",
        "Should I use aggregation?",
        "What about data quality?",
        "How do I ensure accuracy?"
    ]
    
    for _ in range(count // 4):
        query = random.choice(analysis_questions)
        negative_examples.append({
            'text': query,
            'entities': []
        })
    
    return negative_examples[:count]

def check_overlap(entities_list):
    """Check if any entities overlap"""
    for i in range(len(entities_list)):
        for j in range(i+1, len(entities_list)):
            start1, end1, _ = entities_list[i]
            start2, end2, _ = entities_list[j]
            # Check if ranges overlap
            if (start1 < end2 and start2 < end1):
                return True
    return False

def generate_diverse_positive_examples(entities, count=2400):
    """Generate diverse positive examples with actual entities"""
    
    positive_templates = [
        # KPI queries (20+ variations)
        "Show me {kpi} values",
        "What is the {kpi} for this network?",
        "Display {kpi} metrics",
        "Get {kpi} performance data",
        "I need {kpi} information",
        "Check {kpi} levels",
        "Monitor {kpi} values",
        "{kpi} trends please",
        "Analyze {kpi} patterns",
        "Report on {kpi}",
        "What's the average {kpi}?",
        "Show high {kpi}",
        "Display low {kpi}",
        "{kpi} statistics needed",
        "Track {kpi} changes",
        "Compare {kpi} values",
        "{kpi} summary report",
        "Get {kpi} breakdown",
        "{kpi} analysis required",
        "Measure {kpi} performance",
        
        # Location queries (15+ variations)
        "Show data from {location}",
        "Get records for {location}",
        "{location} network data",
        "What about {location}?",
        "Performance in {location}",
        "Display {location} metrics",
        "{location} statistics",
        "Analyze {location} network",
        "{location} area data",
        "Check {location} status",
        "Monitor {location} sites",
        "{location} performance report",
        "Data from {location} region",
        "{location} network quality",
        "Show {location} results",
        
        # Site queries (15+ variations)
        "Show data for site {site}",
        "Site {site} information",
        "Get {site} metrics",
        "{site} performance data",
        "Check site {site}",
        "Monitor {site}",
        "{site} status report",
        "Analyze {site}",
        "{site} network data",
        "Display {site} results",
        "{site} quality metrics",
        "Site {site} analysis",
        "{site} statistics",
        "Track {site} performance",
        "Report on {site}",
        
        # Combined queries
        "Show {kpi} in {location}",
        "Get {kpi} for site {site}",
        "{kpi} values in {location} above {value}",
        "Display {kpi} below {value} in {location}",
        "{location} sites with {kpi} over {value}",
        "Site {site} {kpi} metrics",
        "{kpi} performance in {location} region",
        "Check {kpi} at {site}",
        "{location} {kpi} analysis",
        "Monitor {kpi} in {location} area",
    ]
    
    positive_examples = []
    
    for _ in range(count):
        template = random.choice(positive_templates)
        query = template
        entities_dict = {}  # Track positions to avoid overlaps
        
        # Replace placeholders and track positions
        replacements = []
        
        if '{kpi}' in template:
            kpi = random.choice(entities['kpis'])
            replacements.append(('kpi', kpi, 'KPI_NAME'))
        
        if '{location}' in template:
            location = random.choice(entities['locations'])
            replacements.append(('location', location, 'LOCATION'))
        
        if '{site}' in template:
            site = random.choice(entities['sites'])
            replacements.append(('site', site, 'SITE_ID'))
        
        if '{value}' in template:
            value = str(random.randint(10, 100))
            replacements.append(('value', value, 'NUMERIC_VALUE'))
        
        # Apply replacements and track entity positions
        for placeholder, text, label in replacements:
            placeholder_str = f'{{{placeholder}}}'
            if placeholder_str in query:
                start = query.index(placeholder_str)
                query = query.replace(placeholder_str, text, 1)
                # After replacement, calculate actual position
                actual_start = start
                entities_dict[actual_start] = (actual_start, actual_start + len(text), label)
        
        # Convert to list sorted by position
        entities_list = [[s, e, l] for s, e, l in sorted(entities_dict.values())]
        
        # Skip if overlapping entities detected
        if check_overlap(entities_list):
            continue
        
        positive_examples.append({
            'text': query,
            'entities': entities_list
        })
    
    # Make sure we have enough examples (some were skipped due to overlaps)
    while len(positive_examples) < count:
        template = random.choice(positive_templates)
        query = template
        entities_dict = {}
        
        replacements = []
        if '{kpi}' in template:
            kpi = random.choice(entities['kpis'])
            replacements.append(('kpi', kpi, 'KPI_NAME'))
        if '{location}' in template:
            location = random.choice(entities['locations'])
            replacements.append(('location', location, 'LOCATION'))
        if '{site}' in template:
            site = random.choice(entities['sites'])
            replacements.append(('site', site, 'SITE_ID'))
        if '{value}' in template:
            value = str(random.randint(10, 100))
            replacements.append(('value', value, 'NUMERIC_VALUE'))
        
        for placeholder, text, label in replacements:
            placeholder_str = f'{{{placeholder}}}'
            if placeholder_str in query:
                start = query.index(placeholder_str)
                query = query.replace(placeholder_str, text, 1)
                actual_start = start
                entities_dict[actual_start] = (actual_start, actual_start + len(text), label)
        
        entities_list = [[s, e, l] for s, e, l in sorted(entities_dict.values())]
        
        if not check_overlap(entities_list):
            positive_examples.append({
                'text': query,
                'entities': entities_list
            })
    
    return positive_examples

def main():
    print("="*80)
    print("GENERATING FINAL NER TRAINING DATA")
    print("Focus: Enhanced negative examples to fix specificity issues")
    print("="*80)
    
    # Get real entities
    print("\n📊 Extracting entities from database...")
    entities = get_database_entities()
    print(f"   ✓ Regions: {len(entities['regions'])}")
    print(f"   ✓ Locations: {len(entities['locations'])}")
    print(f"   ✓ Sites: {len(entities['sites'])}")
    print(f"   ✓ KPIs: {len(entities['kpis'])}")
    
    # Generate negative examples (focus on edge cases)
    print(f"\n🚫 Generating 1,000 negative examples (edge cases)...")
    negative_examples = generate_enhanced_negative_examples(count=1000)
    print(f"   ✓ Generated {len(negative_examples)} negative examples")
    
    # Generate positive examples
    print(f"\n✅ Generating 2,400 positive examples...")
    positive_examples = generate_diverse_positive_examples(entities, count=2400)
    print(f"   ✓ Generated {len(positive_examples)} positive examples")
    
    # Combine and shuffle
    all_examples = negative_examples + positive_examples
    random.shuffle(all_examples)
    
    # Save training data
    output_file = config.PROCESSED_DATA_DIR / "ner_training_data_final.json"
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"\n💾 Training data saved: {output_file}")
    print(f"   Total samples: {len(all_examples)}")
    print(f"   Positive: {len(positive_examples)} (70.6%)")
    print(f"   Negative: {len(negative_examples)} (29.4%)")
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE")
    print("="*80)
    print("\n🎯 Key Improvements:")
    print("   • 1,000 negative examples (vs 800) - +25%")
    print("   • Focus on edge cases: 'purpose', 'tables', 'count', 'columns', etc.")
    print("   • 29.4% negative ratio (vs 25%) for better specificity")
    print("   • Total 3,400 samples (vs 3,200) - +6.25%")
    print("\n✓ Ready for training!")

if __name__ == "__main__":
    main()
