#!/usr/bin/env python3
"""
Validate and Clean NER Training Data
=====================================

Checks for and removes duplicate/overlapping entities in training data.
"""

import json
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def check_overlaps(entities):
    """Check if entities have overlaps or duplicates."""
    issues = []
    seen = {}
    
    for i, entity in enumerate(entities):
        start, end, label = entity['start'], entity['end'], entity['label']
        
        # Check for exact duplicates
        key = (start, end, label)
        if key in seen:
            issues.append(f"Duplicate entity: {key}")
            continue
        seen[key] = i
        
        # Check for overlaps with other entities
        for j, other in enumerate(entities):
            if i >= j:
                continue
            
            other_start, other_end = other['start'], other['end']
            
            # Check if ranges overlap
            if not (end <= other_start or start >= other_end):
                issues.append(f"Overlap: ({start}, {end}, {label}) with ({other_start}, {other_end}, {other['label']})")
    
    return issues


def remove_duplicates(entities):
    """Remove duplicate entities, keeping first occurrence."""
    seen = set()
    cleaned = []
    
    for entity in entities:
        key = (entity['start'], entity['end'], entity['label'])
        if key not in seen:
            cleaned.append(entity)
            seen.add(key)
    
    return cleaned


def validate_and_clean_data(input_file, output_file):
    """Validate and clean training data."""
    print("="*80)
    print("🔍 VALIDATING NER TRAINING DATA")
    print("="*80)
    print()
    
    # Load data
    print(f"📁 Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    print(f"   Total samples: {len(data)}")
    print()
    
    # Check for issues
    print("🔬 Checking for duplicates and overlaps...")
    problematic_samples = []
    total_issues = 0
    
    for i, sample in enumerate(data):
        entities = sample.get('entities', [])
        if not entities:
            continue
        
        issues = check_overlaps(entities)
        if issues:
            problematic_samples.append((i, sample, issues))
            total_issues += len(issues)
    
    print(f"   Samples with issues: {len(problematic_samples)}/{len(data)}")
    print(f"   Total issues found: {total_issues}")
    print()
    
    if problematic_samples:
        print("⚠️  Sample Issues (first 5):")
        print("-"*80)
        for i, (idx, sample, issues) in enumerate(problematic_samples[:5]):
            print(f"\nSample {idx}:")
            print(f"  Text: {sample['text'][:70]}...")
            print(f"  Issues:")
            for issue in issues[:3]:
                print(f"    - {issue}")
        print()
    
    # Clean data
    print("🧹 Cleaning data...")
    cleaned_data = []
    removed_count = 0
    
    for sample in data:
        entities = sample.get('entities', [])
        if not entities:
            cleaned_data.append(sample)
            continue
        
        original_count = len(entities)
        cleaned_entities = remove_duplicates(entities)
        
        # Check for overlaps after removing duplicates
        if check_overlaps(cleaned_entities):
            # Skip samples with overlapping entities
            removed_count += 1
            continue
        
        cleaned_sample = {
            'text': sample['text'],
            'entities': cleaned_entities
        }
        cleaned_data.append(cleaned_sample)
    
    print(f"   Removed {removed_count} samples with overlapping entities")
    print(f"   Cleaned samples: {len(cleaned_data)}")
    print()
    
    # Count entities
    entity_counts = {}
    total_entities = 0
    for sample in cleaned_data:
        for entity in sample.get('entities', []):
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
            total_entities += 1
    
    print("📊 Entity Distribution:")
    print("-"*80)
    for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {label}: {count}")
    print(f"   Total entities: {total_entities}")
    print()
    
    # Save cleaned data
    print(f"💾 Saving cleaned data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    file_size = output_file.stat().st_size / 1024
    print(f"   File size: {file_size:.2f} KB")
    print()
    
    print("="*80)
    print("✅ DATA VALIDATION AND CLEANING COMPLETE!")
    print("="*80)
    print()
    print(f"📈 Summary:")
    print(f"   Original samples: {len(data)}")
    print(f"   Cleaned samples: {len(cleaned_data)}")
    print(f"   Removed samples: {removed_count}")
    print(f"   Total entities: {total_entities}")
    print(f"   Entity types: {len(entity_counts)}")
    print()
    

def main():
    config = get_config()
    
    input_file = config.PROCESSED_DATA_DIR / 'ner_training_data_enhanced.json'
    output_file = config.PROCESSED_DATA_DIR / 'ner_training_data_clean.json'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    validate_and_clean_data(input_file, output_file)


if __name__ == "__main__":
    main()
