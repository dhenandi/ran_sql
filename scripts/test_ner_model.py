#!/usr/bin/env python3
"""
NER Model Testing Script
========================

Tests the trained NER model with sample queries.
"""

import sys
from pathlib import Path
import spacy

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def test_ner_model():
    """Test the trained NER model with various queries."""
    config = get_config()
    
    # Load the trained model
    model_path = config.NER_MODELS_DIR / 'ran_ner_model'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first: python scripts/train_ner_model.py")
        return
    
    print("="*80)
    print("🧪 TESTING NER MODEL")
    print("="*80)
    print()
    
    print(f"📁 Loading model from: {model_path}")
    nlp = spacy.load(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Pipeline: {', '.join(nlp.pipe_names)}")
    print()
    
    # Test queries for different scenarios
    test_queries = [
        # KPI queries
        "What is the average RSRP in Sumbagteng?",
        "Show me cell availability for all sites",
        "Get maximum throughput from BTM680",
        
        # Temporal queries
        "Display drop rate for last week",
        "What was the latency yesterday?",
        
        # Aggregation queries
        "Count total cells in Kalimantan region",
        "Sum all traffic volume",
        
        # Threshold queries
        "Find sites with SINR below -10 dB",
        "Show cells where availability exceeds 99%",
        
        # Technology-specific
        "Get 4G handover success rate",
        "Display 2G congestion statistics",
        
        # Complex queries
        "Calculate average energy consumption per site in Sumbagteng region",
        "Show me all KPIs for cell ID BTM678 on 2023-01-15",
    ]
    
    print("="*80)
    print("TEST RESULTS")
    print("="*80)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: \"{query}\"")
        doc = nlp(query)
        
        if doc.ents:
            print("   Detected Entities:")
            for ent in doc.ents:
                print(f"      • {ent.text:25} → {ent.label_:15} (confidence: {ent._.get('confidence', 'N/A')})")
        else:
            print("   ⚠️  No entities detected")
        print()
    
    print("="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    print()
    print("💡 Tips:")
    print("   • If no entities are detected, check training data quality")
    print("   • Low confidence scores may indicate need for more training")
    print("   • Consider adding more diverse training examples")
    print()


if __name__ == "__main__":
    test_ner_model()
