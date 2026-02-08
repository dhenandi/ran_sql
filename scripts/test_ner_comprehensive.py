#!/usr/bin/env python3
"""
Comprehensive NER Model Testing
================================

Tests the trained NER model with diverse queries and generates detailed metrics.
"""

import sys
import json
from pathlib import Path
import spacy
from collections import defaultdict
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def load_model(model_path):
    """Load trained NER model."""
    print(f"Loading model from: {model_path}")
    nlp = spacy.load(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Pipeline: {', '.join(nlp.pipe_names)}")
    
    if "ner" in nlp.pipe_names:
        ner = nlp.get_pipe("ner")
        print(f"   Entity labels: {', '.join(ner.labels)}")
    
    return nlp


def create_test_queries():
    """Create comprehensive test query set."""
    return [
        # KPI + Region queries
        {
            "category": "KPI + Region",
            "query": "What is the average pmcelldowntimeauto in Sumbagteng region?",
            "expected": {"KPI_NAME": ["pmcelldowntimeauto"], "REGION": ["Sumbagteng"]}
        },
        {
            "category": "KPI + Region",
            "query": "Show me ccalls values for Kalimantan",
            "expected": {"KPI_NAME": ["ccalls"], "REGION": ["Kalimantan"]}
        },
        {
            "category": "KPI + Region",
            "query": "Calculate cndrop statistics in Sumbagut",
            "expected": {"KPI_NAME": ["cndrop"], "REGION": ["Sumbagut"]}
        },
        
        # KPI + Location queries
        {
            "category": "KPI + Location",
            "query": "Display tava values in KOTA BATAM",
            "expected": {"KPI_NAME": ["tava"], "LOCATION": ["KOTA BATAM"]}
        },
        {
            "category": "KPI + Location",
            "query": "Show pmconsumedenergy for ASAHAN city",
            "expected": {"KPI_NAME": ["pmconsumedenergy"], "LOCATION": ["ASAHAN"]}
        },
        
        # KPI + Site ID queries
        {
            "category": "KPI + Site",
            "query": "Get pmhoexeattlteintraf for site BTM680",
            "expected": {"KPI_NAME": ["pmhoexeattlteintraf"], "SITE_ID": ["BTM680"]}
        },
        {
            "category": "KPI + Site",
            "query": "Show me data for BTM680 site",
            "expected": {"SITE_ID": ["BTM680"]}
        },
        
        # KPI + Numeric threshold queries
        {
            "category": "KPI + Threshold",
            "query": "Find cells where pmhoexeattlteintraf is above 100",
            "expected": {"KPI_NAME": ["pmhoexeattlteintraf"], "NUMERIC_VALUE": ["100"]}
        },
        {
            "category": "KPI + Threshold",
            "query": "Show cells with tava below 50",
            "expected": {"KPI_NAME": ["tava"], "NUMERIC_VALUE": ["50"]}
        },
        {
            "category": "KPI + Threshold",
            "query": "List sites where ccalls exceeds 1000",
            "expected": {"KPI_NAME": ["ccalls"], "NUMERIC_VALUE": ["1000"]}
        },
        
        # KPI + Time queries
        {
            "category": "KPI + Time",
            "query": "What was the pmcellsleeptime yesterday?",
            "expected": {"KPI_NAME": ["pmcellsleeptime"], "DATE_TIME": ["yesterday"]}
        },
        {
            "category": "KPI + Time",
            "query": "Show pmrrcconnlevsamp data from last week",
            "expected": {"KPI_NAME": ["pmrrcconnlevsamp"], "DATE_TIME": ["last week"]}
        },
        
        # Complex multi-entity queries
        {
            "category": "Complex",
            "query": "Show cells with tava below 50 in KOTA BATAM",
            "expected": {"KPI_NAME": ["tava"], "NUMERIC_VALUE": ["50"], "LOCATION": ["KOTA BATAM"]}
        },
        {
            "category": "Complex",
            "query": "Get pmconsumedenergy above 200 in Sumbagteng region yesterday",
            "expected": {"KPI_NAME": ["pmconsumedenergy"], "NUMERIC_VALUE": ["200"], "REGION": ["Sumbagteng"], "DATE_TIME": ["yesterday"]}
        },
        
        # 2G specific KPIs
        {
            "category": "2G KPIs",
            "query": "What is the ccongs rate in Kalimantan?",
            "expected": {"KPI_NAME": ["ccongs"], "REGION": ["Kalimantan"]}
        },
        {
            "category": "2G KPIs",
            "query": "Show cs12dlack statistics",
            "expected": {"KPI_NAME": ["cs12dlack"]}
        },
        
        # 4G specific KPIs
        {
            "category": "4G KPIs",
            "query": "Display pmerabestabsuccadded metrics",
            "expected": {"KPI_NAME": ["pmerabestabsuccadded"]}
        },
        {
            "category": "4G KPIs",
            "query": "Get pmrrcconnlevsamp values",
            "expected": {"KPI_NAME": ["pmrrcconnlevsamp"]}
        },
        
        # Aggregation queries
        {
            "category": "Aggregation",
            "query": "Calculate average pmcelldowntimeauto",
            "expected": {"KPI_NAME": ["pmcelldowntimeauto"]}
        },
        {
            "category": "Aggregation",
            "query": "Get maximum tava value",
            "expected": {"KPI_NAME": ["tava"]}
        },
        {
            "category": "Aggregation",
            "query": "Count total ccalls",
            "expected": {"KPI_NAME": ["ccalls"]}
        },
    ]


def test_query(nlp, query_data):
    """Test a single query and return results."""
    query = query_data["query"]
    expected = query_data["expected"]
    category = query_data["category"]
    
    doc = nlp(query)
    
    detected = defaultdict(list)
    for ent in doc.ents:
        detected[ent.label_].append(ent.text)
    
    detected = dict(detected)
    
    # Calculate match scores
    correct = 0
    total_expected = sum(len(v) for v in expected.values())
    
    for label, values in expected.items():
        if label in detected:
            for value in values:
                if value in detected[label]:
                    correct += 1
    
    precision = correct / sum(len(v) for v in detected.values()) if detected else 0
    recall = correct / total_expected if total_expected > 0 else 0
    
    return {
        "query": query,
        "category": category,
        "expected": expected,
        "detected": detected,
        "correct": correct,
        "total_expected": total_expected,
        "total_detected": sum(len(v) for v in detected.values()),
        "precision": precision,
        "recall": recall,
        "success": correct == total_expected and sum(len(v) for v in detected.values()) == total_expected
    }


def test_model(nlp, test_queries):
    """Test model with all queries."""
    print("\n" + "="*80)
    print("🧪 TESTING NER MODEL")
    print("="*80)
    print()
    
    results = []
    category_stats = defaultdict(lambda: {"total": 0, "success": 0, "correct": 0, "expected": 0, "detected": 0})
    
    for i, query_data in enumerate(test_queries, 1):
        result = test_query(nlp, query_data)
        results.append(result)
        
        category = result["category"]
        category_stats[category]["total"] += 1
        if result["success"]:
            category_stats[category]["success"] += 1
        category_stats[category]["correct"] += result["correct"]
        category_stats[category]["expected"] += result["total_expected"]
        category_stats[category]["detected"] += result["total_detected"]
        
        status = "✅" if result["success"] else "⚠️"
        print(f"{status} Test {i}/{len(test_queries)}: {result['category']}")
        print(f"   Query: {result['query'][:70]}...")
        print(f"   Expected: {result['total_expected']} | Detected: {result['total_detected']} | Correct: {result['correct']}")
        
        if not result["success"]:
            print(f"   Expected: {result['expected']}")
            print(f"   Detected: {result['detected']}")
        print()
    
    return results, dict(category_stats)


def calculate_overall_metrics(results):
    """Calculate overall performance metrics."""
    total_correct = sum(r["correct"] for r in results)
    total_expected = sum(r["total_expected"] for r in results)
    total_detected = sum(r["total_detected"] for r in results)
    total_success = sum(1 for r in results if r["success"])
    
    overall_precision = total_correct / total_detected if total_detected > 0 else 0
    overall_recall = total_correct / total_expected if total_expected > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    success_rate = total_success / len(results) if results else 0
    
    return {
        "total_queries": len(results),
        "successful_queries": total_success,
        "success_rate": success_rate,
        "total_expected": total_expected,
        "total_detected": total_detected,
        "total_correct": total_correct,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1
    }


def analyze_entity_performance(results):
    """Analyze performance by entity type."""
    entity_stats = defaultdict(lambda: {"expected": 0, "detected": 0, "correct": 0})
    
    for result in results:
        for label, values in result["expected"].items():
            entity_stats[label]["expected"] += len(values)
            
            if label in result["detected"]:
                detected_values = result["detected"][label]
                for value in values:
                    if value in detected_values:
                        entity_stats[label]["correct"] += 1
        
        for label, values in result["detected"].items():
            entity_stats[label]["detected"] += len(values)
    
    # Calculate metrics for each entity type
    for label, stats in entity_stats.items():
        precision = stats["correct"] / stats["detected"] if stats["detected"] > 0 else 0
        recall = stats["correct"] / stats["expected"] if stats["expected"] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats["precision"] = precision
        stats["recall"] = recall
        stats["f1"] = f1
    
    return dict(entity_stats)


def save_results(results, category_stats, overall_metrics, entity_stats, output_dir):
    """Save test results to JSON file."""
    output_data = {
        "overall_metrics": overall_metrics,
        "category_performance": category_stats,
        "entity_performance": entity_stats,
        "detailed_results": results
    }
    
    output_file = output_dir / "ner_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    return output_file


def print_summary(overall_metrics, category_stats, entity_stats):
    """Print test summary."""
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print()
    
    print("📈 Overall Performance:")
    print("-"*80)
    print(f"   Total Queries: {overall_metrics['total_queries']}")
    print(f"   Successful: {overall_metrics['successful_queries']} ({overall_metrics['success_rate']:.2%})")
    print(f"   Precision: {overall_metrics['overall_precision']:.4f} ({overall_metrics['overall_precision']:.2%})")
    print(f"   Recall: {overall_metrics['overall_recall']:.4f} ({overall_metrics['overall_recall']:.2%})")
    print(f"   F1-Score: {overall_metrics['overall_f1']:.4f} ({overall_metrics['overall_f1']:.2%})")
    print()
    
    print("📋 Performance by Category:")
    print("-"*80)
    for category, stats in sorted(category_stats.items()):
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        precision = stats["correct"] / stats["detected"] if stats["detected"] > 0 else 0
        recall = stats["correct"] / stats["expected"] if stats["expected"] > 0 else 0
        print(f"   {category:20} | Success: {stats['success']}/{stats['total']} ({success_rate:.1%}) | "
              f"P: {precision:.2f} | R: {recall:.2f}")
    print()
    
    print("🏷️  Performance by Entity Type:")
    print("-"*80)
    for label, stats in sorted(entity_stats.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"   {label:15} | P: {stats['precision']:.4f} | R: {stats['recall']:.4f} | F1: {stats['f1']:.4f}")
        print(f"   {'':15} | Expected: {stats['expected']:3} | Detected: {stats['detected']:3} | Correct: {stats['correct']:3}")
    print()


def main():
    config = get_config()
    
    print("="*80)
    print("🔬 COMPREHENSIVE NER MODEL TESTING")
    print("="*80)
    print()
    
    # Load model
    model_path = config.NER_MODELS_DIR / 'ran_ner_model_enhanced'
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first: python scripts/train_ner_model_enhanced.py")
        return
    
    nlp = load_model(model_path)
    print()
    
    # Create test queries
    test_queries = create_test_queries()
    print(f"📝 Created {len(test_queries)} test queries")
    print(f"   Categories: {len(set(q['category'] for q in test_queries))}")
    print()
    
    # Test model
    results, category_stats = test_model(nlp, test_queries)
    
    # Calculate metrics
    overall_metrics = calculate_overall_metrics(results)
    entity_stats = analyze_entity_performance(results)
    
    # Save results
    output_file = save_results(results, category_stats, overall_metrics, entity_stats, config.PROCESSED_DATA_DIR)
    print()
    
    # Print summary
    print_summary(overall_metrics, category_stats, entity_stats)
    
    print("="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print()
    print("🎯 Next Steps:")
    print(f"   1. Visualize results: Open src/import_module/ner_performance_analysis.ipynb")
    print(f"   2. Review detailed results: {output_file}")
    print()


if __name__ == "__main__":
    main()
