#!/usr/bin/env python3
"""
NER Model Robustness Validation
================================

Tests for overfitting, underfitting, and generalization capability.
"""

import sys
import json
from pathlib import Path
import spacy
from collections import defaultdict

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def load_model():
    """Load trained NER model."""
    config = get_config()
    model_path = config.NER_MODELS_DIR / 'ran_ner_model_robust'
    return spacy.load(model_path)


def create_unseen_test_queries():
    """Create test queries with patterns NOT seen during training."""
    return [
        # Different wording patterns
        {
            "category": "Unseen Phrasing",
            "query": "Can you tell me about pmcelldowntimeauto metrics in Sumbagteng?",
            "expected": {"KPI_NAME": ["pmcelldowntimeauto"], "REGION": ["Sumbagteng"]}
        },
        {
            "category": "Unseen Phrasing",
            "query": "I need to check ccalls performance",
            "expected": {"KPI_NAME": ["ccalls"]}
        },
        {
            "category": "Unseen Phrasing",
            "query": "Could you display the tava readings from KOTA BATAM?",
            "expected": {"KPI_NAME": ["tava"], "LOCATION": ["KOTA BATAM"]}
        },
        
        # Edge cases - Common words that are NOT entities
        {
            "category": "Negative Examples",
            "query": "Show me the data table for performance metrics",
            "expected": {}  # "data" and "table" should NOT be KPIs
        },
        {
            "category": "Negative Examples",
            "query": "What is the rate of improvement?",
            "expected": {}  # "rate" should NOT be a KPI
        },
        {
            "category": "Negative Examples",
            "query": "Display city information",
            "expected": {}  # "city" should NOT be a KPI
        },
        
        # Complex natural language
        {
            "category": "Natural Language",
            "query": "I'm interested in understanding how pmhoexeattlteintraf is performing in the Kalimantan area",
            "expected": {"KPI_NAME": ["pmhoexeattlteintraf"], "REGION": ["Kalimantan"]}
        },
        {
            "category": "Natural Language",
            "query": "Please help me analyze the cndrop statistics for locations in Sumbagut",
            "expected": {"KPI_NAME": ["cndrop"], "REGION": ["Sumbagut"]}
        },
        
        # Multiple KPIs in one query
        {
            "category": "Multiple KPIs",
            "query": "Compare ccalls and cndrop in Sumbagteng",
            "expected": {"KPI_NAME": ["ccalls", "cndrop"], "REGION": ["Sumbagteng"]}
        },
        {
            "category": "Multiple KPIs",
            "query": "Show pmconsumedenergy and pmcellsleeptime trends",
            "expected": {"KPI_NAME": ["pmconsumedenergy", "pmcellsleeptime"]}
        },
        
        # Abbreviated/informal language
        {
            "category": "Informal",
            "query": "tava stats for BTM680?",
            "expected": {"KPI_NAME": ["tava"], "SITE_ID": ["BTM680"]}
        },
        {
            "category": "Informal",
            "query": "pmcelldowntimeauto > 100",
            "expected": {"KPI_NAME": ["pmcelldowntimeauto"], "NUMERIC_VALUE": ["100"]}
        },
        
        # Ambiguous contexts
        {
            "category": "Ambiguous",
            "query": "Show site data for BTM680 yesterday",
            "expected": {"SITE_ID": ["BTM680"], "DATE_TIME": ["yesterday"]}  # "data" should NOT be KPI
        },
        {
            "category": "Ambiguous",
            "query": "Get the value of ccalls exceeding 500",
            "expected": {"KPI_NAME": ["ccalls"], "NUMERIC_VALUE": ["500"]}  # "value", "exceeding" not KPIs
        },
        
        # Time expressions variety
        {
            "category": "Time Expressions",
            "query": "Show pmconsumedenergy from last month",
            "expected": {"KPI_NAME": ["pmconsumedenergy"], "DATE_TIME": ["last month"]}
        },
        {
            "category": "Time Expressions",
            "query": "What was tava on Monday?",
            "expected": {"KPI_NAME": ["tava"], "DATE_TIME": ["Monday"]}
        },
        {
            "category": "Time Expressions",
            "query": "Display ccalls for the past 7 days",
            "expected": {"KPI_NAME": ["ccalls"], "DATE_TIME": ["past 7 days"]}
        },
        
        # Geographic variations
        {
            "category": "Geographic",
            "query": "Show metrics for all sites in Sumbagteng region",
            "expected": {"REGION": ["Sumbagteng"]}
        },
        {
            "category": "Geographic",
            "query": "Compare performance across Kalimantan and Sumbagut regions",
            "expected": {"REGION": ["Kalimantan", "Sumbagut"]}
        },
    ]


def test_query(nlp, query_data):
    """Test a single query."""
    query = query_data["query"]
    expected = query_data["expected"]
    category = query_data["category"]
    
    doc = nlp(query)
    
    detected = defaultdict(list)
    for ent in doc.ents:
        detected[ent.label_].append(ent.text)
    
    detected = dict(detected)
    
    # For negative examples, any detection is a false positive
    if not expected:
        success = len(detected) == 0
        fp_count = sum(len(v) for v in detected.values())
        return {
            "query": query,
            "category": category,
            "expected": expected,
            "detected": detected,
            "success": success,
            "false_positives": fp_count,
            "is_negative_example": True
        }
    
    # Calculate matches for positive examples
    correct = 0
    total_expected = sum(len(v) for v in expected.values())
    
    for label, values in expected.items():
        if label in detected:
            for value in values:
                if value in detected[label]:
                    correct += 1
    
    total_detected = sum(len(v) for v in detected.values())
    
    return {
        "query": query,
        "category": category,
        "expected": expected,
        "detected": detected,
        "correct": correct,
        "total_expected": total_expected,
        "total_detected": total_detected,
        "success": correct == total_expected and total_detected == total_expected,
        "is_negative_example": False
    }


def analyze_overfitting_signs(results):
    """Check for signs of overfitting."""
    signs = []
    
    # Check negative examples
    negative_results = [r for r in results if r.get("is_negative_example")]
    if negative_results:
        false_positives = sum(r.get("false_positives", 0) for r in negative_results)
        if false_positives > 0:
            signs.append(f"⚠️  Model detects {false_positives} false positive entities in negative examples (overfitting to patterns)")
    
    # Check common word confusion
    common_words_as_entities = []
    for result in results:
        if "detected" in result:
            for label, values in result["detected"].items():
                common_words = ["data", "table", "rate", "city", "value", "exceeding", "stats", "site"]
                for word in values:
                    if word.lower() in common_words:
                        common_words_as_entities.append((word, label))
    
    if common_words_as_entities:
        signs.append(f"⚠️  Model confuses common words as entities: {set(common_words_as_entities)} (memorization vs understanding)")
    
    # Check performance drop on unseen patterns
    unseen_phrasing = [r for r in results if r["category"] == "Unseen Phrasing" and not r.get("is_negative_example")]
    if unseen_phrasing:
        success_rate = sum(r["success"] for r in unseen_phrasing) / len(unseen_phrasing)
        if success_rate < 0.7:
            signs.append(f"⚠️  Low success rate ({success_rate:.1%}) on unseen phrasing patterns (overfitting to training patterns)")
    
    return signs


def analyze_underfitting_signs(results):
    """Check for signs of underfitting."""
    signs = []
    
    # Check if model fails on basic patterns
    basic_results = [r for r in results if r["category"] in ["Unseen Phrasing", "Natural Language"] 
                     and not r.get("is_negative_example")]
    if basic_results:
        success_rate = sum(r["success"] for r in basic_results) / len(basic_results)
        if success_rate < 0.6:
            signs.append(f"⚠️  Model struggles with basic patterns ({success_rate:.1%} success) (underfitting)")
    
    # Check entity detection rate
    positive_results = [r for r in results if not r.get("is_negative_example")]
    if positive_results:
        total_expected = sum(r["total_expected"] for r in positive_results)
        total_detected = sum(r["total_detected"] for r in positive_results)
        detection_rate = total_detected / total_expected if total_expected > 0 else 0
        
        if detection_rate < 0.5:
            signs.append(f"⚠️  Low entity detection rate ({detection_rate:.1%}) (model not learning patterns)")
    
    # Check recall across categories
    category_recalls = defaultdict(list)
    for result in positive_results:
        if result["total_expected"] > 0:
            recall = result["correct"] / result["total_expected"]
            category_recalls[result["category"]].append(recall)
    
    poor_categories = []
    for cat, recalls in category_recalls.items():
        avg_recall = sum(recalls) / len(recalls)
        if avg_recall < 0.5:
            poor_categories.append((cat, avg_recall))
    
    if poor_categories:
        signs.append(f"⚠️  Poor recall in categories: {poor_categories} (model not generalizing)")
    
    return signs


def calculate_generalization_score(results):
    """Calculate overall generalization capability."""
    positive_results = [r for r in results if not r.get("is_negative_example")]
    negative_results = [r for r in results if r.get("is_negative_example")]
    
    if not positive_results:
        return 0.0
    
    # Positive performance (precision & recall)
    total_correct = sum(r["correct"] for r in positive_results)
    total_expected = sum(r["total_expected"] for r in positive_results)
    total_detected = sum(r["total_detected"] for r in positive_results)
    
    recall = total_correct / total_expected if total_expected > 0 else 0
    precision = total_correct / total_detected if total_detected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Negative performance (specificity - avoiding false positives)
    if negative_results:
        correct_negatives = sum(1 for r in negative_results if r["success"])
        specificity = correct_negatives / len(negative_results)
    else:
        specificity = 1.0
    
    # Combined score (weighted average)
    generalization_score = (0.7 * f1) + (0.3 * specificity)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "generalization_score": generalization_score
    }


def main():
    print("="*100)
    print("🔬 NER MODEL ROBUSTNESS VALIDATION")
    print("="*100)
    print()
    print("Testing for:")
    print("  • Overfitting (memorization vs understanding)")
    print("  • Underfitting (insufficient learning)")
    print("  • Generalization (unseen patterns)")
    print()
    
    # Load model
    print("📦 Loading model...")
    nlp = load_model()
    print("✅ Model loaded")
    print()
    
    # Create unseen test queries
    test_queries = create_unseen_test_queries()
    print(f"📝 Created {len(test_queries)} validation queries")
    print(f"   Categories: {len(set(q['category'] for q in test_queries))}")
    
    categories = defaultdict(int)
    for q in test_queries:
        categories[q['category']] += 1
    for cat, count in sorted(categories.items()):
        print(f"   • {cat}: {count}")
    print()
    
    # Test model
    print("🧪 Testing model on unseen queries...")
    print("="*100)
    print()
    
    results = []
    for i, query_data in enumerate(test_queries, 1):
        result = test_query(nlp, query_data)
        results.append(result)
        
        status = "✅" if result["success"] else "⚠️"
        print(f"{status} Test {i}/{len(test_queries)}: {result['category']}")
        print(f"   Query: {result['query'][:70]}...")
        
        if result.get("is_negative_example"):
            fp = result.get("false_positives", 0)
            print(f"   Expected: NO entities | Detected: {fp} entities")
            if fp > 0:
                print(f"   False positives: {result['detected']}")
        else:
            print(f"   Expected: {result['total_expected']} | Detected: {result['total_detected']} | Correct: {result['correct']}")
            if not result["success"]:
                print(f"   Expected: {result['expected']}")
                print(f"   Detected: {result['detected']}")
        print()
    
    # Analyze results
    print("="*100)
    print("📊 ANALYSIS")
    print("="*100)
    print()
    
    # Overall metrics
    positive_results = [r for r in results if not r.get("is_negative_example")]
    negative_results = [r for r in results if r.get("is_negative_example")]
    
    print(f"📈 Overall Performance:")
    print(f"   Total queries: {len(results)}")
    print(f"   Positive examples: {len(positive_results)}")
    print(f"   Negative examples: {len(negative_results)}")
    
    if positive_results:
        success_rate = sum(r["success"] for r in positive_results) / len(positive_results)
        print(f"   Success rate (positive): {success_rate:.1%}")
    
    if negative_results:
        success_rate_neg = sum(r["success"] for r in negative_results) / len(negative_results)
        print(f"   Success rate (negative): {success_rate_neg:.1%}")
    print()
    
    # Generalization metrics
    gen_metrics = calculate_generalization_score(results)
    print(f"🎯 Generalization Metrics:")
    print(f"   Precision: {gen_metrics['precision']:.1%}")
    print(f"   Recall: {gen_metrics['recall']:.1%}")
    print(f"   F1-Score: {gen_metrics['f1']:.1%}")
    print(f"   Specificity (avoiding FP): {gen_metrics['specificity']:.1%}")
    print(f"   Generalization Score: {gen_metrics['generalization_score']:.1%}")
    print()
    
    # Check for overfitting
    print("🔍 Overfitting Analysis:")
    overfitting_signs = analyze_overfitting_signs(results)
    if overfitting_signs:
        for sign in overfitting_signs:
            print(f"   {sign}")
    else:
        print("   ✅ No significant overfitting detected")
    print()
    
    # Check for underfitting
    print("🔍 Underfitting Analysis:")
    underfitting_signs = analyze_underfitting_signs(results)
    if underfitting_signs:
        for sign in underfitting_signs:
            print(f"   {sign}")
    else:
        print("   ✅ No significant underfitting detected")
    print()
    
    # Final verdict
    print("="*100)
    print("🏆 FINAL VERDICT")
    print("="*100)
    print()
    
    gen_score = gen_metrics['generalization_score']
    
    if gen_score >= 0.90:
        verdict = "EXCELLENT"
        emoji = "🌟"
        recommendation = "Model is production-ready! Proceed to SQL generation."
    elif gen_score >= 0.80:
        verdict = "GOOD"
        emoji = "✅"
        recommendation = "Model performs well. Minor improvements possible but ready to proceed."
    elif gen_score >= 0.70:
        verdict = "ACCEPTABLE"
        emoji = "👍"
        recommendation = "Model is usable but consider adding more training data for edge cases."
    else:
        verdict = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
        recommendation = "Model needs more training or architecture adjustments before production."
    
    print(f"{emoji} Model Quality: {verdict}")
    print(f"   Generalization Score: {gen_score:.1%}")
    print()
    print(f"💡 Recommendation:")
    print(f"   {recommendation}")
    print()
    
    if overfitting_signs:
        print("🔧 To Reduce Overfitting:")
        print("   • Add more negative examples (common words that are NOT entities)")
        print("   • Increase training data diversity")
        print("   • Add dropout or regularization")
        print()
    
    if underfitting_signs:
        print("🔧 To Reduce Underfitting:")
        print("   • Increase training iterations")
        print("   • Add more training examples")
        print("   • Verify entity labels are correct")
        print()
    
    # Save results
    config = get_config()
    output_file = config.PROCESSED_DATA_DIR / "ner_robustness_validation.json"
    
    output_data = {
        "generalization_metrics": gen_metrics,
        "overfitting_signs": overfitting_signs,
        "underfitting_signs": underfitting_signs,
        "verdict": verdict,
        "recommendation": recommendation,
        "detailed_results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
