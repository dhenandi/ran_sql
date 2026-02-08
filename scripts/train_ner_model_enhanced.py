#!/usr/bin/env python3
"""
Enhanced NER Model Training Script
===================================

Trains NER model using enhanced training data with real RAN entities.
"""

import sys
import json
import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def setup_logging():
    """Setup logging configuration."""
    config = get_config()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOGS_DIR / 'ner_training_enhanced.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_training_data(data_path: Path):
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def convert_to_spacy_format(training_data):
    """Convert training data to spaCy format."""
    spacy_data = []
    
    for sample in training_data:
        text = sample.get('text', '')
        entities = sample.get('entities', [])
        
        # Convert entities to spaCy format
        ner_entities = []
        for entity in entities:
            start = entity.get('start')
            end = entity.get('end')
            label = entity.get('label')
            
            if start is not None and end is not None and label:
                ner_entities.append((start, end, label))
        
        if text and ner_entities:
            spacy_data.append((text, {"entities": ner_entities}))
    
    return spacy_data


def create_blank_model(entity_labels):
    """Create a blank spaCy model with NER component."""
    nlp = spacy.blank("en")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    for label in entity_labels:
        ner.add_label(label)
    
    return nlp


def train_ner_model(nlp, train_data, n_iter=30, batch_size=16):
    """Train the NER model."""
    logger = logging.getLogger(__name__)
    
    ner = nlp.get_pipe("ner")
    optimizer = nlp.begin_training()
    
    logger.info(f"Starting training for {n_iter} iterations...")
    
    for iteration in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        
        batches = minibatch(train_data, size=compounding(4.0, batch_size, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            nlp.update(examples, drop=0.35, losses=losses)
        
        if (iteration + 1) % 5 == 0:
            logger.info(f"Iteration {iteration + 1}/{n_iter} - Loss: {losses.get('ner', 0):.4f}")
    
    logger.info("Training completed!")
    return nlp


def evaluate_model(nlp, test_data):
    """Evaluate the trained model."""
    logger = logging.getLogger(__name__)
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    for text, annotations in test_data:
        doc = nlp(text)
        predicted_entities = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
        true_entities = set(annotations["entities"])
        
        tp += len(predicted_entities & true_entities)
        fp += len(predicted_entities - true_entities)
        fn += len(true_entities - predicted_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1_score:.4f}")
    
    return metrics


def test_model_predictions(nlp, test_queries):
    """Test the model with sample queries."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("Testing Model with Sample Queries")
    logger.info("="*80)
    
    for query in test_queries:
        doc = nlp(query)
        logger.info(f"\nQuery: {query}")
        if doc.ents:
            logger.info("  Detected Entities:")
            for ent in doc.ents:
                logger.info(f"    - {ent.text:25} → {ent.label_}")
        else:
            logger.info("  No entities detected")


def main():
    logger = setup_logging()
    config = get_config()
    
    print("="*80)
    print("🎓 ENHANCED NER MODEL TRAINING")
    print("="*80)
    print()
    
    # Load enhanced training data
    print("📁 Step 1: Loading Enhanced Training Data")
    print("-"*80)
    
    training_file = config.PROCESSED_DATA_DIR / 'ner_training_data_clean.json'
    if not training_file.exists():
        logger.error(f"Cleaned training data not found: {training_file}")
        logger.error("Please run: python scripts/validate_ner_data.py")
        return
    
    raw_data = load_training_data(training_file)
    logger.info(f"Loaded {len(raw_data)} training samples")
    
    # Convert to spaCy format
    spacy_data = convert_to_spacy_format(raw_data)
    logger.info(f"Converted {len(spacy_data)} samples to spaCy format")
    
    # Get unique entity labels
    entity_labels = set()
    for _, annotations in spacy_data:
        for _, _, label in annotations["entities"]:
            entity_labels.add(label)
    
    logger.info(f"Entity labels: {', '.join(sorted(entity_labels))}")
    print(f"   Training samples: {len(spacy_data)}")
    print(f"   Entity types: {len(entity_labels)}")
    print(f"   Labels: {', '.join(sorted(entity_labels))}")
    print()
    
    # Split data
    print("📊 Step 2: Splitting Data")
    print("-"*80)
    
    random.shuffle(spacy_data)
    split_point = int(len(spacy_data) * 0.8)
    train_data = spacy_data[:split_point]
    test_data = spacy_data[split_point:]
    
    print(f"   Training set: {len(train_data)} samples (80%)")
    print(f"   Test set: {len(test_data)} samples (20%)")
    print()
    
    # Create model
    print("🔧 Step 3: Creating NER Model")
    print("-"*80)
    
    nlp = create_blank_model(entity_labels)
    print(f"   Model: spaCy blank English")
    print(f"   Pipeline: {', '.join(nlp.pipe_names)}")
    print(f"   Entity labels: {len(entity_labels)}")
    print()
    
    # Train model
    print("🎯 Step 4: Training Model (30 iterations)")
    print("-"*80)
    print("   Training in progress...")
    print()
    
    nlp = train_ner_model(nlp, train_data, n_iter=30, batch_size=16)
    print()
    
    # Evaluate
    print("📈 Step 5: Evaluating Model")
    print("-"*80)
    
    metrics = evaluate_model(nlp, test_data)
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print()
    
    # Save model
    print("💾 Step 6: Saving Model")
    print("-"*80)
    
    model_output_dir = config.NER_MODELS_DIR / 'ran_ner_model_enhanced'
    nlp.to_disk(model_output_dir)
    print(f"   Model saved to: {model_output_dir}")
    
    metrics_file = config.NER_MODELS_DIR / 'ner_metrics_enhanced.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to: {metrics_file}")
    print()
    
    # Test
    print("🧪 Step 7: Testing Model")
    print("-"*80)
    
    test_queries = [
        "What is the average pmcelldowntimeauto in Sumbagteng region?",
        "Show me ccalls values for site BTM680",
        "Get maximum pmrrcconnlevsamp from all cells",
        "Calculate cndrop for 2G network in Kalimantan",
        "Display pmconsumedenergy by region",
        "Find sites where pmhoexeattlteintraf is above 100",
        "Show cells with tava below 50 in KOTA BATAM",
        "What was the pmcellsleeptime yesterday?"
    ]
    
    test_model_predictions(nlp, test_queries)
    print()
    
    # Model size
    import os
    model_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(model_output_dir)
        for filename in filenames
    ) / (1024 * 1024)
    
    print()
    print("="*80)
    print("✅ ENHANCED NER MODEL TRAINING COMPLETE!")
    print("="*80)
    print()
    print(f"📊 Final Results:")
    print(f"   Model: {model_output_dir}")
    print(f"   Model Size: {model_size:.2f} MB")
    print(f"   Training Samples: {len(train_data)}")
    print(f"   Test Samples: {len(test_data)}")
    print(f"   Entity Types: {len(entity_labels)}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print()
    print("🎯 Next Steps:")
    print("   1. Test the model: python scripts/test_ner_model.py")
    print("   2. Proceed to SQL model training (Step iv)")
    print()


if __name__ == "__main__":
    main()
