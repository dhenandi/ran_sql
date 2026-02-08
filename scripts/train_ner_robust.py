#!/usr/bin/env python3
"""
Robust NER Model Training Script (Background-Ready)
====================================================

Trains NER model with enhanced data including negative examples.
Includes checkpointing and logging for background execution.
"""

import sys
import json
import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import logging
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def setup_logging():
    """Setup logging for background execution."""
    config = get_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f'ner_training_robust_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger, log_file


def load_training_data(data_path):
    """Load training data."""
    with open(data_path, 'r') as f:
        return json.load(f)


def convert_to_spacy_format(training_data):
    """Convert to spaCy format and validate."""
    spacy_data = []
    skipped = 0
    
    for sample in training_data:
        text = sample.get('text', '')
        entities = sample.get('entities', [])
        
        # Validate and sort entities
        valid_entities = []
        for entity in entities:
            start = entity.get('start')
            end = entity.get('end')
            label = entity.get('label')
            
            if start is not None and end is not None and label and start < end:
                valid_entities.append((start, end, label))
        
        # Check for overlaps
        valid_entities.sort()
        clean_entities = []
        for i, (start, end, label) in enumerate(valid_entities):
            overlap = False
            for j, (s2, e2, l2) in enumerate(clean_entities):
                if not (end <= s2 or start >= e2):
                    overlap = True
                    break
            if not overlap:
                clean_entities.append((start, end, label))
            else:
                skipped += 1
        
        if text:
            spacy_data.append((text, {"entities": clean_entities}))
    
    return spacy_data, skipped


def save_checkpoint(nlp, iteration, metrics, config):
    """Save model checkpoint."""
    checkpoint_dir = config.NER_MODELS_DIR / f'checkpoint_iter_{iteration}'
    nlp.to_disk(checkpoint_dir)
    
    metrics_file = checkpoint_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return checkpoint_dir


def train_ner_model(nlp, train_data, test_data, logger, config, n_iter=50):
    """Train with checkpointing."""
    logger.info(f"Starting training for {n_iter} iterations...")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    ner = nlp.get_pipe("ner")
    optimizer = nlp.begin_training()
    
    best_f1 = 0.0
    
    for iteration in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            nlp.update(examples, drop=0.5, losses=losses)
        
        # Evaluate every 5 iterations
        if (iteration + 1) % 5 == 0:
            metrics = evaluate_model(nlp, test_data, logger)
            f1 = metrics['f1_score']
            
            logger.info(f"Iteration {iteration + 1}/{n_iter}")
            logger.info(f"  Loss: {losses.get('ner', 0):.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            
            # Save checkpoint if best so far
            if f1 > best_f1:
                best_f1 = f1
                checkpoint_path = save_checkpoint(nlp, iteration + 1, metrics, config)
                logger.info(f"  ✅ New best F1! Checkpoint saved: {checkpoint_path}")
            
            logger.info("")
    
    logger.info("Training completed!")
    return nlp, best_f1


def evaluate_model(nlp, test_data, logger):
    """Evaluate model."""
    tp = fp = fn = 0
    
    for text, annotations in test_data:
        doc = nlp(text)
        predicted = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
        true_entities = set(annotations["entities"])
        
        tp += len(predicted & true_entities)
        fp += len(predicted - true_entities)
        fn += len(true_entities - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def main():
    logger, log_file = setup_logging()
    config = get_config()
    
    print("="*100)
    print("🎓 ROBUST NER MODEL TRAINING (Background-Ready)")
    print("="*100)
    print()
    print(f"📝 Logging to: {log_file}")
    print("   Monitor progress: tail -f " + str(log_file))
    print()
    
    # Load data
    logger.info("Loading training data...")
    training_file = config.PROCESSED_DATA_DIR / 'ner_training_data_robust.json'
    
    if not training_file.exists():
        logger.error(f"Training data not found: {training_file}")
        logger.error("Run: python scripts/generate_robust_ner_data.py")
        return
    
    raw_data = load_training_data(training_file)
    logger.info(f"Loaded {  len(raw_data)} samples")
    
    # Convert to spaCy format
    logger.info("Converting to spaCy format...")
    spacy_data, skipped = convert_to_spacy_format(raw_data)
    logger.info(f"Converted {len(spacy_data)} samples ({skipped} skipped due to overlaps)")
    
    # Get entity labels
    entity_labels = set()
    for _, annotations in spacy_data:
        for _, _, label in annotations["entities"]:
            entity_labels.add(label)
    
    logger.info(f"Entity labels: {', '.join(sorted(entity_labels))}")
    
    # Split data
    logger.info("Splitting data (80/20)...")
    random.shuffle(spacy_data)
    split_point = int(len(spacy_data) * 0.8)
    train_data = spacy_data[:split_point]
    test_data = spacy_data[split_point:]
    
    logger.info(f"Training: {len(train_data)} samples")
    logger.info(f"Test: {len(test_data)} samples")
    
    # Create model
    logger.info("Creating blank NER model...")
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    for label in entity_labels:
        ner.add_label(label)
    
    logger.info(f"Model created with {len(entity_labels)} entity types")
    
    # Train
    logger.info("="*100)
    logger.info("TRAINING START")
    logger.info("="*100)
    
    nlp, best_f1 = train_ner_model(nlp, train_data, test_data, logger, config, n_iter=50)
    
    # Final evaluation
    logger.info("="*100)
    logger.info("FINAL EVALUATION")
    logger.info("="*100)
    
    final_metrics = evaluate_model(nlp, test_data, logger)
    logger.info(f"Precision: {final_metrics['precision']:.4f}")
    logger.info(f"Recall: {final_metrics['recall']:.4f}")
    logger.info(f"F1-Score: {final_metrics['f1_score']:.4f}")
    logger.info(f"Best F1 during training: {best_f1:.4f}")
    
    # Save final model
    model_output_dir = config.NER_MODELS_DIR / 'ran_ner_model_robust'
    nlp.to_disk(model_output_dir)
    logger.info(f"Final model saved: {model_output_dir}")
    
    metrics_file = config.NER_MODELS_DIR / 'ner_metrics_robust.json'
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")
    
    print()
    print("="*100)
    print("✅ TRAINING COMPLETE!")
    print("="*100)
    print()
    print(f"📊 Final Results:")
    print(f"   Precision: {final_metrics['precision']:.2%}")
    print(f"   Recall: {final_metrics['recall']:.2%}")
    print(f"   F1-Score: {final_metrics['f1_score']:.2%}")
    print()
    print(f"💾 Model saved to: {model_output_dir}")
    print()
    print("🎯 Next: Validate with robustness test")
    print("   python scripts/validate_ner_robustness.py")
    print()


if __name__ == "__main__":
    main()
