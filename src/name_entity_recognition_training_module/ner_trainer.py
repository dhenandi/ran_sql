"""
NER Trainer
===========

Trains and fine-tunes Named Entity Recognition models for RAN domain entities.
"""

import json
import pickle
import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class NERTrainer:
    """
    Trains NER models for RAN domain entity recognition.
    """
    
    def __init__(self, model_type: str = "spacy", model_name: str = "en_core_web_sm"):
        """
        Initialize the NER trainer.
        
        Args:
            model_type: Type of model to train ("spacy" or "transformers")
            model_name: Name of the base model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # RAN-specific entity labels
        self.ran_labels = [
            "CELL_ID", "SITE_ID", "SECTOR_ID", "ENB_ID", "GNB_ID",
            "KPI_NAME", "KPI_VALUE", "TIMESTAMP", "DATE_TIME",
            "AGGREGATION", "COMPARISON_OP", "NUMERIC_VALUE",
            "TABLE_NAME", "COLUMN_NAME", "LOCATION"
        ]
        
        self.model = None
        self.tokenizer = None
    
    def prepare_training_data(self, training_data: List[Dict]) -> List[Example]:
        """
        Prepare training data for spaCy model.
        
        Args:
            training_data: List of training examples with text and entities
            
        Returns:
            List[Example]: Prepared training examples
        """
        if self.model_type != "spacy":
            raise ValueError("This method is only for spaCy models")
        
        # Load base model
        if not self.model:
            try:
                self.model = spacy.load(self.model_name)
            except OSError:
                # If model not found, create blank model
                self.model = spacy.blank("en")
        
        # Add NER component if not present
        if "ner" not in self.model.pipe_names:
            ner = self.model.add_pipe("ner")
        else:
            ner = self.model.get_pipe("ner")
        
        # Add labels to NER component
        for label in self.ran_labels:
            ner.add_label(label)
        
        # Convert training data to spaCy format
        examples = []
        for item in training_data:
            text = item["text"]
            entities = item.get("entities", [])
            
            # Convert entity format
            spacy_entities = []
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                label = entity["label"]
                spacy_entities.append((start, end, label))
            
            # Create spaCy document
            doc = self.model.make_doc(text)
            example = Example.from_dict(doc, {"entities": spacy_entities})
            examples.append(example)
        
        self.logger.info(f"Prepared {len(examples)} training examples for spaCy")
        return examples
    
    def train_spacy_model(self, training_data: List[Dict], 
                         output_dir: str, n_iter: int = 30) -> str:
        """
        Train a spaCy NER model.
        
        Args:
            training_data: Training data with text and entities
            output_dir: Directory to save the trained model
            n_iter: Number of training iterations
            
        Returns:
            str: Path to the saved model
        """
        if self.model_type != "spacy":
            raise ValueError("This method is only for spaCy models")
        
        # Prepare training data
        examples = self.prepare_training_data(training_data)
        
        # Split into train/validation
        split_idx = int(0.8 * len(examples))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Get NER component
        ner = self.model.get_pipe("ner")
        
        # Disable other pipes during training
        disabled_pipes = [pipe for pipe in self.model.pipe_names if pipe != "ner"]
        
        with self.model.disable_pipes(*disabled_pipes):
            # Update model
            optimizer = self.model.begin_training()
            
            for iteration in range(n_iter):
                losses = {}
                
                # Shuffle training data
                np.random.shuffle(train_examples)
                
                # Update model with training examples
                for example in train_examples:
                    self.model.update([example], sgd=optimizer, losses=losses)
                
                # Evaluate on validation set
                if iteration % 10 == 0:
                    val_scores = self._evaluate_spacy_model(val_examples)
                    self.logger.info(f"Iteration {iteration}: Loss={losses.get('ner', 0):.3f}, "
                                   f"F1={val_scores.get('f1', 0):.3f}")
        
        # Save the model
        output_path = Path(output_dir) / "spacy_ner_model"
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.to_disk(output_path)
        
        self.logger.info(f"spaCy model saved to {output_path}")
        return str(output_path)
    
    def prepare_transformers_data(self, training_data: List[Dict]) -> Dict:
        """
        Prepare training data for Transformers model.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Dict: Tokenized data ready for training
        """
        if self.model_type != "transformers":
            raise ValueError("This method is only for Transformers models")
        
        # Initialize tokenizer
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create label mappings
        label_to_id = {label: i for i, label in enumerate(self.ran_labels)}
        label_to_id["O"] = len(self.ran_labels)  # Outside label
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        tokenized_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for item in training_data:
            text = item["text"]
            entities = item.get("entities", [])
            
            # Tokenize text
            tokens = self.tokenizer(text, truncation=True, padding="max_length", 
                                  max_length=128, return_tensors="pt")
            
            # Create label sequence
            labels = ["O"] * len(tokens["input_ids"][0])
            
            # Map entities to token positions
            for entity in entities:
                start_char = entity["start"]
                end_char = entity["end"]
                label = entity["label"]
                
                # Find token positions for character positions
                token_start = None
                token_end = None
                
                for i, token_id in enumerate(tokens["input_ids"][0]):
                    token_span = tokens.char_to_token(0, start_char)
                    if token_span is not None:
                        token_start = token_span
                        break
                
                if token_start is not None:
                    token_end_span = tokens.char_to_token(0, end_char - 1)
                    if token_end_span is not None:
                        token_end = token_end_span + 1
                        
                        # Assign labels using BIO tagging
                        labels[token_start] = f"B-{label}"
                        for j in range(token_start + 1, min(token_end, len(labels))):
                            labels[j] = f"I-{label}"
            
            # Convert labels to IDs
            label_ids = [label_to_id.get(label, label_to_id["O"]) for label in labels]
            
            tokenized_data["input_ids"].append(tokens["input_ids"][0])
            tokenized_data["attention_mask"].append(tokens["attention_mask"][0])
            tokenized_data["labels"].append(torch.tensor(label_ids))
        
        return tokenized_data, label_to_id, id_to_label
    
    def train_transformers_model(self, training_data: List[Dict], 
                                output_dir: str, epochs: int = 3) -> str:
        """
        Train a Transformers-based NER model.
        
        Args:
            training_data: Training data with text and entities
            output_dir: Directory to save the trained model
            epochs: Number of training epochs
            
        Returns:
            str: Path to the saved model
        """
        if self.model_type != "transformers":
            raise ValueError("This method is only for Transformers models")
        
        # Prepare data
        tokenized_data, label_to_id, id_to_label = self.prepare_transformers_data(training_data)
        
        # Initialize model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=len(label_to_id)
        )
        
        # Create dataset
        dataset = TransformersNERDataset(tokenized_data)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train model
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
        
        self.logger.info(f"Transformers model saved to {output_dir}")
        return output_dir
    
    def _evaluate_spacy_model(self, examples: List[Example]) -> Dict[str, float]:
        """
        Evaluate spaCy model performance.
        """
        tp = fp = fn = 0
        
        for example in examples:
            pred_doc = self.model(example.reference.text)
            true_entities = {(ent.start_char, ent.end_char, ent.label_) 
                           for ent in example.reference.ents}
            pred_entities = {(ent.start_char, ent.end_char, ent.label_) 
                           for ent in pred_doc.ents}
            
            tp += len(true_entities & pred_entities)
            fp += len(pred_entities - true_entities)
            fn += len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        if self.model_type == "spacy":
            self.model = spacy.load(model_path)
        elif self.model_type == "transformers":
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def predict(self, text: str) -> List[Dict]:
        """
        Predict entities in text using the trained model.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List[Dict]: Extracted entities with labels and positions
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        entities = []
        
        if self.model_type == "spacy":
            doc = self.model(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0  # spaCy doesn't provide confidence scores by default
                })
        
        elif self.model_type == "transformers":
            # Tokenize input
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**tokens)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)
            
            # Convert predictions back to entities
            # This is a simplified version - a full implementation would handle BIO tagging properly
            tokens_list = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
            
            for i, (token, label_id) in enumerate(zip(tokens_list, predicted_labels[0])):
                if label_id != 0:  # Not "O" label
                    # Load label mappings
                    label_mappings_path = Path(self.model_name) / "label_mappings.json"
                    if label_mappings_path.exists():
                        with open(label_mappings_path) as f:
                            mappings = json.load(f)
                            id_to_label = mappings["id_to_label"]
                            label = id_to_label.get(str(label_id.item()), "UNKNOWN")
                            
                            entities.append({
                                "text": token,
                                "label": label,
                                "confidence": float(torch.max(predictions[0][i]))
                            })
        
        return entities


class TransformersNERDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Transformers NER training.
    """
    
    def __init__(self, tokenized_data: Dict):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = tokenized_data["labels"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }
