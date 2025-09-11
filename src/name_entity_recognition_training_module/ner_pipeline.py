"""
NER Pipeline
============

End-to-end pipeline for Named Entity Recognition in RAN domain.
"""

import logging
from typing import Dict, List, Optional, Any
from .ner_trainer import NERTrainer
from .entity_extractor import EntityExtractor
from .model_evaluator import ModelEvaluator


class NERPipeline:
    """
    Complete NER pipeline for RAN domain entities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NER pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.trainer = None
        self.extractor = None
        self.evaluator = ModelEvaluator()
    
    def train_model(self, training_data: List[Dict], model_output_path: str) -> str:
        """
        Train a new NER model.
        
        Args:
            training_data: Training examples
            model_output_path: Path to save trained model
            
        Returns:
            str: Path to saved model
        """
        model_type = self.config.get('model_type', 'spacy')
        base_model = self.config.get('base_model', 'en_core_web_sm')
        
        self.trainer = NERTrainer(model_type=model_type, model_name=base_model)
        
        if model_type == 'spacy':
            return self.trainer.train_spacy_model(
                training_data, 
                model_output_path,
                n_iter=self.config.get('epochs', 30)
            )
        else:
            return self.trainer.train_transformers_model(
                training_data,
                model_output_path,
                epochs=self.config.get('epochs', 3)
            )
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model for entity extraction.
        
        Args:
            model_path: Path to the trained model
        """
        model_type = self.config.get('model_type', 'spacy')
        self.extractor = EntityExtractor(model_path, model_type)
        self.logger.info(f"Loaded NER model from {model_path}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using the loaded model.
        
        Args:
            text: Input text
            
        Returns:
            List[Dict]: Extracted entities
        """
        if not self.extractor:
            raise ValueError("No model loaded. Call load_model() first.")
        
        return self.extractor.extract_entities(text)
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test examples with ground truth
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.extractor:
            raise ValueError("No model loaded. Call load_model() first.")
        
        predictions = []
        ground_truth = []
        
        for example in test_data:
            text = example['text']
            true_entities = example.get('entities', [])
            pred_entities = self.extract_entities(text)
            
            predictions.append(pred_entities)
            ground_truth.append(true_entities)
        
        return self.evaluator.evaluate_model(predictions, ground_truth)
