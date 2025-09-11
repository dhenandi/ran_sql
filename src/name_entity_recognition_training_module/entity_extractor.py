"""
Entity Extractor
================

Extracts entities from user queries using trained NER models.
"""

import logging
from typing import List, Dict, Optional, Any
from .ner_trainer import NERTrainer


class EntityExtractor:
    """
    Extracts RAN entities from natural language queries.
    """
    
    def __init__(self, model_path: str, model_type: str = "spacy"):
        """
        Initialize the entity extractor.
        
        Args:
            model_path: Path to the trained NER model
            model_type: Type of model ("spacy" or "transformers")
        """
        self.model_path = model_path
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Load the trained model
        self.trainer = NERTrainer(model_type=model_type)
        self.trainer.load_model(model_path)
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Extract entities from input text.
        
        Args:
            text: Input text to extract entities from
            confidence_threshold: Minimum confidence score for entities
            
        Returns:
            List[Dict]: Extracted entities with metadata
        """
        try:
            # Use the trainer's predict method
            raw_entities = self.trainer.predict(text)
            
            # Filter by confidence and add metadata
            filtered_entities = []
            for entity in raw_entities:
                if entity.get('confidence', 1.0) >= confidence_threshold:
                    filtered_entities.append({
                        'text': entity['text'],
                        'label': entity['label'],
                        'start': entity.get('start'),
                        'end': entity.get('end'),
                        'confidence': entity.get('confidence', 1.0),
                        'context': self._get_entity_context(text, entity)
                    })
            
            self.logger.info(f"Extracted {len(filtered_entities)} entities from text")
            return filtered_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _get_entity_context(self, text: str, entity: Dict) -> str:
        """
        Get surrounding context for an entity.
        
        Args:
            text: Full input text
            entity: Entity information
            
        Returns:
            str: Context around the entity
        """
        # Implementation placeholder
        return ""
    
    def get_entity_types(self) -> List[str]:
        """
        Get supported entity types.
        
        Returns:
            List[str]: List of supported entity labels
        """
        return self.trainer.ran_labels if hasattr(self.trainer, 'ran_labels') else []
