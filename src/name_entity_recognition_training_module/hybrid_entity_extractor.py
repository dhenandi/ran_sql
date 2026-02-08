"""
Hybrid Entity Extractor
========================

Combines dictionary-based lookup with NER model for improved RAN domain entity extraction.
This approach preserves high specificity while adding domain-specific recall.
"""

import re
import spacy
from typing import List, Dict, Tuple, Set
from pathlib import Path


class HybridEntityExtractor:
    """
    Hybrid entity extraction combining:
    1. Dictionary lookup for known RAN technical terms
    2. Pattern matching for numeric values and identifiers
    3. NER model for general entities
    4. Post-processing to resolve conflicts
    """
    
    def __init__(self, ner_model_path: str):
        """
        Initialize hybrid extractor.
        
        Args:
            ner_model_path: Path to trained spaCy NER model
        """
        self.nlp = spacy.load(ner_model_path)
        self._load_dictionaries()
        self._compile_patterns()
    
    def _load_dictionaries(self):
        """Load domain-specific dictionaries."""
        # RAN KPI names (technical acronyms and terms)
        self.kpi_dictionary = {
            # Signal quality metrics
            'rsrp', 'rsrq', 'sinr', 'rssi', 'cqi', 'bler',
            # Throughput and capacity
            'throughput', 'bandwidth', 'capacity', 'traffic',
            'ul_throughput', 'dl_throughput', 'uplink', 'downlink',
            # Performance metrics
            'latency', 'delay', 'jitter', 'packet_loss',
            # Call metrics
            'call_drop', 'drop_rate', 'drop_call', 'call_setup',
            'call_success', 'success_rate', 'setup_time',
            # Availability metrics
            'availability', 'uptime', 'downtime', 'outage',
            # Utilization
            'utilization', 'usage', 'load', 'congestion',
            # 2G specific
            'ccalls', 'ccongs', 'cndrop', 'tava', 'pmcount',
            # 4G specific
            'pm_count', 'pmcell', 'pmera', 'pmho', 'pmuethp',
            # Units
            'dbm', 'db', 'mbps', 'kbps', 'gbps', 'ms', 'mhz', 'ghz'
        }
        
        # Common location words (Indonesian cities/regions)
        self.location_dictionary = {
            # Major cities
            'jakarta', 'bandung', 'surabaya', 'medan', 'semarang',
            'makassar', 'palembang', 'tangerang', 'depok', 'bekasi',
            'bogor', 'malang', 'yogyakarta', 'denpasar', 'bali',
            # Regions/provinces
            'java', 'sumatra', 'kalimantan', 'sulawesi', 'papua',
            'banten', 'jawa', 'timur', 'tengah', 'barat', 'utara', 'selatan',
            # Directional qualifiers
            'east', 'west', 'north', 'south', 'central'
        }
        
        # Site/Cell identifier patterns (will be in pattern matching)
        self.identifier_prefixes = {
            'site', 'cell', 'enb', 'gnb', 'bts', 'nodeb', 'sector'
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for entity matching."""
        # Site/Cell IDs (e.g., SITE_001, JKT001, ENB_123)
        self.site_id_pattern = re.compile(
            r'\b(?:SITE|JKT|BDG|SBY|ENB|GNB|BTS|CELL)[_\-]?\d{1,6}\b',
            re.IGNORECASE
        )
        
        # Numeric values (integers, floats, with optional negative sign)
        self.numeric_pattern = re.compile(
            r'-?\d+(?:\.\d+)?(?:\s*(?:dbm|db|mbps|kbps|gbps|ms|mhz|ghz))?',
            re.IGNORECASE
        )
        
        # Date/time patterns (simplified)
        self.date_pattern = re.compile(
            r'\b\d{4}-\d{2}-\d{2}|\b\d{2}/\d{2}/\d{4}|\b(?:yesterday|today|tomorrow|last\s+\w+)\b',
            re.IGNORECASE
        )
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using hybrid approach.
        
        Args:
            text: Input natural language text
            
        Returns:
            List of entity dictionaries with text, label, start, end
        """
        entities = []
        
        # Step 1: Dictionary-based extraction (highest priority)
        dict_entities = self._extract_with_dictionary(text)
        
        # Step 2: Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text)
        
        # Step 3: NER model extraction
        ner_entities = self._extract_with_ner(text)
        
        # Step 4: Merge and resolve conflicts
        entities = self._merge_entities(
            text, dict_entities, pattern_entities, ner_entities
        )
        
        # Step 5: Post-processing corrections
        entities = self._post_process(text, entities)
        
        return entities
    
    def _extract_with_dictionary(self, text: str) -> List[Dict]:
        """Extract entities using dictionary lookup."""
        entities = []
        text_lower = text.lower()
        
        # Find KPI terms
        for kpi in self.kpi_dictionary:
            # Use word boundaries to avoid partial matches
            pattern = re.compile(r'\b' + re.escape(kpi) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append({
                    'text': match.group(),
                    'label': 'KPI_NAME',
                    'start': match.start(),
                    'end': match.end(),
                    'source': 'dictionary',
                    'confidence': 1.0
                })
        
        # Find location terms (only if not part of compound technical term)
        for location in self.location_dictionary:
            pattern = re.compile(r'\b' + re.escape(location) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                # Check if it's part of a larger phrase (e.g., "Central Java")
                start_ctx = max(0, match.start() - 10)
                end_ctx = min(len(text), match.end() + 10)
                context = text[start_ctx:end_ctx].lower()
                
                # More likely to be location if preceded by "in", "at", "from", etc.
                location_indicators = ['in ', 'at ', 'from ', 'for ', 'to ']
                is_location_context = any(ind in context[:match.start()-start_ctx+5] 
                                         for ind in location_indicators)
                
                entities.append({
                    'text': match.group(),
                    'label': 'LOCATION',
                    'start': match.start(),
                    'end': match.end(),
                    'source': 'dictionary',
                    'confidence': 0.9 if is_location_context else 0.6
                })
        
        return entities
    
    def _extract_with_patterns(self, text: str) -> List[Dict]:
        """Extract entities using pattern matching."""
        entities = []
        
        # Site/Cell IDs
        for match in self.site_id_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'SITE_ID',
                'start': match.start(),
                'end': match.end(),
                'source': 'pattern',
                'confidence': 0.95
            })
        
        # Numeric values (only if they look like thresholds/measurements)
        for match in self.numeric_pattern.finditer(text):
            matched_text = match.group()
            
            # Context check: is this a measurement value?
            start_ctx = max(0, match.start() - 15)
            context = text[start_ctx:match.start()].lower()
            
            # Look for measurement context words
            measurement_words = ['above', 'below', 'than', 'with', 'of', 
                                '>', '<', '=', 'greater', 'less', 'equal']
            is_measurement = any(word in context for word in measurement_words)
            
            if is_measurement or matched_text.startswith('-'):
                entities.append({
                    'text': matched_text,
                    'label': 'NUMERIC_VALUE',
                    'start': match.start(),
                    'end': match.end(),
                    'source': 'pattern',
                    'confidence': 0.85 if is_measurement else 0.6
                })
        
        # Dates (simplified)
        for match in self.date_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'DATE_TIME',
                'start': match.start(),
                'end': match.end(),
                'source': 'pattern',
                'confidence': 0.8
            })
        
        return entities
    
    def _extract_with_ner(self, text: str) -> List[Dict]:
        """Extract entities using trained NER model."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'source': 'ner_model',
                'confidence': 0.7  # Lower confidence than dictionary
            })
        
        return entities
    
    def _merge_entities(self, text: str, dict_entities: List[Dict], 
                       pattern_entities: List[Dict], ner_entities: List[Dict]) -> List[Dict]:
        """
        Merge entities from different sources, resolving conflicts.
        
        Priority: Dictionary > Pattern > NER
        """
        all_entities = dict_entities + pattern_entities + ner_entities
        
        # Sort by position
        all_entities.sort(key=lambda x: (x['start'], -x['confidence']))
        
        # Remove overlaps (keep highest confidence)
        merged = []
        for entity in all_entities:
            # Check if it overlaps with any existing entity
            overlaps = False
            for existing in merged:
                if self._spans_overlap(entity, existing):
                    overlaps = True
                    # If new entity has higher confidence, replace
                    if entity['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(entity)
                    break
            
            if not overlaps:
                merged.append(entity)
        
        # Sort by position again
        merged.sort(key=lambda x: x['start'])
        
        return merged
    
    def _spans_overlap(self, span1: Dict, span2: Dict) -> bool:
        """Check if two entity spans overlap."""
        return not (span1['end'] <= span2['start'] or span2['end'] <= span1['start'])
    
    def _post_process(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Post-process entities to correct obvious mistakes.
        """
        corrected = []
        
        for entity in entities:
            entity_text = entity['text'].lower()
            
            # Correction 1: If labeled as KPI but is actually a location word
            if entity['label'] == 'KPI_NAME' and entity_text in self.location_dictionary:
                # Check context: if preceded by "in", "at", "from", it's a location
                start_ctx = max(0, entity['start'] - 10)
                context = text[start_ctx:entity['start']].lower()
                if any(word in context for word in ['in ', 'at ', 'from ', 'for ']):
                    entity['label'] = 'LOCATION'
            
            # Correction 2: If labeled as LOCATION but is actually a KPI
            if entity['label'] == 'LOCATION' and entity_text in self.kpi_dictionary:
                entity['label'] = 'KPI_NAME'
            
            # Correction 3: Numeric values shouldn't be labeled as KPI_NAME
            if entity['label'] == 'KPI_NAME' and re.match(r'^-?\d+(?:\.\d+)?$', entity_text):
                entity['label'] = 'NUMERIC_VALUE'
            
            # Correction 4: Site IDs that are labeled incorrectly
            if self.site_id_pattern.match(entity_text):
                entity['label'] = 'SITE_ID'
            
            corrected.append(entity)
        
        return corrected
    
    def format_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Format entities for output (remove internal fields).
        
        Args:
            entities: Raw entities with confidence and source
            
        Returns:
            Cleaned entity list
        """
        return [
            {
                'text': ent['text'],
                'label': ent['label'],
                'start': ent['start'],
                'end': ent['end']
            }
            for ent in entities
        ]
