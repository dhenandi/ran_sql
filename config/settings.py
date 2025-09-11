"""
Configuration Settings for RAN SQL Question Answering System
============================================================

This module contains all configuration settings for the system.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Main configuration class for the RAN SQL QA system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CONFIG_DIR = PROJECT_ROOT / "config"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    DATABASE_DIR = DATA_DIR / "databases"
    
    # Model paths
    NER_MODELS_DIR = MODELS_DIR / "ner"
    SQL_MODELS_DIR = MODELS_DIR / "sql_generation"
    
    # Database settings
    DATABASE_NAME = "ran_performance.db"
    DATABASE_PATH = DATABASE_DIR / DATABASE_NAME
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "ran_sql_qa.log"
    
    # Model settings
    NER_MODEL_TYPE = "spacy"  # or "transformers"
    NER_BASE_MODEL = "en_core_web_sm"
    SQL_MODEL_TYPE = "transformers"
    SQL_BASE_MODEL = "facebook/bart-large"
    
    # Training settings
    NER_TRAINING_EPOCHS = 30
    SQL_TRAINING_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    
    # API settings
    STREAMLIT_PORT = 8501
    API_TIMEOUT = 30
    
    # RAN specific settings
    RAN_ENTITY_LABELS = [
        "CELL_ID", "SITE_ID", "SECTOR_ID", "ENB_ID", "GNB_ID",
        "KPI_NAME", "KPI_VALUE", "TIMESTAMP", "DATE_TIME",
        "AGGREGATION", "COMPARISON_OP", "NUMERIC_VALUE",
        "TABLE_NAME", "COLUMN_NAME", "LOCATION"
    ]
    
    RAN_KPIS = [
        "rsrp", "rsrq", "sinr", "throughput", "latency", 
        "bler", "cqi", "pci", "earfcn", "bandwidth"
    ]
    
    # Validation settings
    MIN_CONFIDENCE_THRESHOLD = 0.7
    MAX_QUERY_LENGTH = 500
    MAX_RESULTS_DISPLAY = 1000
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.DATABASE_DIR, cls.MODELS_DIR, cls.NER_MODELS_DIR,
            cls.SQL_MODELS_DIR, cls.LOGS_DIR, cls.CONFIG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL."""
        return f"sqlite:///{cls.DATABASE_PATH}"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    NER_TRAINING_EPOCHS = 5  # Reduced for faster development
    SQL_TRAINING_EPOCHS = 3


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    NER_TRAINING_EPOCHS = 50
    SQL_TRAINING_EPOCHS = 20


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    DATABASE_NAME = "test_ran_performance.db"
    NER_TRAINING_EPOCHS = 1
    SQL_TRAINING_EPOCHS = 1


# Environment-based configuration selection
def get_config() -> Config:
    """Get configuration based on environment variable."""
    env = os.getenv('RAN_SQL_ENV', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)
