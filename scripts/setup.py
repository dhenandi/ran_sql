#!/usr/bin/env python3
"""
Setup Script for RAN SQL Question Answering System
==================================================

This script initializes the project environment and sets up necessary directories.
"""

import sys
import subprocess
from pathlib import Path
import logging

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


def setup_logging():
    """Setup logging configuration."""
    config = get_config()
    config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def install_dependencies():
    """Install required Python packages."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def download_spacy_model():
    """Download spaCy model for NER training."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        logger.info("spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to download spaCy model: {e}")
        logger.info("You can download it later using: python -m spacy download en_core_web_sm")
        return False


def create_sample_data():
    """Create sample RAN data for testing."""
    logger = logging.getLogger(__name__)
    config = get_config()
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("Creating sample RAN data...")
        
        # Generate sample cell performance data
        np.random.seed(42)
        n_records = 1000
        
        sample_data = {
            'timestamp': [
                (datetime.now() - timedelta(hours=i//10)).strftime('%Y-%m-%d %H:%M:%S')
                for i in range(n_records)
            ],
            'cell_id': [f"CELL_{i:06d}" for i in np.random.randint(1, 100, n_records)],
            'site_id': [f"SITE_{i:04d}" for i in np.random.randint(1, 20, n_records)],
            'sector_id': np.random.randint(1, 4, n_records),
            'enb_id': np.random.randint(1000, 2000, n_records),
            'rsrp': np.random.normal(-85, 10, n_records),  # dBm
            'rsrq': np.random.normal(-10, 3, n_records),   # dB
            'sinr': np.random.normal(15, 5, n_records),    # dB
            'throughput': np.random.exponential(50, n_records),  # Mbps
            'latency': np.random.exponential(20, n_records),     # ms
            'bler': np.random.beta(1, 10, n_records) * 100,     # %
            'cqi': np.random.randint(1, 16, n_records)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        sample_file = config.RAW_DATA_DIR / "sample_ran_data.csv"
        df.to_csv(sample_file, index=False)
        
        logger.info(f"Sample data created: {sample_file}")
        logger.info(f"Records: {len(df)}, Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


def validate_installation():
    """Validate that the installation was successful."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test imports
        logger.info("Validating installation...")
        
        import pandas
        import numpy
        import spacy
        import transformers
        import streamlit
        
        logger.info("Core packages imported successfully")
        
        # Test spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found - download manually if needed")
        
        # Test project imports
        from src.import_module import CSVImporter
        from src.database_structure_module import DatabaseAnalyzer
        
        logger.info("Project modules imported successfully")
        logger.info("Installation validation completed")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error during validation: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up RAN SQL Question Answering System...")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting project setup...")
    
    # Create directories
    config = get_config()
    config.create_directories()
    logger.info("Project directories created")
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed during dependency installation")
        return False
    
    # Download spaCy model
    download_spacy_model()
    
    # Create sample data
    if create_sample_data():
        logger.info("Sample data created successfully")
    
    # Validate installation
    if validate_installation():
        logger.info("✅ Setup completed successfully!")
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Import your CSV data: python scripts/import_data.py")
        print("2. Analyze database structure: python scripts/analyze_schema.py")
        print("3. Train models: python scripts/train_models.py")
        print("4. Launch web interface: streamlit run src/question_answering_module/streamlit_app.py")
        return True
    else:
        logger.error("❌ Setup validation failed")
        print("❌ Setup validation failed - check logs for details")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
