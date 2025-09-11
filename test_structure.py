#!/usr/bin/env python3
"""
Quick Test Script
=================

Test basic imports and structure of the RAN SQL QA system.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test configuration
        from config.settings import get_config
        config = get_config()
        print("✅ Configuration module imported successfully")
        
        # Test import module
        from src.import_module import CSVImporter, DataValidator, SchemaOptimizer
        print("✅ Import module imported successfully")
        
        # Test database structure module
        from src.database_structure_module import DatabaseAnalyzer, SchemaMapper
        print("✅ Database structure module imported successfully")
        
        # Test NER module
        from src.name_entity_recognition_training_module import NERTrainer
        print("✅ NER training module imported successfully")
        
        # Test SQL module
        from src.sql_model_generation_module import QueryGenerator, SQLValidator
        print("✅ SQL model generation module imported successfully")
        
        # Test QA module
        from src.question_answering_module import QuestionAnsweringPipeline
        print("✅ Question answering module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test configuration
        from config.settings import get_config
        config = get_config()
        config.create_directories()
        print("✅ Directories created successfully")
        
        # Test CSV importer initialization
        from src.import_module import CSVImporter
        importer = CSVImporter(":memory:")  # In-memory database for testing
        print("✅ CSV importer initialized successfully")
        
        # Test database analyzer
        from src.database_structure_module import DatabaseAnalyzer
        analyzer = DatabaseAnalyzer(":memory:")
        print("✅ Database analyzer initialized successfully")
        
        # Test NER trainer
        from src.name_entity_recognition_training_module import NERTrainer
        trainer = NERTrainer()
        print("✅ NER trainer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def show_project_structure():
    """Display the project structure."""
    print("\n📁 Project Structure:")
    print("""
ran_sql/
├── src/
│   ├── import_module/                     ✅ CSV to SQLite import
│   ├── database_structure_module/        ✅ Schema analysis & mapping
│   ├── name_entity_recognition_training_module/  ✅ NER training
│   ├── sql_model_generation_module/      ✅ SQL query generation
│   └── question_answering_module/        ✅ End-to-end QA pipeline
├── config/                               ✅ Configuration settings
├── data/                                 ✅ Data storage
│   ├── raw/                             ✅ Original CSV files
│   ├── processed/                       ✅ Processed data
│   └── databases/                       ✅ SQLite databases
├── models/                              ✅ Trained models
├── tests/                               ✅ Test suite
├── scripts/                             ✅ Utility scripts
└── requirements.txt                     ✅ Dependencies
    """)

def show_next_steps():
    """Show next steps for development."""
    print("\n🚀 Next Steps:")
    print("""
1. 📦 Install dependencies:
   pip install -r requirements.txt

2. 🚀 Run setup script:
   python scripts/setup.py

3. 📊 Import your CSV data:
   python scripts/import_data.py --file your_data.csv

4. 🧠 Train NER model:
   # Implement training workflow in each module

5. 🔍 Train SQL generation model:
   # Implement training workflow in each module

6. 🌐 Launch web interface:
   streamlit run src/question_answering_module/streamlit_app.py

7. 🧪 Run tests:
   pytest tests/

Development Workflow:
- Start with import_module to get data into SQLite
- Use database_structure_module to analyze your schema
- Generate training data for your specific RAN dataset
- Train NER models to recognize your entities
- Train SQL generation models with your query patterns
- Integrate everything in the QA pipeline
- Launch the Streamlit interface for user interaction
    """)

def main():
    """Main test function."""
    print("🏗️  RAN SQL Question Answering System - Structure Test")
    print("=" * 60)
    
    # Test imports
    import_success = test_imports()
    
    # Test basic functionality
    if import_success:
        functionality_success = test_basic_functionality()
    else:
        functionality_success = False
    
    # Show structure
    show_project_structure()
    
    # Show next steps
    show_next_steps()
    
    # Summary
    print("\n" + "=" * 60)
    if import_success and functionality_success:
        print("✅ All tests passed! The modular structure is ready for development.")
        print("🎯 You can now proceed to implement the detailed functionality of each module.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return import_success and functionality_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
