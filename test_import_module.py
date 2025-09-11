#!/usr/bin/env python3
"""
RAN Import Module Test Script
=============================

Test the import_module functionality with actual RAN CSV data.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import specific modules without triggering all imports
from src.import_module.csv_importer import RANCSVImporter
from config.settings import get_config
import logging

def test_ran_import():
    """Test importing the actual RAN CSV files."""
    
    print("🧪 Testing RAN CSV Import Module")
    print("=" * 50)
    
    # Setup configuration
    config = get_config()
    config.create_directories()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize importer
    db_path = config.DATABASE_DIR / "ran_performance.db"
    importer = RANCSVImporter(str(db_path))
    
    # Test data directory
    data_dir = PROJECT_ROOT / "data" / "raw"
    
    if not data_dir.exists():
        print("❌ Data directory not found: data/raw")
        return False
    
    # Find CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw")
        return False
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    print("\n🔍 Testing individual file detection:")
    # Test detection for each file
    for csv_file in csv_files:
        tech = importer.detect_ran_technology(str(csv_file))
        print(f"   - {csv_file.name}: {tech}")
    
    print("\n📊 Starting import process:")
    # Import all files
    results = importer.import_all_ran_files(str(data_dir))
    
    print("\n📈 Import Results:")
    success_count = 0
    for file_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   - {file_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(results)} files imported successfully")
    
    # Show database info
    if success_count > 0:
        print("\n🗄️  Database Information:")
        tables = importer.list_tables()
        print(f"   Created tables: {len(tables)}")
        
        for table in tables:
            table_info = importer.get_table_info(table)
            if table_info:
                print(f"   - {table}: {table_info['row_count']} rows, {len(table_info['columns'])} columns")
    
    return success_count == len(csv_files)

def test_single_file_import():
    """Test importing a single file with detailed output."""
    
    print("\n" + "=" * 50)
    print("🔬 Detailed Single File Import Test")
    print("=" * 50)
    
    # Setup
    config = get_config()
    db_path = config.DATABASE_DIR / "test_single.db"
    importer = RANCSVImporter(str(db_path))
    
    # Find first CSV file
    data_dir = PROJECT_ROOT / "data" / "raw"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    test_file = csv_files[0]
    print(f"📄 Testing with: {test_file.name}")
    
    # Detect technology
    tech = importer.detect_ran_technology(str(test_file))
    print(f"🔍 Detected technology: {tech}")
    
    # Import
    success = importer.import_ran_csv(str(test_file), f"test_{tech.lower()}")
    
    if success:
        print("✅ Import successful!")
        
        # Show detailed info
        table_info = importer.get_table_info(f"test_{tech.lower()}")
        if table_info:
            print(f"📊 Table Info:")
            print(f"   - Rows: {table_info['row_count']}")
            print(f"   - Columns: {len(table_info['columns'])}")
            print(f"   - Column details:")
            for col in table_info['columns'][:10]:  # Show first 10 columns
                print(f"     • {col['name']}: {col['type']}")
            if len(table_info['columns']) > 10:
                print(f"     ... and {len(table_info['columns']) - 10} more columns")
        
        return True
    else:
        print("❌ Import failed!")
        return False

def main():
    """Main test function."""
    
    print("🏗️  RAN SQL Import Module Test")
    print("🎯 Testing with actual CSV data from data/raw/")
    print()
    
    try:
        # Test 1: Import all files
        test1_success = test_ran_import()
        
        # Test 2: Detailed single file test
        test2_success = test_single_file_import()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"   All files import: {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"   Single file test: {'✅ PASS' if test2_success else '❌ FAIL'}")
        
        if test1_success and test2_success:
            print("\n🎉 All tests passed! Import module is working correctly.")
            print("\n🚀 Next steps:")
            print("   1. Use the imported data for database structure analysis")
            print("   2. Train NER models using the imported RAN data")
            print("   3. Generate SQL training data from the schema")
            
            return True
        else:
            print("\n❌ Some tests failed. Check the error messages above.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
