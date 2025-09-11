#!/usr/bin/env python3
"""
RAN Import Test - Large File Edition
====================================

Test the import functionality with large CSV files using chunked processing.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.import_module.csv_importer import RANCSVImporter
from config.settings import get_config
import logging
import time

def test_large_file_import():
    """Test importing large CSV files with progress monitoring."""
    
    print("🏗️  RAN Large File Import Test")
    print("=" * 50)
    
    # Setup configuration
    config = get_config()
    config.create_directories()
    
    # Setup detailed logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOGS_DIR / 'import_test.log')
        ]
    )
    
    # Initialize importer
    db_path = config.DATABASE_DIR / "ran_large_test.db"
    importer = RANCSVImporter(str(db_path))
    
    # Test data directory
    data_dir = PROJECT_ROOT / "data" / "raw"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw")
        return False
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Get file sizes
    for csv_file in csv_files:
        size_gb = csv_file.stat().st_size / (1024**3)
        tech = importer.detect_ran_technology(str(csv_file))
        print(f"   - {csv_file.name}: {size_gb:.2f} GB ({tech})")
    
    # Import each file one by one
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n📊 Processing file {i}/{len(csv_files)}: {csv_file.name}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            success = importer.import_ran_csv(str(csv_file))
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                print(f"✅ Import completed in {duration:.1f} seconds")
                
                # Show table info
                tables = importer.list_tables()
                if tables:
                    table_name = tables[-1]  # Get the last created table
                    table_info = importer.get_table_info(table_name)
                    if table_info:
                        row_count = table_info['row_count']
                        col_count = len(table_info['columns'])
                        print(f"📊 Table '{table_name}': {row_count:,} rows, {col_count} columns")
                        
                        # Calculate processing rate
                        rows_per_sec = row_count / duration if duration > 0 else 0
                        print(f"⚡ Processing rate: {rows_per_sec:,.0f} rows/second")
            else:
                print(f"❌ Import failed after {duration:.1f} seconds")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️  Import interrupted by user")
            return False
        except Exception as e:
            print(f"💥 Error during import: {e}")
            return False
    
    # Final summary
    print(f"\n🎉 All {len(csv_files)} files imported successfully!")
    
    # Show final database summary
    tables = importer.list_tables()
    total_rows = 0
    
    print(f"\n🗄️  Final Database Summary:")
    print(f"   Database: {db_path}")
    print(f"   Tables created: {len(tables)}")
    
    for table in tables:
        table_info = importer.get_table_info(table)
        if table_info:
            rows = table_info['row_count']
            total_rows += rows
            print(f"   - {table}: {rows:,} rows")
    
    print(f"   Total rows: {total_rows:,}")
    
    return True

def test_sample_import():
    """Test with just a small sample first."""
    
    print("🧪 Sample Data Test")
    print("=" * 30)
    
    # Setup
    config = get_config()
    db_path = config.DATABASE_DIR / "ran_sample_test.db"
    importer = RANCSVImporter(str(db_path))
    
    # Test with first 1000 rows only
    data_dir = PROJECT_ROOT / "data" / "raw"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    test_file = csv_files[0]  # Test with first file
    print(f"📄 Testing sample from: {test_file.name}")
    
    try:
        # Create a temporary sample file
        import pandas as pd
        
        print("📖 Reading sample (first 1000 rows)...")
        sample_df = pd.read_csv(test_file, nrows=1000)
        
        sample_file = PROJECT_ROOT / "data" / "temp_sample.csv"
        sample_df.to_csv(sample_file, index=False)
        print(f"💾 Created sample file: {sample_file}")
        
        # Import the sample
        success = importer.import_ran_csv(str(sample_file), "sample_test")
        
        # Clean up
        sample_file.unlink()
        
        if success:
            print("✅ Sample import successful!")
            table_info = importer.get_table_info("sample_test")
            if table_info:
                print(f"📊 Sample table: {table_info['row_count']} rows, {len(table_info['columns'])} columns")
            return True
        else:
            print("❌ Sample import failed!")
            return False
            
    except Exception as e:
        print(f"💥 Sample test error: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 RAN Large File Import Testing")
    print("🎯 Handling multi-gigabyte CSV files efficiently")
    print()
    
    try:
        # First test with a small sample
        print("Step 1: Testing with sample data...")
        sample_success = test_sample_import()
        
        if not sample_success:
            print("❌ Sample test failed. Stopping here.")
            return False
        
        print("\n" + "="*60)
        input("✅ Sample test passed! Press Enter to continue with full import...")
        
        # Then test with full files
        print("\nStep 2: Testing with full large files...")
        full_success = test_large_file_import()
        
        if full_success:
            print("\n🎉 All tests completed successfully!")
            print("✅ The import module can handle your large RAN CSV files!")
            return True
        else:
            print("\n❌ Full import test failed.")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
