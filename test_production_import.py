#!/usr/bin/env python3
"""
RAN Import - Production Test
============================

Test importing larger chunks of the RAN data to verify the system works.
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
import pandas as pd

def test_chunk_import():
    """Test importing a reasonable chunk of data (10,000-50,000 rows)."""
    
    print("🏗️  RAN Chunk Import Test")
    print("🎯 Testing with manageable data chunks")
    print("=" * 50)
    
    # Setup configuration
    config = get_config()
    config.create_directories()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize importer
    db_path = config.DATABASE_DIR / "ran_chunk_test.db"
    importer = RANCSVImporter(str(db_path))
    
    # Test data directory
    data_dir = PROJECT_ROOT / "data" / "raw"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw")
        return False
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Test with different chunk sizes
    chunk_sizes = [10000, 25000, 50000]  # 10k, 25k, 50k rows
    
    for chunk_size in chunk_sizes:
        print(f"\n📊 Testing with {chunk_size:,} rows")
        print("-" * 40)
        
        for i, csv_file in enumerate(csv_files):
            file_name = csv_file.stem
            tech = importer.detect_ran_technology(str(csv_file))
            
            print(f"📄 Processing {file_name} ({tech}) - {chunk_size:,} rows")
            
            try:
                # Read a chunk of data
                print(f"   📖 Reading {chunk_size:,} rows...")
                start_read = time.time()
                chunk_df = pd.read_csv(csv_file, nrows=chunk_size)
                read_time = time.time() - start_read
                print(f"   ⏱️  Read time: {read_time:.1f}s")
                
                # Create temporary file
                temp_file = PROJECT_ROOT / f"temp_{file_name}_{chunk_size}.csv"
                chunk_df.to_csv(temp_file, index=False)
                
                # Import the chunk
                print(f"   💾 Importing...")
                start_import = time.time()
                
                table_name = f"{file_name}_{chunk_size}_rows"
                success = importer.import_ran_csv(str(temp_file), table_name)
                
                import_time = time.time() - start_import
                
                # Clean up temp file
                temp_file.unlink()
                
                if success:
                    print(f"   ✅ Import completed in {import_time:.1f}s")
                    
                    # Show table info
                    table_info = importer.get_table_info(table_name)
                    if table_info:
                        rows = table_info['row_count']
                        cols = len(table_info['columns'])
                        rate = rows / import_time if import_time > 0 else 0
                        print(f"   📊 Result: {rows:,} rows, {cols} columns")
                        print(f"   ⚡ Rate: {rate:,.0f} rows/second")
                else:
                    print(f"   ❌ Import failed")
                    return False
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
                return False
    
    # Final summary
    print(f"\n🎉 All chunk tests completed successfully!")
    
    # Show database summary
    tables = importer.list_tables()
    total_rows = 0
    
    print(f"\n🗄️  Database Summary:")
    print(f"   Database: {db_path}")
    print(f"   Tables: {len(tables)}")
    
    for table in sorted(tables):
        table_info = importer.get_table_info(table)
        if table_info:
            rows = table_info['row_count']
            total_rows += rows
            print(f"   - {table}: {rows:,} rows")
    
    print(f"   Total rows imported: {total_rows:,}")
    
    return True

def main():
    """Main test function."""
    
    print("🚀 RAN Import Module - Production Testing")
    print("🎯 Verifying the system can handle real RAN data efficiently")
    print()
    
    try:
        success = test_chunk_import()
        
        if success:
            print("\n" + "="*60)
            print("🎉 SUCCESS! The import module is working correctly!")
            print("✅ Ready to proceed with full data import or next module")
            print("\n🚀 Next steps:")
            print("   1. Import full datasets (will take time due to size)")
            print("   2. Move to database_structure_module for schema analysis")
            print("   3. Train NER models with imported data")
            return True
        else:
            print("\n❌ Testing failed. Please check the errors above.")
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
