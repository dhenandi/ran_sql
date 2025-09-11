#!/usr/bin/env python3
"""
Data Import Script
==================

Script to import CSV files into the SQLite database using the import module.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.import_module import CSVImporter
from config.settings import get_config
import logging


def setup_logging():
    """Setup logging for the script."""
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    return logging.getLogger(__name__)


def import_csv_file(csv_path: str, table_name: str = None):
    """
    Import a single CSV file.
    
    Args:
        csv_path: Path to the CSV file
        table_name: Name for the database table (optional)
    """
    logger = setup_logging()
    config = get_config()
    
    # Ensure database directory exists
    config.create_directories()
    
    # Initialize importer
    importer = CSVImporter(str(config.DATABASE_PATH))
    
    # Generate table name if not provided
    if not table_name:
        table_name = Path(csv_path).stem.lower().replace('-', '_')
    
    logger.info(f"Importing {csv_path} to table '{table_name}'")
    
    # Import the file
    success = importer.import_csv(csv_path, table_name)
    
    if success:
        logger.info(f"✅ Successfully imported {csv_path}")
        
        # Display table information
        table_info = importer.get_table_info(table_name)
        if table_info:
            logger.info(f"Table info: {table_info}")
    else:
        logger.error(f"❌ Failed to import {csv_path}")
    
    return success


def import_directory(directory_path: str):
    """
    Import all CSV files from a directory.
    
    Args:
        directory_path: Path to directory containing CSV files
    """
    logger = setup_logging()
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return False
    
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {directory_path}")
        return True
    
    logger.info(f"Found {len(csv_files)} CSV files to import")
    
    success_count = 0
    for csv_file in csv_files:
        if import_csv_file(str(csv_file)):
            success_count += 1
    
    logger.info(f"Successfully imported {success_count}/{len(csv_files)} files")
    return success_count == len(csv_files)


def list_tables():
    """List all tables in the database."""
    logger = setup_logging()
    config = get_config()
    
    if not config.DATABASE_PATH.exists():
        logger.warning("Database does not exist yet")
        return
    
    importer = CSVImporter(str(config.DATABASE_PATH))
    tables = importer.list_tables()
    
    if tables:
        logger.info("Tables in database:")
        for table in tables:
            logger.info(f"  - {table}")
    else:
        logger.info("No tables found in database")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Import CSV data into SQLite database")
    parser.add_argument("--file", "-f", help="CSV file to import")
    parser.add_argument("--directory", "-d", help="Directory containing CSV files")
    parser.add_argument("--table", "-t", help="Table name (for single file import)")
    parser.add_argument("--list", "-l", action="store_true", help="List existing tables")
    
    args = parser.parse_args()
    
    if args.list:
        list_tables()
    elif args.file:
        import_csv_file(args.file, args.table)
    elif args.directory:
        import_directory(args.directory)
    else:
        # Default: import sample data
        config = get_config()
        sample_file = config.RAW_DATA_DIR / "sample_ran_data.csv"
        
        if sample_file.exists():
            print(f"Importing sample data from {sample_file}")
            import_csv_file(str(sample_file), "ran_performance")
        else:
            print("No arguments provided and no sample data found.")
            print("Usage examples:")
            print("  python scripts/import_data.py --file data/raw/my_data.csv")
            print("  python scripts/import_data.py --directory data/raw/")
            print("  python scripts/import_data.py --list")


if __name__ == "__main__":
    main()
