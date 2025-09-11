#!/usr/bin/env python3
"""
CSV Analysis Script for RAN Data
"""

import pandas as pd
import os

def analyze_csv_structure():
    """Analyze the structure of both CSV files."""
    
    data_dir = "data/raw"
    files = ["RAN_2G.csv", "RAN_4G.csv"]
    
    analysis_results = {}
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            print(f"\n{'='*50}")
            print(f"ANALYZING: {file}")
            print(f"{'='*50}")
            
            try:
                # Read CSV
                df = pd.read_csv(filepath)
                
                # Basic info
                print(f"Shape: {df.shape}")
                print(f"Total columns: {len(df.columns)}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # Column names
                print(f"\nColumn names:")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                
                # Data types
                print(f"\nData types summary:")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    print(f"  {dtype}: {count} columns")
                
                # Sample data
                print(f"\nFirst few rows:")
                print(df.head(2).to_string())
                
                # Missing values
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    print(f"\nMissing values:")
                    missing_cols = missing[missing > 0]
                    for col, count in missing_cols.items():
                        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
                else:
                    print(f"\nNo missing values found.")
                
                # Common columns analysis
                common_cols = ['id', 'region', 'kabupaten', 'siteid', 'eniqhost', 'last_update']
                print(f"\nCommon identification columns:")
                for col in common_cols:
                    if col in df.columns:
                        unique_count = df[col].nunique()
                        print(f"  {col}: {unique_count} unique values")
                
                # Store for comparison
                analysis_results[file] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': dict(df.dtypes),
                    'sample': df.head(1).to_dict('records')[0] if len(df) > 0 else {}
                }
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    # Compare structures
    if len(analysis_results) == 2:
        print(f"\n{'='*50}")
        print("COMPARISON ANALYSIS")
        print(f"{'='*50}")
        
        files_list = list(analysis_results.keys())
        file1, file2 = files_list[0], files_list[1]
        
        cols1 = set(analysis_results[file1]['columns'])
        cols2 = set(analysis_results[file2]['columns'])
        
        common = cols1 & cols2
        unique_1 = cols1 - cols2
        unique_2 = cols2 - cols1
        
        print(f"Common columns ({len(common)}):")
        for col in sorted(common):
            print(f"  - {col}")
        
        print(f"\nUnique to {file1} ({len(unique_1)}):")
        for col in sorted(unique_1):
            print(f"  - {col}")
            
        print(f"\nUnique to {file2} ({len(unique_2)}):")
        for col in sorted(unique_2):
            print(f"  - {col}")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_csv_structure()
