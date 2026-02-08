#!/usr/bin/env python3
"""
Generate SQL Training Data
===========================

Generates diverse query-SQL pairs from RAN database schema for training
the SQL generation model.
"""

import sys
import json
import sqlite3
import random
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config

def get_database_schema(db_path):
    """Extract schema information from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        
        columns = []
        for col in columns_info:
            columns.append({
                'name': col[1],
                'type': col[2],
                'notnull': col[3],
                'pk': col[5]
            })
        
        schema.append({
            'table': table_name,
            'columns': columns,
            'column_names': [c['name'] for c in columns]
        })
    
    conn.close()
    return schema


def identify_kpi_columns(columns):
    """Identify KPI columns from column names."""
    kpi_patterns = [
        'rsrp', 'rsrq', 'sinr', 'throughput', 'latency',
        'bler', 'cqi', 'drop', 'call', 'success',
        'availability', 'utilization', 'traffic', 'erlang',
        'count', 'rate', 'ratio', 'pct', 'avg', 'max', 'min'
    ]
    return [col for col in columns if any(p in col.lower() for p in kpi_patterns)]


def identify_location_columns(columns):
    """Identify location columns."""
    location_patterns = ['region', 'kabupaten', 'kota', 'city', 'area', 'zone']
    return [col for col in columns if any(p in col.lower() for p in location_patterns)]


def identify_identifier_columns(columns):
    """Identify identifier columns."""
    id_patterns = ['siteid', 'site_id', 'cellid', 'cell_id', 'enb', 'id']
    return [col for col in columns if any(p in col.lower() for p in id_patterns)]


def identify_temporal_columns(columns):
    """Identify time-related columns."""
    time_patterns = ['date', 'time', 'timestamp', 'last_update', 'created']
    return [col for col in columns if any(p in col.lower() for p in time_patterns)]


def generate_aggregation_queries(table_info, count=50):
    """Generate aggregation queries (AVG, SUM, COUNT, MAX, MIN)."""
    queries = []
    table_name = table_info['table']
    columns = table_info['column_names']
    
    kpi_cols = identify_kpi_columns(columns)
    location_cols = identify_location_columns(columns)
    id_cols = identify_identifier_columns(columns)
    
    aggregations = [
        ('average', 'AVG', 'KPI_NAME'),
        ('maximum', 'MAX', 'KPI_NAME'),
        ('minimum', 'MIN', 'KPI_NAME'),
        ('total', 'SUM', 'KPI_NAME'),
        ('count of', 'COUNT', 'SITE_ID')
    ]
    
    for _ in range(count):
        agg_nl, agg_sql, entity_type = random.choice(aggregations)
        
        if agg_sql in ['AVG', 'MAX', 'MIN', 'SUM'] and kpi_cols:
            kpi_col = random.choice(kpi_cols)
            
            # With location filter
            if location_cols and random.random() > 0.5:
                loc_col = random.choice(location_cols)
                location_val = random.choice(['Jakarta', 'Bandung', 'Surabaya', 'Semarang'])
                
                nl_query = f"What is the {agg_nl} {kpi_col} in {location_val}?"
                sql_query = f"SELECT {agg_sql}({kpi_col}) FROM {table_name} WHERE {loc_col} = '{location_val}';"
                entities = [
                    {'text': kpi_col, 'label': 'KPI_NAME'},
                    {'text': location_val, 'label': 'LOCATION'}
                ]
            else:
                nl_query = f"What is the {agg_nl} {kpi_col}?"
                sql_query = f"SELECT {agg_sql}({kpi_col}) FROM {table_name};"
                entities = [
                    {'text': kpi_col, 'label': 'KPI_NAME'}
                ]
            
            queries.append({
                'natural_language': nl_query,
                'sql': sql_query,
                'entities': entities,
                'query_type': f'aggregation_{agg_sql.lower()}',
                'table': table_name
            })
        
        elif agg_sql == 'COUNT' and id_cols:
            id_col = random.choice(id_cols)
            
            if location_cols and random.random() > 0.5:
                loc_col = random.choice(location_cols)
                location_val = random.choice(['Jakarta', 'Bandung', 'Surabaya'])
                
                nl_query = f"How many {id_col} are in {location_val}?"
                sql_query = f"SELECT COUNT(DISTINCT {id_col}) FROM {table_name} WHERE {loc_col} = '{location_val}';"
                entities = [
                    {'text': id_col, 'label': 'SITE_ID'},
                    {'text': location_val, 'label': 'LOCATION'}
                ]
            else:
                nl_query = f"Count the number of {id_col}"
                sql_query = f"SELECT COUNT(DISTINCT {id_col}) FROM {table_name};"
                entities = [
                    {'text': id_col, 'label': 'SITE_ID'}
                ]
            
            queries.append({
                'natural_language': nl_query,
                'sql': sql_query,
                'entities': entities,
                'query_type': 'aggregation_count',
                'table': table_name
            })
    
    return queries


def generate_selection_queries(table_info, count=50):
    """Generate simple selection queries."""
    queries = []
    table_name = table_info['table']
    columns = table_info['column_names']
    
    kpi_cols = identify_kpi_columns(columns)
    location_cols = identify_location_columns(columns)
    id_cols = identify_identifier_columns(columns)
    
    for _ in range(count):
        if kpi_cols and id_cols:
            kpi_col = random.choice(kpi_cols)
            id_col = random.choice(id_cols)
            id_val = f"SITE_{random.randint(1, 100):03d}"
            
            nl_query = f"Show me {kpi_col} for {id_val}"
            sql_query = f"SELECT {kpi_col} FROM {table_name} WHERE {id_col} = '{id_val}';"
            entities = [
                {'text': kpi_col, 'label': 'KPI_NAME'},
                {'text': id_val, 'label': 'SITE_ID'}
            ]
            
            queries.append({
                'natural_language': nl_query,
                'sql': sql_query,
                'entities': entities,
                'query_type': 'selection',
                'table': table_name
            })
    
    return queries


def generate_filtering_queries(table_info, count=50):
    """Generate filtering queries with conditions."""
    queries = []
    table_name = table_info['table']
    columns = table_info['column_names']
    
    kpi_cols = identify_kpi_columns(columns)
    location_cols = identify_location_columns(columns)
    
    operators = [
        ('above', '>', 'NUMERIC_VALUE'),
        ('below', '<', 'NUMERIC_VALUE'),
        ('greater than', '>', 'NUMERIC_VALUE'),
        ('less than', '<', 'NUMERIC_VALUE')
    ]
    
    for _ in range(count):
        if kpi_cols:
            kpi_col = random.choice(kpi_cols)
            op_nl, op_sql, entity_type = random.choice(operators)
            threshold = random.choice(['-100', '-90', '-80', '50', '10', '20'])
            
            # With location
            if location_cols and random.random() > 0.5:
                loc_col = random.choice(location_cols)
                location_val = random.choice(['Jakarta', 'Bandung', 'Surabaya'])
                
                nl_query = f"List sites with {kpi_col} {op_nl} {threshold} in {location_val}"
                sql_query = f"SELECT * FROM {table_name} WHERE {kpi_col} {op_sql} {threshold} AND {loc_col} = '{location_val}';"
                entities = [
                    {'text': kpi_col, 'label': 'KPI_NAME'},
                    {'text': threshold, 'label': 'NUMERIC_VALUE'},
                    {'text': location_val, 'label': 'LOCATION'}
                ]
            else:
                nl_query = f"Show all records where {kpi_col} is {op_nl} {threshold}"
                sql_query = f"SELECT * FROM {table_name} WHERE {kpi_col} {op_sql} {threshold};"
                entities = [
                    {'text': kpi_col, 'label': 'KPI_NAME'},
                    {'text': threshold, 'label': 'NUMERIC_VALUE'}
                ]
            
            queries.append({
                'natural_language': nl_query,
                'sql': sql_query,
                'entities': entities,
                'query_type': 'filtering',
                'table': table_name
            })
    
    return queries


def generate_grouping_queries(table_info, count=30):
    """Generate queries with GROUP BY."""
    queries = []
    table_name = table_info['table']
    columns = table_info['column_names']
    
    kpi_cols = identify_kpi_columns(columns)
    location_cols = identify_location_columns(columns)
    
    if not (kpi_cols and location_cols):
        return queries
    
    for _ in range(count):
        kpi_col = random.choice(kpi_cols)
        loc_col = random.choice(location_cols)
        agg = random.choice(['AVG', 'SUM', 'COUNT', 'MAX', 'MIN'])
        agg_nl = {
            'AVG': 'average',
            'SUM': 'total',
            'COUNT': 'count of',
            'MAX': 'maximum',
            'MIN': 'minimum'
        }[agg]
        
        nl_query = f"Show {agg_nl} {kpi_col} by {loc_col}"
        sql_query = f"SELECT {loc_col}, {agg}({kpi_col}) FROM {table_name} GROUP BY {loc_col};"
        entities = [
            {'text': kpi_col, 'label': 'KPI_NAME'},
            {'text': loc_col, 'label': 'REGION'}
        ]
        
        queries.append({
            'natural_language': nl_query,
            'sql': sql_query,
            'entities': entities,
            'query_type': 'grouping',
            'table': table_name
        })
    
    return queries


def generate_ordering_queries(table_info, count=30):
    """Generate queries with ORDER BY."""
    queries = []
    table_name = table_info['table']
    columns = table_info['column_names']
    
    kpi_cols = identify_kpi_columns(columns)
    id_cols = identify_identifier_columns(columns)
    
    if not (kpi_cols and id_cols):
        return queries
    
    for _ in range(count):
        kpi_col = random.choice(kpi_cols)
        id_col = random.choice(id_cols)
        order_nl, order_sql = random.choice([
            ('highest', 'DESC'),
            ('lowest', 'ASC'),
            ('best', 'DESC'),
            ('worst', 'ASC')
        ])
        limit = random.choice([5, 10, 20])
        
        nl_query = f"Show top {limit} {id_col} with {order_nl} {kpi_col}"
        sql_query = f"SELECT {id_col}, {kpi_col} FROM {table_name} ORDER BY {kpi_col} {order_sql} LIMIT {limit};"
        entities = [
            {'text': id_col, 'label': 'SITE_ID'},
            {'text': kpi_col, 'label': 'KPI_NAME'},
            {'text': str(limit), 'label': 'NUMERIC_VALUE'}
        ]
        
        queries.append({
            'natural_language': nl_query,
            'sql': sql_query,
            'entities': entities,
            'query_type': 'ordering',
            'table': table_name
        })
    
    return queries


def main():
    """Main function to generate SQL training data."""
    print("=" * 80)
    print("SQL TRAINING DATA GENERATION")
    print("=" * 80)
    
    config = get_config()
    db_path = config.DATABASE_PATH
    output_path = config.PROCESSED_DATA_DIR / "sql_training_data.json"
    
    print(f"\n📂 Database: {db_path}")
    print(f"📂 Output: {output_path}")
    
    # Get schema
    print("\n🔍 Analyzing database schema...")
    schema = get_database_schema(db_path)
    print(f"   ✓ Found {len(schema)} tables")
    
    # Generate queries for each table
    all_queries = []
    
    for table_info in schema:
        table_name = table_info['table']
        print(f"\n📊 Generating queries for table: {table_name}")
        
        # Generate different query types
        agg_queries = generate_aggregation_queries(table_info, count=100)
        sel_queries = generate_selection_queries(table_info, count=80)
        filt_queries = generate_filtering_queries(table_info, count=80)
        group_queries = generate_grouping_queries(table_info, count=50)
        order_queries = generate_ordering_queries(table_info, count=50)
        
        table_queries = agg_queries + sel_queries + filt_queries + group_queries + order_queries
        
        print(f"   - Aggregation: {len(agg_queries)}")
        print(f"   - Selection: {len(sel_queries)}")
        print(f"   - Filtering: {len(filt_queries)}")
        print(f"   - Grouping: {len(group_queries)}")
        print(f"   - Ordering: {len(order_queries)}")
        print(f"   ✓ Total: {len(table_queries)} queries")
        
        all_queries.extend(table_queries)
    
    # Statistics
    print(f"\n📊 Training Data Statistics:")
    print(f"   Total queries: {len(all_queries)}")
    
    query_type_counts = defaultdict(int)
    for q in all_queries:
        query_type_counts[q['query_type']] += 1
    
    print(f"\n   Query type distribution:")
    for qtype, count in sorted(query_type_counts.items()):
        pct = (count / len(all_queries)) * 100
        print(f"      - {qtype}: {count} ({pct:.1f}%)")
    
    # Save training data
    print(f"\n💾 Saving training data...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_queries, f, indent=2)
    
    print(f"   ✓ Saved {len(all_queries)} training samples to: {output_path}")
    
    # File size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ File size: {file_size:.2f} MB")
    
    print("\n✅ SQL TRAINING DATA GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
