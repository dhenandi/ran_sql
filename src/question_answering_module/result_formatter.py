"""
Result Formatter
================

Formats query results for presentation in various formats.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime


class ResultFormatter:
    """
    Formats query results for different presentation formats.
    """
    
    def __init__(self):
        """Initialize the result formatter."""
        self.logger = logging.getLogger(__name__)
    
    def format_results(self, question: str, sql_query: str, raw_results: List[Dict],
                      entities: List[Dict], schema_info: Dict = None) -> Dict[str, Any]:
        """
        Format query results for presentation.
        
        Args:
            question: Original natural language question
            sql_query: Generated SQL query
            raw_results: Raw query results
            entities: Extracted entities
            schema_info: Database schema information
            
        Returns:
            Dict: Formatted results with multiple presentation options
        """
        try:
            formatted_result = {
                'summary': self._create_summary(question, raw_results, entities),
                'table': self._format_as_table(raw_results),
                'text': self._format_as_text(question, raw_results, entities),
                'charts': self._suggest_charts(raw_results, entities),
                'metadata': {
                    'original_question': question,
                    'sql_query': sql_query,
                    'result_count': len(raw_results),
                    'entities_found': len(entities),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Result formatting failed: {e}")
            return {
                'error': f"Formatting error: {str(e)}",
                'raw_results': raw_results
            }
    
    def _create_summary(self, question: str, results: List[Dict], 
                       entities: List[Dict]) -> str:
        """
        Create a text summary of the results.
        
        Args:
            question: Original question
            results: Query results
            entities: Extracted entities
            
        Returns:
            str: Summary text
        """
        if not results:
            return "No results found for your query."
        
        result_count = len(results)
        
        # Try to identify the main metric being queried
        main_metric = self._identify_main_metric(entities, results)
        
        if result_count == 1:
            if main_metric and main_metric in results[0]:
                value = results[0][main_metric]
                return f"The {main_metric} is {value}."
            else:
                return f"Found 1 result for your query."
        else:
            if main_metric and main_metric in results[0]:
                values = [row.get(main_metric) for row in results if main_metric in row]
                if values and all(isinstance(v, (int, float)) for v in values):
                    avg_value = sum(values) / len(values)
                    return f"Found {result_count} results. Average {main_metric}: {avg_value:.2f}"
            
            return f"Found {result_count} results for your query."
    
    def _identify_main_metric(self, entities: List[Dict], results: List[Dict]) -> Optional[str]:
        """
        Identify the main metric being queried.
        
        Args:
            entities: Extracted entities
            results: Query results
            
        Returns:
            Optional[str]: Main metric column name
        """
        # Look for KPI entities
        for entity in entities:
            if entity.get('label') == 'KPI_NAME':
                kpi_name = entity.get('text', '').lower()
                
                # Find matching column in results
                if results:
                    for col_name in results[0].keys():
                        if kpi_name in col_name.lower():
                            return col_name
        
        # Look for numeric columns (likely metrics)
        if results:
            for col_name, value in results[0].items():
                if isinstance(value, (int, float)) and 'id' not in col_name.lower():
                    return col_name
        
        return None
    
    def _format_as_table(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Format results as a structured table.
        
        Args:
            results: Query results
            
        Returns:
            Dict: Table representation
        """
        if not results:
            return {'columns': [], 'rows': []}
        
        # Get column names
        columns = list(results[0].keys())
        
        # Format rows
        rows = []
        for row in results:
            formatted_row = []
            for col in columns:
                value = row.get(col)
                formatted_value = self._format_cell_value(value)
                formatted_row.append(formatted_value)
            rows.append(formatted_row)
        
        return {
            'columns': columns,
            'rows': rows,
            'total_rows': len(rows)
        }
    
    def _format_cell_value(self, value: Any) -> str:
        """
        Format a single cell value for display.
        
        Args:
            value: Cell value
            
        Returns:
            str: Formatted value
        """
        if value is None:
            return "N/A"
        elif isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, (int, str)):
            return str(value)
        else:
            return str(value)
    
    def _format_as_text(self, question: str, results: List[Dict], 
                       entities: List[Dict]) -> str:
        """
        Format results as natural language text.
        
        Args:
            question: Original question
            results: Query results
            entities: Extracted entities
            
        Returns:
            str: Natural language description of results
        """
        if not results:
            return "No data was found that matches your query."
        
        # Start with summary
        text_parts = [self._create_summary(question, results, entities)]
        
        # Add details for small result sets
        if len(results) <= 5:
            text_parts.append("\nDetailed results:")
            
            for i, row in enumerate(results, 1):
                row_text = f"{i}. "
                row_parts = []
                
                for key, value in row.items():
                    if value is not None:
                        formatted_value = self._format_cell_value(value)
                        row_parts.append(f"{key}: {formatted_value}")
                
                row_text += ", ".join(row_parts)
                text_parts.append(row_text)
        
        elif len(results) > 5:
            text_parts.append(f"\nShowing first 5 of {len(results)} results:")
            
            for i, row in enumerate(results[:5], 1):
                row_text = f"{i}. "
                # Show only key columns for large result sets
                key_columns = self._identify_key_columns(results[0])
                row_parts = []
                
                for key in key_columns:
                    if key in row and row[key] is not None:
                        formatted_value = self._format_cell_value(row[key])
                        row_parts.append(f"{key}: {formatted_value}")
                
                row_text += ", ".join(row_parts)
                text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _identify_key_columns(self, sample_row: Dict) -> List[str]:
        """
        Identify the most important columns to display.
        
        Args:
            sample_row: Sample row to analyze
            
        Returns:
            List[str]: Key column names
        """
        key_columns = []
        
        # Prioritize certain column patterns
        priority_patterns = ['id', 'name', 'rsrp', 'rsrq', 'sinr', 'throughput', 'latency']
        
        for col_name in sample_row.keys():
            col_lower = col_name.lower()
            if any(pattern in col_lower for pattern in priority_patterns):
                key_columns.append(col_name)
        
        # If no priority columns found, use first few columns
        if not key_columns:
            key_columns = list(sample_row.keys())[:3]
        
        return key_columns[:5]  # Limit to 5 columns
    
    def _suggest_charts(self, results: List[Dict], entities: List[Dict]) -> List[Dict]:
        """
        Suggest appropriate chart types for the results.
        
        Args:
            results: Query results
            entities: Extracted entities
            
        Returns:
            List[Dict]: Chart suggestions
        """
        charts = []
        
        if not results or len(results) < 2:
            return charts
        
        # Analyze data structure
        numeric_columns = []
        categorical_columns = []
        temporal_columns = []
        
        for col_name, value in results[0].items():
            if isinstance(value, (int, float)):
                numeric_columns.append(col_name)
            elif isinstance(value, str):
                # Check if it might be temporal
                if any(word in col_name.lower() for word in ['date', 'time', 'hour']):
                    temporal_columns.append(col_name)
                else:
                    categorical_columns.append(col_name)
        
        # Suggest charts based on data structure
        
        # Time series chart
        if temporal_columns and numeric_columns:
            charts.append({
                'type': 'line',
                'title': 'Time Series',
                'x_axis': temporal_columns[0],
                'y_axis': numeric_columns[0],
                'description': f'{numeric_columns[0]} over time'
            })
        
        # Bar chart for categorical vs numeric
        if categorical_columns and numeric_columns and len(results) <= 20:
            charts.append({
                'type': 'bar',
                'title': 'Bar Chart',
                'x_axis': categorical_columns[0],
                'y_axis': numeric_columns[0],
                'description': f'{numeric_columns[0]} by {categorical_columns[0]}'
            })
        
        # Histogram for single numeric column
        if len(numeric_columns) >= 1 and len(results) > 10:
            charts.append({
                'type': 'histogram',
                'title': 'Distribution',
                'x_axis': numeric_columns[0],
                'description': f'Distribution of {numeric_columns[0]}'
            })
        
        # Scatter plot for two numeric columns
        if len(numeric_columns) >= 2:
            charts.append({
                'type': 'scatter',
                'title': 'Scatter Plot',
                'x_axis': numeric_columns[0],
                'y_axis': numeric_columns[1],
                'description': f'{numeric_columns[1]} vs {numeric_columns[0]}'
            })
        
        return charts
    
    def export_to_csv(self, results: List[Dict], filename: str) -> bool:
        """
        Export results to CSV file.
        
        Args:
            results: Query results
            filename: Output filename
            
        Returns:
            bool: True if successful
        """
        try:
            if not results:
                return False
            
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_json(self, formatted_results: Dict, filename: str) -> bool:
        """
        Export formatted results to JSON file.
        
        Args:
            formatted_results: Formatted results
            filename: Output filename
            
        Returns:
            bool: True if successful
        """
        try:
            import json
            
            with open(filename, 'w') as f:
                json.dump(formatted_results, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False
