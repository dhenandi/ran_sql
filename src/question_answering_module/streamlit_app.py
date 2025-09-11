"""
Streamlit Web Application
=========================

Web interface for the RAN SQL Question Answering System.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import sys
from pathlib import Path
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config
from .qa_pipeline import QuestionAnsweringPipeline


class StreamlitApp:
    """
    Streamlit web application for the QA system.
    """
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.config = get_config()
        self.qa_pipeline = None
        
        # Initialize session state
        if 'qa_pipeline' not in st.session_state:
            st.session_state.qa_pipeline = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="RAN SQL QA System",
            page_icon="📡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_header()
        self._render_sidebar()
        self._render_main_interface()
    
    def _render_header(self):
        """Render the application header."""
        st.title("📡 RAN SQL Question Answering System")
        st.markdown("""
        Ask questions about your Radio Access Network performance data in natural language.
        The system will translate your questions into SQL queries and provide insights.
        """)
    
    def _render_sidebar(self):
        """Render the sidebar with configuration and status."""
        with st.sidebar:
            st.header("🔧 System Configuration")
            
            # System status
            self._render_system_status()
            
            # Configuration options
            st.subheader("Settings")
            
            # Model selection
            ner_model_type = st.selectbox(
                "NER Model Type",
                ["spacy", "transformers"],
                index=0 if self.config.NER_MODEL_TYPE == "spacy" else 1
            )
            
            sql_model_type = st.selectbox(
                "SQL Model Type",
                ["template", "transformers"],
                index=0
            )
            
            # Initialize pipeline button
            if st.button("🚀 Initialize Pipeline"):
                self._initialize_pipeline(ner_model_type, sql_model_type)
            
            # Query history
            self._render_query_history()
    
    def _render_system_status(self):
        """Render system status indicators."""
        st.subheader("System Status")
        
        # Database status
        db_exists = self.config.DATABASE_PATH.exists()
        st.write(f"📊 Database: {'✅ Connected' if db_exists else '❌ Not Found'}")
        
        # Pipeline status
        pipeline_ready = st.session_state.qa_pipeline is not None
        st.write(f"🔧 Pipeline: {'✅ Ready' if pipeline_ready else '❌ Not Initialized'}")
        
        if pipeline_ready:
            status = st.session_state.qa_pipeline.get_pipeline_status()
            st.write(f"🤖 NER Model: {'✅' if status['ner_pipeline'] else '❌'}")
            st.write(f"💬 SQL Generator: {'✅' if status['query_generator'] else '❌'}")
            st.write(f"🗃️ Schema Loaded: {'✅' if status['schema_info_loaded'] else '❌'}")
    
    def _initialize_pipeline(self, ner_model_type: str, sql_model_type: str):
        """Initialize the QA pipeline."""
        try:
            with st.spinner("Initializing pipeline..."):
                pipeline_config = {
                    'ner': {
                        'model_type': ner_model_type,
                        'base_model': self.config.NER_BASE_MODEL
                    },
                    'sql_model_type': sql_model_type,
                    'database_path': str(self.config.DATABASE_PATH),
                    'ner_model_path': str(self.config.NER_MODELS_DIR / "latest"),
                    'sql_model_path': str(self.config.SQL_MODELS_DIR / "templates")
                }
                
                st.session_state.qa_pipeline = QuestionAnsweringPipeline(pipeline_config)
                st.success("✅ Pipeline initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize pipeline: {str(e)}")
    
    def _render_main_interface(self):
        """Render the main query interface."""
        # Query input
        st.header("💬 Ask a Question")
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What is the average RSRP for all cells?",
                "Show me the maximum throughput yesterday",
                "Which cells have RSRQ below -10 dB?",
                "Count the number of unique sites",
                "What is the hourly average latency trend?",
                "Compare SINR values between different sectors"
            ]
            
            for question in sample_questions:
                if st.button(question, key=f"sample_{hash(question)}"):
                    self._process_question(question)
        
        # Main query input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the average RSRP for cell ABC123?"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Ask Question", type="primary"):
                if question.strip():
                    self._process_question(question)
                else:
                    st.warning("Please enter a question.")
        
        with col2:
            if st.button("🧪 Analyze Question"):
                if question.strip():
                    self._analyze_question(question)
        
        # Display results
        if st.session_state.current_results:
            self._render_results(st.session_state.current_results)
    
    def _process_question(self, question: str):
        """Process a natural language question."""
        if not st.session_state.qa_pipeline:
            st.error("❌ Pipeline not initialized. Please initialize it first in the sidebar.")
            return
        
        try:
            with st.spinner("🤔 Processing your question..."):
                start_time = time.time()
                
                # Process the question
                result = st.session_state.qa_pipeline.answer_question(question)
                
                # Store result and add to history
                st.session_state.current_results = result
                st.session_state.query_history.append({
                    'question': question,
                    'timestamp': time.time(),
                    'success': result.success,
                    'processing_time': result.processing_time
                })
                
                # Show processing time
                st.info(f"⏱️ Processed in {result.processing_time:.2f} seconds")
                
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
    
    def _analyze_question(self, question: str):
        """Analyze a question without executing it."""
        if not st.session_state.qa_pipeline:
            st.error("❌ Pipeline not initialized.")
            return
        
        try:
            # Extract entities
            entities = st.session_state.qa_pipeline._extract_entities(question, [])
            
            st.subheader("🔍 Question Analysis")
            
            # Show extracted entities
            if entities:
                st.write("**Extracted Entities:**")
                entity_df = pd.DataFrame(entities)
                st.dataframe(entity_df[['text', 'label', 'confidence']])
            else:
                st.write("No entities extracted.")
            
            # Show query type prediction
            # This would need to be implemented in the pipeline
            st.write("**Predicted Query Type:** Analysis not implemented yet")
            
        except Exception as e:
            st.error(f"❌ Error analyzing question: {str(e)}")
    
    def _render_results(self, result):
        """Render query results."""
        st.header("📊 Results")
        
        if not result.success:
            st.error("❌ Query failed")
            
            if result.errors:
                st.subheader("Errors:")
                for error in result.errors:
                    st.error(error)
            
            return
        
        # Success indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Confidence", f"{result.confidence:.1%}")
        
        with col2:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
        
        with col3:
            st.metric("Entities Found", len(result.entities))
        
        with col4:
            result_count = len(result.query_results) if result.query_results else 0
            st.metric("Results", result_count)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "📊 Data", "📈 Charts", "🔧 Technical"])
        
        with tab1:
            self._render_summary_tab(result)
        
        with tab2:
            self._render_data_tab(result)
        
        with tab3:
            self._render_charts_tab(result)
        
        with tab4:
            self._render_technical_tab(result)
        
        # Warnings
        if result.warnings:
            with st.expander("⚠️ Warnings"):
                for warning in result.warnings:
                    st.warning(warning)
    
    def _render_summary_tab(self, result):
        """Render the summary tab."""
        if result.formatted_results and 'text' in result.formatted_results:
            st.write(result.formatted_results['text'])
        else:
            st.write("No summary available.")
        
        # Show extracted entities
        if result.entities:
            st.subheader("🏷️ Extracted Entities")
            entity_df = pd.DataFrame(result.entities)
            st.dataframe(entity_df)
    
    def _render_data_tab(self, result):
        """Render the data tab."""
        if not result.query_results:
            st.write("No data to display.")
            return
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(result.query_results)
        
        # Show data
        st.subheader(f"📊 Query Results ({len(df)} rows)")
        st.dataframe(df)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="query_results.json",
                mime="application/json"
            )
    
    def _render_charts_tab(self, result):
        """Render the charts tab."""
        if not result.query_results:
            st.write("No data available for charts.")
            return
        
        df = pd.DataFrame(result.query_results)
        
        # Get chart suggestions
        chart_suggestions = []
        if result.formatted_results and 'charts' in result.formatted_results:
            chart_suggestions = result.formatted_results['charts']
        
        if not chart_suggestions:
            st.write("No chart suggestions available.")
            return
        
        # Render suggested charts
        for chart in chart_suggestions:
            st.subheader(f"📈 {chart['title']}")
            st.write(chart['description'])
            
            try:
                if chart['type'] == 'line':
                    fig = px.line(df, x=chart['x_axis'], y=chart['y_axis'])
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart['type'] == 'bar':
                    fig = px.bar(df, x=chart['x_axis'], y=chart['y_axis'])
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart['type'] == 'histogram':
                    fig = px.histogram(df, x=chart['x_axis'])
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart['type'] == 'scatter':
                    fig = px.scatter(df, x=chart['x_axis'], y=chart['y_axis'])
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    def _render_technical_tab(self, result):
        """Render the technical details tab."""
        st.subheader("🔍 Generated SQL Query")
        if result.sql_query:
            st.code(result.sql_query, language="sql")
        else:
            st.write("No SQL query generated.")
        
        st.subheader("🏷️ Entity Extraction Details")
        if result.entities:
            for entity in result.entities:
                with st.expander(f"{entity.get('label', 'Unknown')}: {entity.get('text', '')}"):
                    st.json(entity)
        
        st.subheader("📊 Processing Statistics")
        stats = {
            "Processing Time": f"{result.processing_time:.3f} seconds",
            "Confidence Score": f"{result.confidence:.2%}",
            "Entities Extracted": len(result.entities),
            "Result Rows": len(result.query_results) if result.query_results else 0
        }
        
        for key, value in stats.items():
            st.write(f"**{key}:** {value}")
    
    def _render_query_history(self):
        """Render query history in sidebar."""
        st.subheader("📜 Query History")
        
        if not st.session_state.query_history:
            st.write("No queries yet.")
            return
        
        # Show recent queries
        recent_queries = st.session_state.query_history[-5:]  # Last 5 queries
        
        for i, query in enumerate(reversed(recent_queries)):
            status_icon = "✅" if query['success'] else "❌"
            truncated_question = query['question'][:30] + "..." if len(query['question']) > 30 else query['question']
            
            if st.button(f"{status_icon} {truncated_question}", key=f"history_{i}"):
                self._process_question(query['question'])


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
