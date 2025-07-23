"""
Streamlit Frontend for Document Similarity Analysis
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io

# --- Fix Import Paths ---
# Get the absolute path to the backend directory
current_dir = Path(__file__).parent
backend_path = current_dir.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import backend modules with error handling
try:
    from document_processor import DocumentProcessor
    from similarity_engine import SimilarityEngine
    import config
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.error(f"Python path: {sys.path}")
    st.error(f"Current dir: {current_dir}")
    st.error(f"Backend path: {backend_path}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Similarity Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# --- Initialize Components ---
@st.cache_resource
def initialize_components():
    """Initialize document processor and similarity engine."""
    try:
        doc_processor = DocumentProcessor()
        similarity_engine = SimilarityEngine()
        return doc_processor, similarity_engine
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        st.stop()

doc_processor, similarity_engine = initialize_components()

# --- Main Application Functions ---
def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("📄 Document Similarity Analysis")
    st.markdown("**Find similarities between documents using advanced NLP models**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Model info
        with st.expander("ℹ️ Model Information", expanded=False):
            st.write(f"**Model:** {config.MODEL_NAME}")
            st.write(f"**Min Similarity:** {config.MIN_SIMILARITY_THRESHOLD}")
            st.write(f"**Max File Size:** {config.MAX_FILE_SIZE // (1024*1024)} MB")
        
        # Clear all data
        if st.button("🗑️ Clear All Data", type="secondary"):
            clear_all_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Documents", 
        "🔍 Analyze Similarity", 
        "📊 Results & Visualization",
        "🔎 Query Search"
    ])
    
    with tab1:
        upload_documents_tab()
    
    with tab2:
        analyze_similarity_tab()
    
    with tab3:
        results_visualization_tab()
    
    with tab4:
        query_search_tab()

def upload_documents_tab():
    """Document upload interface."""
    st.header("📤 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to analyze",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        help="Upload multiple documents to find similarities between them"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} files selected")
        
        # Display file preview
        with st.expander("📝 File Preview", expanded=False):
            preview_file = st.selectbox("Select file to preview", [f.name for f in uploaded_files])
            selected_file = next(f for f in uploaded_files if f.name == preview_file)
            
            try:
                if preview_file.endswith('.txt') or preview_file.endswith('.md'):
                    content = selected_file.getvalue().decode('utf-8')
                    st.text_area("File Content", content, height=200)
                else:
                    st.warning("Preview only available for text files (.txt, .md)")
            except Exception as e:
                st.error(f"Couldn't preview file: {e}")
        
        # Save uploaded files
        if st.button("💾 Save Files for Analysis", type="primary"):
            with st.spinner("Saving files..."):
                saved_files = save_uploaded_files(uploaded_files)
                if saved_files:
                    st.session_state.uploaded_files = saved_files
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Successfully saved {len(saved_files)} files!")
                    
                    # Show file details
                    df = pd.DataFrame(saved_files)
                    st.dataframe(df, use_container_width=True)
    
    # Show currently loaded files
    if st.session_state.documents_loaded:
        st.subheader("📋 Currently Loaded Files")
        
        # Load and display current documents
        documents = doc_processor.load_documents(config.INPUT_DIR)
        if documents:
            doc_data = []
            for doc in documents:
                doc_data.append({
                    'Name': doc['name'],
                    'Size (KB)': round(doc['size'] / 1024, 2),
                    'Extension': doc['extension'],
                    'Characters': len(doc['content'])
                })
            
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
            
            # Add option to view document content
            doc_to_view = st.selectbox("View document content", [""] + [d['name'] for d in documents])
            if doc_to_view:
                selected_doc = next(d for d in documents if d['name'] == doc_to_view)
                st.text_area("Document Content", selected_doc['content'], height=300)
        else:
            st.info("No documents currently loaded.")

def analyze_similarity_tab():
    """Similarity analysis interface."""
    st.header("🔍 Analyze Document Similarity")
    
    # Check if we have documents
    documents = doc_processor.load_documents(config.INPUT_DIR)
    
    if len(documents) < 2:
        st.warning("⚠️ Please upload at least 2 documents to perform similarity analysis.")
        return
    
    st.success(f"✅ Found {len(documents)} documents ready for analysis")
    
    # Analysis settings
    with st.expander("⚙️ Analysis Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            min_threshold = st.slider(
                "Minimum Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.MIN_SIMILARITY_THRESHOLD,
                step=0.05,
                help="Only show pairs with similarity above this threshold"
            )
        
        with col2:
            save_cache = st.checkbox(
                "Save embeddings cache",
                value=config.SAVE_EMBEDDINGS,
                help="Cache embeddings for faster future analysis"
            )
    
    # Analyze button
    if st.button("🚀 Start Analysis", type="primary"):
        with st.spinner("🧠 Computing document embeddings and similarities..."):
            # Update config temporarily
            original_threshold = config.MIN_SIMILARITY_THRESHOLD
            original_save = config.SAVE_EMBEDDINGS
            
            config.MIN_SIMILARITY_THRESHOLD = min_threshold
            config.SAVE_EMBEDDINGS = save_cache
            
            try:
                # Perform analysis
                results = similarity_engine.find_similarities(documents)
                st.session_state.analysis_results = results
                
                if 'error' in results:
                    st.error(f"❌ Analysis failed: {results['error']}")
                else:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Show quick stats
                    metadata = results.get('metadata', {})
                    cols = st.columns(4)
                    stats = [
                        ("Documents", metadata.get('total_documents', 0)),
                        ("Pairs Analyzed", metadata.get('total_pairs_analyzed', 0)),
                        ("Similar Pairs", metadata.get('pairs_above_threshold', 0)),
                        ("Embedding Dim", metadata.get('embedding_dimension', 0))
                    ]
                    
                    for col, (label, value) in zip(cols, stats):
                        col.metric(label, value)
            
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
            
            finally:
                # Restore original config
                config.MIN_SIMILARITY_THRESHOLD = original_threshold
                config.SAVE_EMBEDDINGS = original_save

def results_visualization_tab():
    """Results display and visualization."""
    st.header("📊 Analysis Results & Visualization")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 No analysis results yet. Please run the analysis first.")
        return
    
    results = st.session_state.analysis_results
    
    if 'error' in results:
        st.error(f"❌ {results['error']}")
        return
    
    # Metadata display
    metadata = results.get('metadata', {})
    similarities = results.get('similarities', [])
    
    st.subheader("📈 Analysis Overview")
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", metadata.get('total_documents', 0))
        st.metric("Similar Pairs Found", len(similarities))
    
    with col2:
        if similarities:
            avg_similarity = sum(s['score'] for s in similarities) / len(similarities)
            max_similarity = max(s['score'] for s in similarities)
            st.metric("Average Similarity", f"{avg_similarity:.3f}")
            st.metric("Highest Similarity", f"{max_similarity:.3f}")
    
    with col3:
        st.metric("Model Used", metadata.get('model_used', 'N/A'))
        timestamp = metadata.get('timestamp', '')
        if timestamp:
            st.metric("Analysis Time", timestamp.split('T')[1][:8])
    
    if not similarities:
        st.warning("No similar document pairs found above the threshold.")
        return
    
    # Similarity pairs table
    st.subheader("🔗 Similar Document Pairs")
    
    # Create DataFrame for display
    pairs_data = []
    for sim in similarities:
        pairs_data.append({
            'Document 1': sim['doc1'],
            'Document 2': sim['doc2'],
            'Similarity Score': round(sim['score'], 4),
            'Score %': f"{sim['score']*100:.1f}%"
        })
    
    df_pairs = pd.DataFrame(pairs_data)
    st.dataframe(
        df_pairs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Similarity Score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=1,
                format="%.3f"
            )
        }
    )
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Similarity score distribution
        fig_hist = px.histogram(
            x=[s['score'] for s in similarities],
            nbins=20,
            title="Distribution of Similarity Scores",
            labels={'x': 'Similarity Score', 'y': 'Count'}
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with viz_col2:
        # Top 10 similar pairs
        top_pairs = sorted(similarities, key=lambda x: x['score'], reverse=True)[:10]
        pair_labels = [f"{s['doc1'][:15]}... ↔ {s['doc2'][:15]}..." for s in top_pairs]
        
        fig_bar = px.bar(
            x=[s['score'] for s in top_pairs],
            y=pair_labels,
            orientation='h',
            title="Top 10 Most Similar Pairs",
            labels={'x': 'Similarity Score', 'y': 'Document Pairs'},
            color=[s['score'] for s in top_pairs],
            color_continuous_scale='Bluered'
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Similarity network graph
    st.subheader("🕸️ Document Similarity Network")
    
    try:
        # Create nodes and edges for the graph
        nodes = set()
        edges = []
        
        for sim in similarities:
            nodes.add(sim['doc1'])
            nodes.add(sim['doc2'])
            edges.append((sim['doc1'], sim['doc2'], sim['score']))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[edge[0], edge[1]],
                y=[1, 1],
                line=dict(width=edge[2]*10, color='gray'),
                hoverinfo='none',
                mode='lines'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=list(nodes),
            y=[1]*len(nodes),
            mode='markers+text',
            marker=dict(size=20, color='blue'),
            text=list(nodes),
            textposition="top center",
            hoverinfo='text'
        ))
        
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't generate network graph: {e}")
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download JSON
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Download as JSON",
            data=json_str,
            file_name=f"similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Download CSV
        csv_buffer = io.StringIO()
        df_pairs.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"similarity_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def query_search_tab():
    """Search similar documents using query text."""
    st.header("🔎 Find Similar Documents to Query")
    
    documents = doc_processor.load_documents(config.INPUT_DIR)
    
    if not documents:
        st.warning("⚠️ Please upload documents first.")
        return
    
    st.info(f"📚 {len(documents)} documents available for search")
    
    # Query input
    query_text = st.text_area(
        "Enter your query text:",
        height=150,
        placeholder="Type or paste text to find similar documents...",
        help="The system will find documents most similar to this text"
    )
    
    # Search settings
    with st.expander("🔎 Search Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider(
                "Number of results",
                min_value=1,
                max_value=10,
                value=5,
                help="How many similar documents to return"
            )
        
        with col2:
            min_score = st.slider(
                "Minimum similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Only show documents with similarity above this value"
            )
    
    # Search button
    if st.button("🔍 Search Similar Documents", type="primary") and query_text.strip():
        with st.spinner("🧠 Finding similar documents..."):
            try:
                results = similarity_engine.find_most_similar_to_query(
                    query_text.strip(), documents, top_k
                )
                
                # Filter by minimum score
                results = [r for r in results if r['score'] >= min_score]
                
                if results:
                    st.success(f"✅ Found {len(results)} similar documents")
                    
                    # Display results in tabs
                    tabs = st.tabs([f"#{i+1}" for i in range(len(results))])
                    
                    for i, (result, tab) in enumerate(zip(results, tabs)):
                        with tab:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.subheader(result['document'])
                                st.write(f"**Similarity Score:** {result['score']:.4f} ({result['score']*100:.1f}%)")
                                st.write(f"**File Size:** {result['size']} bytes")
                                st.write(f"**File Type:** {result['extension']}")
                                
                                with st.expander("View Content"):
                                    st.text(result['content'][:1000] + ("..." if len(result['content']) > 1000 else ""))
                            
                            with col2:
                                # Gauge chart for similarity
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=result['score'],
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Similarity"},
                                    gauge={
                                        'axis': {'range': [0, 1]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 0.3], 'color': "red"},
                                            {'range': [0.3, 0.7], 'color': "orange"},
                                            {'range': [0.7, 1], 'color': "green"}
                                        ],
                                    }
                                ))
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No documents found with similarity above {min_score}")
                    
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                st.exception(e)

# --- Helper Functions ---
def save_uploaded_files(uploaded_files):
    """Save uploaded files to the input directory."""
    saved_files = []
    
    # Ensure input directory exists
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        try:
            # Save file
            file_path = config.INPUT_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate file
            if doc_processor.validate_file(file_path):
                file_info = doc_processor.get_file_info(file_path)
                saved_files.append(file_info)
            else:
                file_path.unlink()  # Remove invalid file
                st.error(f"❌ Invalid file: {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error saving {uploaded_file.name}: {str(e)}")
    
    return saved_files

def clear_all_data():
    """Clear all uploaded files and results."""
    try:
        # Clear uploaded files
        if config.INPUT_DIR.exists():
            for file_path in config.INPUT_DIR.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        
        # Clear cache and results
        similarity_engine.clear_cache()
        if config.RESULTS_FILE.exists():
            config.RESULTS_FILE.unlink()
        
        # Clear session state
        st.session_state.analysis_results = None
        st.session_state.uploaded_files = []
        st.session_state.documents_loaded = False
        
        st.success("🗑️ All data cleared successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing data: {str(e)}")

# --- Run the Application ---
if __name__ == "__main__":
    main()