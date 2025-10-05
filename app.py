import streamlit as st
import sys
from pathlib import Path

# Add src to Python path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from src.config.config import config
from src.modules.data_loader import DataLoader
from src.modules.advanced_analytics import AdvancedAnalytics
from src.utils.logger import setup_logger

def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #8facc9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def show_dashboard():
    """Main dashboard view"""
    st.markdown('<div class="main-header">🧠 Advanced Analytics Platform</div>', 
                unsafe_allow_html=True)
    
    # Check if data is loaded
    data_loaded = 'current_data' in st.session_state and st.session_state['current_data'] is not None
    
    if data_loaded:
        df = st.session_state['current_data']
        filename = st.session_state.get('current_filename', 'your data')
        
        st.success(f"📁 Currently loaded: {filename} ({df.shape[0]} rows × {df.shape[1]} columns)")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Upload Data</h3>
            <p>Start by uploading your CSV, Excel, or JSON files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Explore Data</h3>
            <p>Analyze patterns and relationships in your data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Build Models</h3>
            <p>Create predictive models with neural networks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Get Insights</h3>
            <p>Generate reports and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    if not data_loaded:
        st.info("📁 **No data loaded yet.** Upload your data to begin analysis!")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📊 Upload Data", use_container_width=True):
            st.session_state.page = "Data Management"
            st.rerun()
    
    with quick_col2:
        if data_loaded:
            if st.button("🔍 Explore Data", use_container_width=True):
                st.session_state.page = "Advanced EDA"
                st.rerun()
        else:
            st.button("🔍 Explore Data", use_container_width=True, disabled=True)
    
    with quick_col3:
        if data_loaded:
            if st.button("🧠 Train Models", use_container_width=True):
                st.session_state.page = "Neural Networks"
                st.rerun()
        else:
            st.button("🧠 Train Models", use_container_width=True, disabled=True)

def main():
    """Main application entry point"""
    setup_page()
    
    # Initialize components
    logger = setup_logger()
    data_loader = DataLoader()
    analytics = AdvancedAnalytics()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_filename' not in st.session_state:
        st.session_state.current_filename = None
    
    # Sidebar navigation
    st.sidebar.title("🧠 Navigation")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio(
        "Go to:",
        [
            "🏠 Dashboard",
            "📊 Data Management", 
            "🔍 Advanced EDA",
            "🧠 Neural Networks",
            "📈 Time Series",
            "📝 Text Analytics",
            "🤖 AutoML",
            "⚙️ Settings"
        ]
    )
    
    # Update session state
    st.session_state.page = page
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Management":
        from src.pages.data_management import show_data_management
        show_data_management(data_loader)
    elif page == "🔍 Advanced EDA":
        from src.pages.advanced_eda import show_advanced_eda
        show_advanced_eda(analytics)
    elif page == "🧠 Neural Networks":
        from src.pages.neural_networks import show_neural_networks
        show_neural_networks(analytics)
    elif page == "📈 Time Series":
        from src.pages.time_series import show_time_series
        show_time_series(analytics)
    elif page == "📝 Text Analytics":
        from src.pages.text_analytics import show_text_analytics
        show_text_analytics(analytics)
    elif page == "🤖 AutoML":
        from src.pages.automl import show_automl
        show_automl(analytics)
    elif page == "⚙️ Settings":
        from src.pages.settings import show_settings
        show_settings()

if __name__ == "__main__":
    main()