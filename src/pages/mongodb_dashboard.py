import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from src.config.config import config
from src.modules.mongodb_manager import get_mongodb_manager

def show_mongodb_dashboard():
    """MongoDB Atlas monitoring and management dashboard"""
    st.header("🗄️ MongoDB Atlas Dashboard")
    
    mongodb_manager = get_mongodb_manager()
    
    if not mongodb_manager:
        st.error("MongoDB is not configured. Please check your connection settings.")
        return
    
    # Dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Collection statistics
        raw_stats = mongodb_manager.get_collection_stats(config.MONGODB_COLLECTION_RAW)
        processed_stats = mongodb_manager.get_collection_stats(config.MONGODB_COLLECTION_PROCESSED)
        model_stats = mongodb_manager.get_collection_stats(config.MONGODB_COLLECTION_MODELS)
        results_stats = mongodb_manager.get_collection_stats(config.MONGODB_COLLECTION_RESULTS)
        
        with col1:
            st.metric(
                "Raw Data Documents", 
                raw_stats['total_documents'],
                help="Total documents in raw data collection"
            )
        
        with col2:
            st.metric(
                "Processed Data Documents",
                processed_stats['total_documents'],
                help="Total documents in processed data collection"
            )
        
        with col3:
            st.metric(
                "Saved Models",
                model_stats['total_documents'],
                help="Total trained model metadata"
            )
        
        with col4:
            st.metric(
                "Analysis Results",
                results_stats['total_documents'],
                help="Total analysis results saved"
            )
        
        # Collection overview
        st.subheader("📊 Collection Overview")
        
        collections_data = {
            'Collection': ['Raw Data', 'Processed Data', 'Models', 'Results'],
            'Documents': [
                raw_stats['total_documents'],
                processed_stats['total_documents'], 
                model_stats['total_documents'],
                results_stats['total_documents']
            ]
        }
        
        fig = px.bar(
            collections_data, 
            x='Collection', 
            y='Documents',
            title="Documents per Collection",
            color='Collection'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model history
        st.subheader("🧠 Model Training History")
        model_history = mongodb_manager.get_model_history(limit=10)
        
        if model_history:
            model_df = pd.DataFrame([
                {
                    'Model Name': model['model_name'],
                    'Type': model['model_type'],
                    'Created': model['created_at'].strftime('%Y-%m-%d %H:%M'),
                    'Performance': model['performance'].get('accuracy', model['performance'].get('r2', 'N/A'))
                }
                for model in model_history
            ])
            
            st.dataframe(model_df, use_container_width=True)
        else:
            st.info("No model history found. Train some models to see history here.")
        
        # Database operations
        st.subheader("⚙️ Database Operations")
        
        op_col1, op_col2 = st.columns(2)
        
        with op_col1:
            if st.button("🔄 Refresh Statistics"):
                st.rerun()
            
            if st.button("🗂️ Create Indexes"):
                with st.spinner("Creating indexes..."):
                    try:
                        mongodb_manager.create_index(
                            config.MONGODB_COLLECTION_RAW, 
                            ['created_at']
                        )
                        mongodb_manager.create_index(
                            config.MONGODB_COLLECTION_PROCESSED,
                            ['created_at']  
                        )
                        st.success("Indexes created successfully!")
                    except Exception as e:
                        st.error(f"Error creating indexes: {e}")
        
        with op_col2:
            if st.button("💾 Backup Data"):
                with st.spinner("Creating backup..."):
                    try:
                        backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        mongodb_manager.backup_data(
                            config.MONGODB_COLLECTION_RAW,
                            backup_name
                        )
                        st.success(f"Backup created: {backup_name}")
                    except Exception as e:
                        st.error(f"Error creating backup: {e}")
        
        # Sample data preview
        st.subheader("🔍 Sample Data Preview")
        
        preview_tab1, preview_tab2, preview_tab3 = st.tabs([
            "Raw Data", "Processed Data", "Model Metadata"
        ])
        
        with preview_tab1:
            raw_sample = mongodb_manager.find_dataframe(
                config.MONGODB_COLLECTION_RAW, limit=5
            )
            if not raw_sample.empty:
                st.dataframe(raw_sample, use_container_width=True)
            else:
                st.info("No raw data available")
        
        with preview_tab2:
            processed_sample = mongodb_manager.find_dataframe(
                config.MONGODB_COLLECTION_PROCESSED, limit=5
            )
            if not processed_sample.empty:
                st.dataframe(processed_sample, use_container_width=True)
            else:
                st.info("No processed data available")
        
        with preview_tab3:
            model_sample = mongodb_manager.get_model_history(limit=3)
            if model_sample:
                for model in model_sample:
                    with st.expander(f"Model: {model['model_name']}"):
                        st.json(model, expanded=False)
            else:
                st.info("No model metadata available")
    
    except Exception as e:
        st.error(f"Error loading MongoDB dashboard: {e}")