import streamlit as st
import pandas as pd
import plotly.express as px
from src.modules.advanced_analytics import AdvancedAnalytics

def show_advanced_eda(analytics: AdvancedAnalytics):
    """Advanced Exploratory Data Analysis page"""
    st.header("🔍 Advanced Exploratory Data Analysis")
    
    # Check if data is loaded
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("📁 No data loaded yet!")
        st.info("Please go to **Data Management** page to upload your data first.")
        
        if st.button("📊 Go to Data Management"):
            st.session_state.page = "Data Management"
            st.rerun()
        return
    
    df = st.session_state['current_data']
    filename = st.session_state.get('current_filename', 'your data')
    
    st.success(f"📊 Analyzing: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")
    
    # EDA Options
    st.subheader("📊 Analysis Options")
    
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Data Profile", "Correlation Analysis", "Distribution Analysis", "PCA Analysis"]
    )
    
    if analysis_type == "Data Profile":
        if st.button("Generate Data Profile"):
            with st.spinner("Generating data profile..."):
                try:
                    profile = analytics.generate_advanced_profile(df)
                    
                    if 'error' in profile:
                        st.error(profile['error'])
                    else:
                        display_basic_profile(profile, df)
                        
                except Exception as e:
                    st.error(f"Error generating profile: {e}")
    
    elif analysis_type == "Correlation Analysis":
        display_correlation_analysis(analytics, df)
    
    elif analysis_type == "Distribution Analysis":
        display_distribution_analysis(df)
    
    elif analysis_type == "PCA Analysis":
        display_pca_analysis(analytics, df)

def display_basic_profile(profile, df):
    """Display basic data profile"""
    st.subheader("📊 Data Profile Overview")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", profile['overview']['shape'][0])
    with col2:
        st.metric("Columns", profile['overview']['shape'][1])
    with col3:
        st.metric("Memory Usage", f"{profile['overview']['memory_usage']:.2f} MB")
    with col4:
        st.metric("Duplicates", profile['overview']['duplicates'])
    
    # Column details
    st.subheader("📋 Column Details")
    for col, info in profile['columns'].items():
        with st.expander(f"**{col}** ({info['dtype']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Non-null:** {info['non_null_count']}")
                st.write(f"**Null values:** {info['null_count']} ({info['null_percentage']:.1f}%)")
                st.write(f"**Unique values:** {info['unique_count']}")
            with col2:
                if 'mean' in info:
                    st.write(f"**Mean:** {info['mean']:.2f}")
                    st.write(f"**Std:** {info['std']:.2f}")
                    st.write(f"**Range:** {info['min']:.2f} to {info['max']:.2f}")
    
    # Basic statistics
    if profile['basic_stats']:
        st.subheader("📈 Basic Statistics")
        st.dataframe(pd.DataFrame(profile['basic_stats']), width='stretch')

def display_correlation_analysis(analytics, df):
    """Display correlation analysis"""
    st.subheader("📈 Correlation Analysis")
    
    try:
        corr_fig = analytics.generate_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
    except Exception as e:
        st.error(str(e))

def display_distribution_analysis(df):
    """Display distribution analysis"""
    st.subheader("📊 Distribution Analysis")
    
    selected_col = st.selectbox("Select Column:", df.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            value_counts = df[selected_col].value_counts().head(10)
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Top 10 Values in {selected_col}")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig_box, use_container_width=True)

def display_pca_analysis(analytics, df):
    """Display PCA analysis"""
    st.subheader("🔮 Principal Component Analysis")
    
    n_components = st.slider("Number of Components:", 2, 5, 2)
    
    if st.button("Run PCA Analysis"):
        with st.spinner("Performing PCA..."):
            try:
                pca_results = analytics.perform_pca_analysis(df, n_components)
                
                if 'error' in pca_results:
                    st.error(pca_results['error'])
                else:
                    # Display explained variance
                    exp_var = pca_results['explained_variance_ratio']
                    st.write("**Explained Variance Ratio:**")
                    for i, var in enumerate(exp_var):
                        st.write(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
                    
                    total_variance = sum(exp_var) * 100
                    st.write(f"**Total Variance Explained:** {total_variance:.1f}%")
                    
            except Exception as e:
                st.error(f"Error performing PCA: {e}")