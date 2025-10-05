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
        if st.button("Generate Comprehensive Data Profile"):
            with st.spinner("Generating detailed data profile report..."):
                try:
                    profile = analytics.generate_advanced_profile(df)
                    st.components.v1.html(profile.to_html(), height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating profile: {e}")
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("📈 Correlation Heatmap")
        
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_cols) < 2:
            st.warning("Need at least 2 numerical columns for correlation analysis.")
        else:
            try:
                corr_fig = analytics.generate_correlation_heatmap(df[numerical_cols])
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Show correlation insights
                st.subheader("🔍 Correlation Insights")
                corr_matrix = df[numerical_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.7:  # Strong correlation threshold
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_val
                            ))
                
                if high_corr_pairs:
                    st.write("**Strong Correlations Found:**")
                    for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                        st.write(f"- {col1} ↔ {col2}: {corr:.3f}")
                else:
                    st.info("No strong correlations (> 0.7) found between numerical variables.")
                    
            except Exception as e:
                st.error(f"Error generating correlation heatmap: {e}")
    
    elif analysis_type == "Distribution Analysis":
        st.subheader("📊 Distribution Analysis")
        
        # Column selection
        selected_col = st.selectbox("Select Column:", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if df[selected_col].dtype in ['int64', 'float64']:
                # Numerical column
                fig_hist = px.histogram(
                    df, 
                    x=selected_col,
                    title=f"Distribution of {selected_col}",
                    nbins=30
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                # Categorical column
                value_counts = df[selected_col].value_counts().head(10)
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 Values in {selected_col}",
                    labels={'x': selected_col, 'y': 'Count'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            if df[selected_col].dtype in ['int64', 'float64']:
                # Box plot for numerical
                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Statistics
                st.write("**Statistics:**")
                st.write(f"- Mean: {df[selected_col].mean():.2f}")
                st.write(f"- Median: {df[selected_col].median():.2f}")
                st.write(f"- Standard Deviation: {df[selected_col].std():.2f}")
                st.write(f"- Range: {df[selected_col].min():.2f} to {df[selected_col].max():.2f}")
            else:
                # Value counts table for categorical
                st.write("**Value Counts:**")
                value_counts = df[selected_col].value_counts()
                st.dataframe(value_counts, width='stretch')
    
    elif analysis_type == "PCA Analysis":
        st.subheader("🔮 Principal Component Analysis")
        
        try:
            pca_results = analytics.perform_pca_analysis(df)
            
            if 'error' in pca_results:
                st.error(pca_results['error'])
            else:
                # Explained variance
                exp_var = pca_results['explained_variance_ratio']
                st.write("**Explained Variance Ratio:**")
                for i, var in enumerate(exp_var):
                    st.write(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
                
                total_variance = sum(exp_var) * 100
                st.write(f"**Total Variance Explained:** {total_variance:.1f}%")
                
                # Create scree plot
                fig_scree = px.line(
                    x=range(1, len(exp_var) + 1),
                    y=exp_var,
                    title="Scree Plot - Variance Explained by Each Principal Component",
                    labels={'x': 'Principal Component', 'y': 'Variance Explained'}
                )
                st.plotly_chart(fig_scree, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error performing PCA analysis: {e}")