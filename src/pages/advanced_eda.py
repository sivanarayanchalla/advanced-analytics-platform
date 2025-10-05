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
        if st.button("Generate Data Profile Report"):
            with st.spinner("Generating data profile report..."):
                try:
                    profile = analytics.generate_advanced_profile(df)
                    if hasattr(profile, 'show_html'):
                        # Sweetviz report
                        profile_file = "sweetviz_report.html"
                        profile.show_html(profile_file)
                        with open(profile_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800, scrolling=True)
                    else:
                        # Basic profile fallback
                        st.subheader("📊 Basic Data Profile")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Dataset Overview**")
                            st.write(f"**Shape:** {profile['shape'][0]} rows, {profile['shape'][1]} columns")
                            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                            
                        with col2:
                            st.write("**Data Types**")
                            for col, dtype in profile['data_types'].items():
                                st.write(f"- {col}: {dtype}")
                        
                        # Missing values
                        st.subheader("🔍 Missing Values Analysis")
                        missing_data = {k: v for k, v in profile['missing_values'].items() if v > 0}
                        if missing_data:
                            st.write("Columns with missing values:")
                            for col, count in missing_data.items():
                                percentage = (count / len(df)) * 100
                                st.write(f"- **{col}**: {count} missing ({percentage:.1f}%)")
                        else:
                            st.success("🎉 No missing values found!")
                        
                        # Basic statistics
                        st.subheader("📈 Basic Statistics")
                        st.dataframe(pd.DataFrame(profile['basic_stats']), width='stretch')
                        
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
                    st.write("**Strong Correlations Found (> 0.7):**")
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
                
                # Outliers detection
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                if len(outliers) > 0:
                    st.warning(f"⚠️ Found {len(outliers)} potential outliers")
                else:
                    st.success("✅ No outliers detected using IQR method")
                    
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
                
                # Show feature loadings for first two components
                if len(exp_var) >= 2:
                    st.subheader("📊 Feature Loadings (First Two Components)")
                    
                    # Create loadings dataframe
                    loadings_df = pd.DataFrame({
                        'Feature': pca_results['feature_names'],
                        'PC1': pca_results['components'][0],
                        'PC2': pca_results['components'][1]
                    })
                    
                    # Show top features for each component
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Features - PC1:**")
                        top_pc1 = loadings_df.nlargest(5, 'PC1')[['Feature', 'PC1']]
                        st.dataframe(top_pc1, width='stretch')
                    
                    with col2:
                        st.write("**Top Features - PC2:**")
                        top_pc2 = loadings_df.nlargest(5, 'PC2')[['Feature', 'PC2']]
                        st.dataframe(top_pc2, width='stretch')
                    
                    # Create PCA scatter plot if we have at least 2 components
                    if len(exp_var) >= 2:
                        pca_df = pd.DataFrame({
                            'PC1': [x[0] for x in pca_results['principal_components']],
                            'PC2': [x[1] for x in pca_results['principal_components']]
                        })
                        
                        # Add index for coloring
                        pca_df['index'] = range(len(pca_df))
                        
                        fig_pca = px.scatter(
                            pca_df, 
                            x='PC1', 
                            y='PC2',
                            title="PCA - First Two Principal Components",
                            hover_data=['index']
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error performing PCA analysis: {e}")
    
    # Quick insights section
    st.markdown("---")
    st.subheader("💡 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Data quality insights
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            st.error(f"❌ {total_missing} missing values")
        else:
            st.success("✅ No missing values")
    
    with col2:
        # Data type insights
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.info(f"📊 {numeric_cols} numeric, {categorical_cols} categorical columns")
    
    with col3:
        # Size insights
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.info(f"💾 {memory_mb:.1f} MB memory usage")

def show_data_quality_report(df):
    """Show comprehensive data quality report"""
    st.subheader("📋 Data Quality Report")
    
    # Create quality metrics
    quality_data = []
    
    for col in df.columns:
        col_data = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null Percentage': (df[col].isnull().sum() / len(df)) * 100,
            'Unique Values': df[col].nunique(),
            'Duplicate Rows': df.duplicated(subset=[col]).sum()
        }
        quality_data.append(col_data)
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, width='stretch')
    
    # Overall quality score
    total_null_percentage = quality_df['Null Percentage'].mean()
    if total_null_percentage == 0:
        st.success("🎉 Excellent data quality - No missing values!")
    elif total_null_percentage < 5:
        st.warning(f"⚠️ Good data quality - {total_null_percentage:.1f}% missing values on average")
    else:
        st.error(f"❌ Poor data quality - {total_null_percentage:.1f}% missing values on average")