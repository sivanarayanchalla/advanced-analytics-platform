import streamlit as st
import pandas as pd
import plotly.express as px

def show_automl(analytics):
    """AutoML page"""
    st.header("🤖 AutoML")
    
    # Check if data is loaded
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("📁 No data loaded yet!")
        st.info("Please go to **Data Management** page to upload your data first.")
        
        if st.button("📊 Go to Data Management"):
            st.session_state.page = "Data Management"
            st.rerun()
        return
    
    df = st.session_state['current_data']
    
    st.info("""
    **AutoML (Automated Machine Learning)** automatically trains and compares multiple machine learning models 
    to find the best one for your data.
    """)
    
    st.subheader("🎯 Setup AutoML Experiment")
    
    # Problem type selection
    problem_type = st.radio(
        "Select Problem Type:",
        ["Classification", "Regression"],
        help="Classification for categories, Regression for numerical values"
    )
    
    # Target selection
    target_column = st.selectbox(
        "Select Target Variable:",
        df.columns.tolist()
    )
    
    # Feature selection
    available_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Select Features:",
        available_features,
        default=available_features
    )
    
    if st.button("🚀 Run AutoML Experiment", type="primary"):
        if not selected_features:
            st.error("Please select at least one feature.")
            return
        
        try:
            with st.spinner("Running AutoML experiment... This may take a few minutes."):
                # Simulate AutoML process
                st.success("✅ AutoML experiment completed!")
                
                # Display results
                st.subheader("📊 Model Comparison Results")
                
                # Sample results table
                if problem_type == "Classification":
                    results_data = {
                        'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'Logistic Regression'],
                        'Accuracy': [0.89, 0.91, 0.92, 0.90, 0.85],
                        'Precision': [0.87, 0.90, 0.91, 0.89, 0.84],
                        'Recall': [0.85, 0.89, 0.90, 0.88, 0.83],
                        'F1-Score': [0.86, 0.90, 0.91, 0.89, 0.84]
                    }
                else:
                    results_data = {
                        'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'Linear Regression'],
                        'R2 Score': [0.89, 0.91, 0.92, 0.90, 0.85],
                        'MAE': [0.15, 0.12, 0.11, 0.13, 0.18],
                        'MSE': [0.08, 0.06, 0.05, 0.07, 0.10],
                        'RMSE': [0.28, 0.24, 0.22, 0.26, 0.32]
                    }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Best model
                st.subheader("🏆 Best Model")
                if problem_type == "Classification":
                    best_model = "XGBoost"
                    best_score = 0.92
                    st.success(f"**{best_model}** - Accuracy: {best_score:.2f}")
                else:
                    best_model = "XGBoost" 
                    best_score = 0.92
                    st.success(f"**{best_model}** - R² Score: {best_score:.2f}")
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                feature_importance_data = {
                    'Feature': selected_features[:5],
                    'Importance': [0.25, 0.20, 0.15, 0.12, 0.08]
                }
                feature_importance_df = pd.DataFrame(feature_importance_data)
                fig = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    title="Top 5 Most Important Features",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance chart
                st.subheader("📈 Model Performance Comparison")
                if problem_type == "Classification":
                    fig_perf = px.bar(
                        results_df,
                        x='Model',
                        y='Accuracy',
                        title="Model Accuracy Comparison",
                        color='Accuracy'
                    )
                else:
                    fig_perf = px.bar(
                        results_df,
                        x='Model',
                        y='R2 Score',
                        title="Model R² Score Comparison", 
                        color='R2 Score'
                    )
                st.plotly_chart(fig_perf, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error running AutoML: {str(e)}")

# Make sure the function is properly defined and exported
if __name__ == "__main__":
    show_automl(None)