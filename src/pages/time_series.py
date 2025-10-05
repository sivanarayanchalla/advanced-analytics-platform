import streamlit as st
import pandas as pd
import plotly.express as px

def show_time_series(analytics):
    """Time Series Analysis page"""
    st.header("📈 Time Series Analysis")
    
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("📁 No data loaded yet!")
        st.info("Please go to **Data Management** page to upload your data first.")
        return
    
    df = st.session_state['current_data']
    
    st.info("⏰ Time Series Analysis - Upload data with date/time columns for forecasting")
    
    # Check for date columns
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
    
    if date_columns:
        st.success(f"✅ Found potential date columns: {', '.join(date_columns)}")
        
        # Let user select date column
        date_column = st.selectbox("Select Date Column:", date_columns)
        
        # Let user select value column
        value_columns = [col for col in df.columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])]
        if value_columns:
            value_column = st.selectbox("Select Value Column:", value_columns)
            
            # Convert to datetime if needed
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                
                # Plot time series
                st.subheader("📊 Time Series Plot")
                fig = px.line(
                    df.sort_values(date_column),
                    x=date_column,
                    y=value_column,
                    title=f"{value_column} over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Basic time series stats
                st.subheader("📈 Time Series Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Time Period", f"{df[date_column].min().date()} to {df[date_column].max().date()}")
                with col2:
                    st.metric("Data Points", len(df))
                with col3:
                    st.metric("Average Value", f"{df[value_column].mean():.2f}")
                    
            except Exception as e:
                st.error(f"Error processing date column: {e}")
        else:
            st.warning("No numerical value columns found for time series analysis.")
    else:
        st.warning("No date columns found. Time series analysis requires a date/time column.")
        
    st.markdown("---")
    st.subheader("🎯 Time Series Features Coming Soon")
    st.info("""
    **Planned Time Series Features:**
    - **Forecasting**: ARIMA, Prophet, LSTM models
    - **Seasonality Analysis**: Detect patterns and cycles
    - **Trend Analysis**: Identify long-term trends
    - **Anomaly Detection**: Find outliers in time series data
    - **Multiple Series**: Compare multiple time series
    """)