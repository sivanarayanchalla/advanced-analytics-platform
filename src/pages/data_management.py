import streamlit as st
import pandas as pd
from src.modules.data_loader import DataLoader

def show_data_management(data_loader: DataLoader):
    """Data management page - focused on user uploads"""
    st.header("📊 Data Management")
    
    st.markdown("""
    ### Upload Your Data
    Supported formats: CSV, Excel, JSON
    """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload your dataset in CSV, Excel, or JSON format"
    )
    
    if uploaded_file is not None:
        try:
            # Load the file
            with st.spinner(f"Loading {uploaded_file.name}..."):
                df = data_loader.load_file(uploaded_file)
            
            # Store in session state
            st.session_state['current_data'] = df
            st.session_state['current_filename'] = uploaded_file.name
            
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            # Display data overview
            display_data_overview(df)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Try uploading a different file format or check your file structure.")
    
    else:
        # Show welcome message when no data is loaded
        show_welcome_message()

def display_data_overview(df: pd.DataFrame):
    """Display comprehensive data overview"""
    st.subheader("📋 Data Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    # Data preview tabs
    st.subheader("🔍 Data Preview")
    
    tab1, tab2, tab3, tab4 = st.tabs(["First 10 Rows", "Last 10 Rows", "Data Types", "Missing Values"])
    
    with tab1:
        st.dataframe(df.head(10), width='stretch')
    
    with tab2:
        st.dataframe(df.tail(10), width='stretch')
    
    with tab3:
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_info, width='stretch')
    
    with tab4:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.write("Columns with missing values:")
            for col, count in missing_data.items():
                st.write(f"- **{col}**: {count} missing values ({count/len(df)*100:.1f}%)")
        else:
            st.success("🎉 No missing values found!")
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        st.dataframe(df[numerical_cols].describe(), width='stretch')
    else:
        st.info("No numerical columns found for statistical analysis.")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Start EDA Analysis", use_container_width=True):
            st.session_state.page = "Advanced EDA"
            st.rerun()
    
    with col2:
        if st.button("🧠 Train Models", use_container_width=True):
            st.session_state.page = "Neural Networks"
            st.rerun()
    
    with col3:
        if st.button("🤖 AutoML", use_container_width=True):
            st.session_state.page = "AutoML"
            st.rerun()

def show_welcome_message():
    """Show welcome message when no data is loaded"""
    st.info("""
    ## 👋 Welcome to the Advanced Analytics Platform!
    
    **To get started:**
    
    1. **Upload your data** using the file uploader above
    2. **Explore your data** with interactive visualizations
    3. **Build machine learning models** using neural networks or AutoML
    4. **Generate insights** with advanced analytics
    
    ### 📁 Supported File Formats:
    - **CSV** (.csv) - Comma-separated values
    - **Excel** (.xlsx, .xls) - Microsoft Excel files
    - **JSON** (.json) - JavaScript Object Notation
    
    ### 🎯 What You Can Do:
    - **Data Profiling**: Automatic data quality reports
    - **Visual Analysis**: Interactive charts and graphs
    - **Machine Learning**: Neural networks, AutoML, and predictive modeling
    - **Time Series Analysis**: Forecasting and trend analysis
    - **Text Analytics**: NLP and sentiment analysis
    
    *Upload a file to begin your analysis!*
    """)
    
    # Sample data option for testing
    st.markdown("---")
    st.subheader("🎯 Quick Start with Sample Data")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("Load Iris Dataset", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target
            st.session_state['current_data'] = df
            st.session_state['current_filename'] = "iris_sample.csv"
            st.rerun()
    
    with sample_col2:
        if st.button("Load Titanic Dataset", use_container_width=True):
            try:
                df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
                st.session_state['current_data'] = df
                st.session_state['current_filename'] = "titanic_sample.csv"
                st.rerun()
            except:
                st.error("Could not load sample data. Please check your internet connection.")
    
    with sample_col3:
        if st.button("Load Sales Data", use_container_width=True):
            # Create sample sales data
            import random
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            data = {
                'date': dates,
                'sales': [random.randint(1000, 5000) for _ in range(100)],
                'customers': [random.randint(50, 200) for _ in range(100)],
                'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(100)]
            }
            df = pd.DataFrame(data)
            st.session_state['current_data'] = df
            st.session_state['current_filename'] = "sales_sample.csv"
            st.rerun()