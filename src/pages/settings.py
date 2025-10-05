import streamlit as st
from src.config.config import config

def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Application Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**App Version:** {config.VERSION}")
        st.write(f"**Environment:** {config.APP_ENV}")
        st.write(f"**Debug Mode:** {config.DEBUG}")
    
    with col2:
        st.write(f"**Log Level:** {config.LOG_LEVEL}")
        st.write(f"**MongoDB Enabled:** {config.USE_MONGODB}")
    
    st.subheader("📊 System Information")
    
    # Display system info
    try:
        import platform
        import psutil
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Python Version:** {platform.python_version()}")
            st.write(f"**Operating System:** {platform.system()} {platform.release()}")
        
        with col2:
            memory = psutil.virtual_memory()
            st.write(f"**Memory Usage:** {memory.percent}%")
            st.write(f"**CPU Cores:** {psutil.cpu_count()}")
            
    except ImportError:
        st.info("Install `psutil` for detailed system information")
    
    st.subheader("🛠️ Configuration")
    
    st.info("""
    To modify settings, edit the `.env` file in your project directory.
    Available configuration options:
    
    - `USE_MONGODB`: Enable/disable MongoDB (True/False)
    - `DEBUG`: Enable debug mode (True/False) 
    - `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `HUGGINGFACE_API_KEY`: Your HuggingFace API key
    """)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear Session Data"):
            if 'current_data' in st.session_state:
                del st.session_state['current_data']
            if 'current_filename' in st.session_state:
                del st.session_state['current_filename']
            st.success("Session data cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 View Logs"):
            st.info("Log files are stored in the `logs/` directory")