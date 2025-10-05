import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    # python-dotenv not available, use environment variables directly
    DOTENV_AVAILABLE = False
    print("Info: python-dotenv not installed. Using environment variables directly.")

# Add src to Python path
SRC_DIR = Path(__file__).parent.parent
sys.path.append(str(SRC_DIR))

class Config:
    """Configuration class for the application"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # App Settings
    APP_NAME = "Advanced Analytics Platform"
    VERSION = "1.0.0"
    DEBUG = False  # Hardcoded for cloud
    APP_ENV = "production"  # Hardcoded for cloud
    LOG_LEVEL = "INFO"
    
    # Feature Flags
    USE_MONGODB = False  # Hardcoded - MongoDB disabled
    
    # API Keys
    OPENAI_API_KEY = ""  # Empty for now
    
    # Streamlit Settings
    PAGE_TITLE = "AI Analytics Platform"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

# Global config instance
config = Config()