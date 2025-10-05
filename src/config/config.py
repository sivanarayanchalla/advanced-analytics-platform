import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
SRC_DIR = Path(__file__).parent.parent
sys.path.append(str(SRC_DIR))

class Config:
    """Configuration class for the application"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # MongoDB Configuration
    USE_MONGODB = os.getenv('USE_MONGODB', 'False').lower() == 'true'
    MONGODB_ATLAS_URI = os.getenv('MONGODB_ATLAS_URI', '')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'analytics_platform')
    
    # App Settings
    APP_NAME = "Advanced Analytics Platform"
    VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    APP_ENV = os.getenv('APP_ENV', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    
    # Streamlit Settings
    PAGE_TITLE = "AI Analytics Platform"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    def __init__(self):
        """Initialize configuration and create directories"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed", 
            self.DATA_DIR / "exports",
            self.MODELS_DIR / "saved_models",
            self.MODELS_DIR / "pipelines",
            self.LOGS_DIR / "app",
            self.LOGS_DIR / "training",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()