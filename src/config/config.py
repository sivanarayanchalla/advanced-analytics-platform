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
    
    # For cloud deployment, don't create directories
    if os.getenv('APP_ENV') != 'production':
        DATA_DIR = BASE_DIR / "data"
        MODELS_DIR = BASE_DIR / "models"
        LOGS_DIR = BASE_DIR / "logs"
    else:
        # In production, use temporary paths or avoid file operations
        DATA_DIR = Path("/tmp/data")  # Temporary directory
        MODELS_DIR = Path("/tmp/models")
        LOGS_DIR = Path("/tmp/logs")
    
    # MongoDB Configuration
    USE_MONGODB = os.getenv('USE_MONGODB', 'False').lower() == 'true'
    MONGODB_ATLAS_URI = os.getenv('MONGODB_ATLAS_URI', '')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'analytics_platform')
    
    # App Settings
    APP_NAME = "Advanced Analytics Platform"
    VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    APP_ENV = os.getenv('APP_ENV', 'production')  # Default to production for cloud
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    
    # Streamlit Settings
    PAGE_TITLE = "AI Analytics Platform"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    def __init__(self):
        """Initialize configuration - only create directories in development"""
        if self.APP_ENV != 'production':
            self.create_directories()
    
    def create_directories(self):
        """Create necessary directories only in development"""
        try:
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
        except Exception as e:
            # Silently fail directory creation in cloud environment
            pass

# Global config instance
config = Config()