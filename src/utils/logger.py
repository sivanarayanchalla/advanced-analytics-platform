import logging
import sys
from src.config.config import config

def setup_logger(name=__name__, log_level=logging.INFO):
    """Setup application logger - cloud compatible"""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level from config
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # In production, only use stream handler
    if config.APP_ENV == 'production':
        # Stream handler only for cloud
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        # Development: use both file and stream handlers
        from datetime import datetime
        log_file = config.LOGS_DIR / "app" / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except:
            pass  # Skip file handler if directory doesn't exist
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger