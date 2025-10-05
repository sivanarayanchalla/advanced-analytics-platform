import pandas as pd
import streamlit as st
from typing import Union, Dict, Any, List
import io
import requests
from sqlalchemy import create_engine
import logging
from datetime import datetime
from src.config.config import config

class DataLoader:
    """Data loader for user-uploaded files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, file) -> pd.DataFrame:
        """Load CSV file"""
        try:
            df = pd.read_csv(file)
            self.logger.info(f"Loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            st.error(f"Error loading CSV file: {e}")
            raise
    
    def load_excel(self, file) -> pd.DataFrame:
        """Load Excel file"""
        try:
            df = pd.read_excel(file)
            self.logger.info(f"Loaded Excel with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel: {e}")
            st.error(f"Error loading Excel file: {e}")
            raise
    
    def load_json(self, file) -> pd.DataFrame:
        """Load JSON file"""
        try:
            df = pd.read_json(file)
            self.logger.info(f"Loaded JSON with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            st.error(f"Error loading JSON file: {e}")
            raise
    
    def detect_file_type(self, filename: str) -> str:
        """Detect file type from extension"""
        if filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename.endswith('.json'):
            return 'json'
        else:
            return 'unknown'
    
    def load_file(self, file) -> pd.DataFrame:
        """Auto-detect file type and load data"""
        filename = file.name if hasattr(file, 'name') else 'uploaded_file'
        file_type = self.detect_file_type(filename)
        
        if file_type == 'csv':
            return self.load_csv(file)
        elif file_type == 'excel':
            return self.load_excel(file)
        elif file_type == 'json':
            return self.load_json(file)
        else:
            raise ValueError(f"Unsupported file type: {filename}")