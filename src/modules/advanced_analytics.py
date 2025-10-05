import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Handles advanced analytics and machine learning operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for machine learning training"""
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Encode categorical features
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                if X_processed[col].nunique() <= 10:
                    # One-hot encode for low cardinality
                    encoded = pd.get_dummies(X_processed[col], prefix=col)
                    X_processed = pd.concat([X_processed, encoded], axis=1)
                    X_processed.drop(col, axis=1, inplace=True)
                else:
                    # Label encode for high cardinality
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Convert to numeric and handle missing values
        X_processed = X_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(y_processed):
            le = LabelEncoder()
            y_processed = pd.Series(le.fit_transform(y_processed.astype(str)))
        
        y_processed = pd.to_numeric(y_processed, errors='coerce').fillna(0)
        
        return X_processed, y_processed
    
    def generate_advanced_profile(self, df: pd.DataFrame):
        """Generate basic data profile using pandas only"""
        try:
            profile_data = self._generate_basic_profile(df)
            return profile_data
        except Exception as e:
            st.error(f"Error generating profile: {e}")
            return {"error": str(e)}
    
    def _generate_basic_profile(self, df: pd.DataFrame):
        """Generate comprehensive basic data profile"""
        profile = {
            'overview': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
                'duplicates': df.duplicated().sum(),
                'total_missing': df.isnull().sum().sum()
            },
            'columns': {},
            'correlation': df.select_dtypes(include=['number']).corr() if len(df.select_dtypes(include=['number']).columns) > 1 else None,
            'basic_stats': df.describe().to_dict()
        }
        
        # Column-level information
        for col in df.columns:
            profile['columns'][col] = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'sample_values': df[col].head(5).tolist() if df[col].dtype == 'object' else None
            }
            
            # Add basic statistics for numerical columns
            if pd.api.types.is_numeric_dtype(df[col]):
                profile['columns'][col].update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
        
        return profile
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series, 
                           hidden_layers: list = [100, 50], 
                           problem_type: str = "regression",
                           epochs: int = 100) -> Tuple[Any, StandardScaler, list]:
        """Train neural network using scikit-learn MLP"""
        try:
            # Preprocess data
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_processed, test_size=0.2, random_state=42
            )
            
            # Create and train model
            if problem_type == "regression":
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=epochs,
                    random_state=42,
                    learning_rate_init=0.001
                )
            else:
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=epochs,
                    random_state=42,
                    learning_rate_init=0.001
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get training loss
            losses = model.loss_curve_ if hasattr(model, 'loss_curve_') else []
            
            self.logger.info(f"Training completed. Final loss: {losses[-1] if losses else 'N/A'}")
            return model, scaler, losses
        
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
            raise
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Generate interactive correlation heatmap"""
        try:
            numerical_df = df.select_dtypes(include=['number'])
            if len(numerical_df.columns) < 2:
                raise ValueError("Need at least 2 numerical columns for correlation analysis")
                
            corr_matrix = numerical_df.corr()
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")
            raise
    
    def perform_pca_analysis(self, df: pd.DataFrame, n_components: int = 2) -> Dict:
        """Perform PCA analysis on numerical data"""
        try:
            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])
            
            if numerical_df.empty:
                return {"error": "No numerical columns found for PCA"}
            if len(numerical_df.columns) < 2:
                return {"error": "Need at least 2 numerical columns for PCA"}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_df)
            
            # Perform PCA
            pca = PCA(n_components=min(n_components, len(numerical_df.columns)))
            principal_components = pca.fit_transform(scaled_data)
            
            # Create results dictionary
            results = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'principal_components': principal_components.tolist(),
                'feature_names': numerical_df.columns.tolist()
            }
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error performing PCA analysis: {e}")
            return {"error": str(e)}