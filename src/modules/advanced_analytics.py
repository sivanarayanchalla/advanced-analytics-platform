import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Handles advanced analytics and neural network operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for neural network training"""
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
    
    def generate_advanced_profile(self, df: pd.DataFrame) -> ProfileReport:
        """Generate advanced data profile"""
        try:
            profile = ProfileReport(
                df, 
                title="Advanced Data Profile",
                explorative=True,
                minimal=False,
                progress_bar=False
            )
            self.logger.info("Generated advanced data profile")
            return profile
        except Exception as e:
            self.logger.error(f"Error generating profile: {e}")
            raise
    
    def train_tabular_nn(self, X: pd.DataFrame, y: pd.Series, 
                        hidden_layers: list = [128, 64, 32], 
                        epochs: int = 100) -> Tuple[nn.Module, StandardScaler, list]:
        """Train tabular neural network with automatic preprocessing"""
        try:
            # Preprocess data
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y_processed.values).unsqueeze(1).to(self.device)
            
            # Define model
            class TabularNN(nn.Module):
                def __init__(self, input_size, hidden_layers):
                    super().__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_layers:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.3)
                        ])
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 1))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            model = TabularNN(X_processed.shape[1], hidden_layers).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if epoch % 20 == 0:
                    self.logger.info(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            
            self.logger.info(f"Training completed. Final loss: {losses[-1]:.4f}")
            return model, scaler, losses
        
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
            raise
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Generate interactive correlation heatmap"""
        try:
            corr_matrix = df.corr()
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
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])
            
            if numerical_df.empty:
                return {"error": "No numerical columns found for PCA"}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_df)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
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