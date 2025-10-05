import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def show_neural_networks(analytics):
    """Neural Networks page"""
    st.header("🧠 Neural Networks")
    
    # Check if data is loaded
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("📁 No data loaded yet!")
        st.info("Please go to **Data Management** page to upload your data first.")
        
        if st.button("📊 Go to Data Management"):
            st.session_state.page = "Data Management"
            st.rerun()
        return
    
    df = st.session_state['current_data']
    
    st.subheader("📊 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data preprocessing section
    st.subheader("🔧 Data Preprocessing")
    
    # Select target variable
    st.subheader("🎯 Select Target Variable")
    target_column = st.selectbox(
        "Choose the column you want to predict:",
        df.columns.tolist()
    )
    
    # Show target variable info
    target_dtype = df[target_column].dtype
    if pd.api.types.is_numeric_dtype(target_dtype):
        st.success(f"✅ Target '{target_column}' is numeric - suitable for regression")
        problem_type = "regression"
    else:
        st.warning(f"⚠️ Target '{target_column}' is categorical - will be encoded for classification")
        problem_type = "classification"
    
    # Feature selection with data type information
    st.subheader("🔧 Feature Selection")
    available_features = [col for col in df.columns if col != target_column]
    
    # Show feature information
    feature_info = []
    for col in available_features:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        if pd.api.types.is_numeric_dtype(dtype):
            feature_type = "Numerical"
        else:
            feature_type = f"Categorical ({unique_count} unique values)"
        feature_info.append(f"{col} - {feature_type}")
    
    selected_features = st.multiselect(
        "Select features for training:",
        available_features,
        default=available_features[:min(5, len(available_features))],
        help="Neural networks work best with numerical data. Categorical features will be automatically encoded."
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for training.")
        return
    
    # Show preprocessing preview
    st.subheader("🔍 Preprocessing Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Selected Features:**")
        for feature in selected_features:
            dtype = "Numerical" if pd.api.types.is_numeric_dtype(df[feature].dtype) else "Categorical"
            st.write(f"- {feature} ({dtype})")
    
    with col2:
        st.write("**Target Variable:**")
        st.write(f"- {target_column} ({problem_type})")
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_layers = st.text_input(
            "Hidden Layer Sizes:",
            value="128, 64, 32",
            help="Comma-separated list of layer sizes (e.g., 128, 64, 32)"
        )
        epochs = st.slider("Training Epochs:", 10, 500, 100)
    
    with col2:
        learning_rate = st.selectbox(
            "Learning Rate:",
            [0.001, 0.01, 0.1, 0.0001],
            index=0
        )
        test_size = st.slider("Test Set Size (%):", 10, 40, 20)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
            dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.3)
        with col2:
            activation = st.selectbox("Activation Function:", ["relu", "tanh", "sigmoid"])
            early_stopping = st.checkbox("Use Early Stopping", value=True)
    
    # Train model
    if st.button("🚀 Train Neural Network", type="primary"):
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
            return
        
        try:
            with st.spinner("Preprocessing data and training neural network..."):
                # Prepare data
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                # Preprocess features
                X_processed, preprocessing_info = preprocess_features(X)
                y_processed, target_info = preprocess_target(y, problem_type)
                
                # Show preprocessing results
                st.subheader("🔄 Preprocessing Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Features Processing:**")
                    st.write(f"- Original features: {X.shape[1]}")
                    st.write(f"- After encoding: {X_processed.shape[1]}")
                    if preprocessing_info['encoded_categorical']:
                        st.write(f"- Categorical features encoded: {len(preprocessing_info['encoded_categorical'])}")
                
                with col2:
                    st.write("**Target Processing:**")
                    st.write(f"- Problem type: {target_info['problem_type']}")
                    if target_info['encoded']:
                        st.write(f"- Unique classes: {target_info['n_classes']}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size/100, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model using the analytics module
                model, scaler, losses = analytics.train_tabular_nn(
                    pd.DataFrame(X_train_scaled), 
                    pd.Series(y_train),
                    hidden_layers=[int(x.strip()) for x in hidden_layers.split(',')],
                    epochs=epochs
                )
                
                # Display results
                st.success("✅ Training completed!")
                
                # Plot training loss
                st.subheader("📈 Training Progress")
                fig_loss = px.line(
                    x=range(len(losses)),
                    y=losses,
                    title="Training Loss Over Time",
                    labels={'x': 'Epoch', 'y': 'Loss'}
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Model summary
                st.subheader("📋 Model Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Loss", f"{losses[-1]:.4f}")
                with col2:
                    st.metric("Features Used", X_processed.shape[1])
                with col3:
                    st.metric("Training Epochs", epochs)
                with col4:
                    st.metric("Test Size", f"{test_size}%")
                
                # Make predictions
                import torch
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_scaled)
                    predictions = model(X_test_tensor).numpy().flatten()
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
                
                if problem_type == "regression":
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    
                    st.subheader("📊 Regression Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    with col2:
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Plot predictions vs actual
                    fig_pred = px.scatter(
                        x=y_test,
                        y=predictions,
                        title="Predictions vs Actual Values",
                        labels={'x': 'Actual', 'y': 'Predicted'}
                    )
                    fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), 
                                     x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash'))
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                else:
                    # For classification, convert probabilities to classes
                    predicted_classes = (predictions > 0.5).astype(int)
                    accuracy = accuracy_score(y_test, predicted_classes)
                    
                    st.subheader("📊 Classification Results")
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, predicted_classes)
                    
                    fig_cm = px.imshow(
                        cm,
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Class 0', 'Class 1'],
                        y=['Class 0', 'Class 1']
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.info("💡 Tip: Make sure your data contains numerical values or categorical data that can be encoded.")

def preprocess_features(X):
    """Preprocess features for neural network training"""
    X_processed = X.copy()
    preprocessing_info = {
        'encoded_categorical': [],
        'scaled_numerical': []
    }
    
    # Encode categorical features
    for col in X_processed.columns:
        if not pd.api.types.is_numeric_dtype(X_processed[col]):
            # One-hot encode categorical variables with few unique values
            if X_processed[col].nunique() <= 10:
                encoded = pd.get_dummies(X_processed[col], prefix=col)
                X_processed = pd.concat([X_processed, encoded], axis=1)
                X_processed.drop(col, axis=1, inplace=True)
                preprocessing_info['encoded_categorical'].append(col)
            else:
                # Label encode for high cardinality
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                preprocessing_info['encoded_categorical'].append(f"{col} (label encoded)")
    
    # Convert to numeric and handle any remaining non-numeric values
    X_processed = X_processed.apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values with 0
    X_processed = X_processed.fillna(0)
    
    return X_processed, preprocessing_info

def preprocess_target(y, problem_type):
    """Preprocess target variable"""
    y_processed = y.copy()
    target_info = {
        'problem_type': problem_type,
        'encoded': False,
        'n_classes': None
    }
    
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_processed):
        # Encode categorical target
        le = LabelEncoder()
        y_processed = le.fit_transform(y_processed)
        target_info['encoded'] = True
        target_info['n_classes'] = len(le.classes_)
        target_info['classes'] = le.classes_
    
    # Convert to numeric
    y_processed = pd.to_numeric(y_processed, errors='coerce')
    
    # Fill NaN values with 0
    y_processed = y_processed.fillna(0)
    
    return y_processed, target_info