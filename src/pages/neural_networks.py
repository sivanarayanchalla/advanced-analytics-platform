import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def show_neural_networks(analytics):
    """Neural Networks page using scikit-learn MLP"""
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
            value="100, 50",
            help="Comma-separated list of layer sizes (e.g., 100, 50)"
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
            activation = st.selectbox("Activation Function:", ["relu", "tanh", "logistic"])
            early_stopping = st.checkbox("Use Early Stopping", value=True)
        with col2:
            solver = st.selectbox("Solver:", ["adam", "lbfgs", "sgd"])
            shuffle = st.checkbox("Shuffle Data", value=True)
    
    # Train model
    if st.button("🚀 Train Neural Network", type="primary"):
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
            return
        
        try:
            with st.spinner("Training neural network..."):
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Train model using the CORRECT method name
                model, scaler, losses = analytics.train_neural_network(
                    X, y, 
                    hidden_layers=[int(x.strip()) for x in hidden_layers.split(',')],
                    problem_type=problem_type,
                    epochs=epochs
                )
                
                # Display results
                st.success("✅ Training completed!")
                
                # Plot training loss if available
                if losses:
                    st.subheader("📈 Training Progress")
                    fig_loss = px.line(
                        x=range(len(losses)),
                        y=losses,
                        title="Training Loss Over Time",
                        labels={'x': 'Epoch', 'y': 'Loss'}
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.info("📊 Training loss curve not available for this solver.")
                
                # Model summary
                st.subheader("📋 Model Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if losses:
                        final_loss = losses[-1]
                        st.metric("Final Loss", f"{final_loss:.4f}")
                    else:
                        st.metric("Training", "Completed")
                with col2:
                    st.metric("Features Used", len(selected_features))
                with col3:
                    st.metric("Training Epochs", epochs)
                with col4:
                    st.metric("Problem Type", problem_type)
                
                # Make predictions and show metrics
                X_processed, y_processed = analytics.preprocess_data(X, y)
                X_scaled = scaler.transform(X_processed)
                predictions = model.predict(X_scaled)
                
                from sklearn.metrics import accuracy_score, r2_score, classification_report
                
                if problem_type == "regression":
                    r2 = r2_score(y_processed, predictions)
                    mse = np.mean((y_processed - predictions) ** 2)
                    
                    st.subheader("📊 Regression Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("MSE", f"{mse:.4f}")
                    
                    # Plot predictions vs actual
                    fig_pred = px.scatter(
                        x=y_processed,
                        y=predictions,
                        title="Predictions vs Actual Values",
                        labels={'x': 'Actual', 'y': 'Predicted'}
                    )
                    fig_pred.add_shape(
                        type='line', 
                        x0=y_processed.min(), 
                        y0=y_processed.min(), 
                        x1=y_processed.max(), 
                        y1=y_processed.max(), 
                        line=dict(dash='dash', color='red')
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                else:
                    # For classification
                    accuracy = accuracy_score(y_processed, predictions)
                    
                    st.subheader("📊 Classification Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Classes", f"{len(np.unique(y_processed))}")
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_processed, predictions)
                    
                    fig_cm = px.imshow(
                        cm,
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[f'Class {i}' for i in range(cm.shape[1])],
                        y=[f'Class {i}' for i in range(cm.shape[0])],
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Classification report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_processed, predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, width='stretch')
                
                # Model information
                st.subheader("🔍 Model Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Type:**", type(model).__name__)
                    st.write("**Layers:**", model.hidden_layer_sizes)
                    st.write("**Activation:**", model.activation)
                
                with col2:
                    st.write("**Solver:**", model.solver)
                    st.write("**Iterations:**", model.n_iter_)
                    st.write("**Layers:**", len(model.coefs_))
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.info("💡 Tip: Make sure your data contains numerical values or categorical data that can be encoded.")

def show_neural_network_info():
    """Show information about neural networks"""
    with st.expander("💡 About Neural Networks"):
        st.markdown("""
        **Multi-Layer Perceptron (MLP) Neural Networks:**
        
        - **Input Layer**: Your selected features (automatically encoded)
        - **Hidden Layers**: Learn complex patterns in the data
        - **Output Layer**: Prediction (regression or classification)
        
        **Advantages:**
        - Can learn non-linear relationships
        - Works with both numerical and categorical data
        - Automatic feature preprocessing
        
        **Tips for Better Results:**
        - Use more data for better performance
        - Scale numerical features (automatically done)
        - Start with simpler architectures (e.g., 50, 25)
        - Use more epochs for complex patterns
        """)

# Call the info function at the bottom
show_neural_network_info()