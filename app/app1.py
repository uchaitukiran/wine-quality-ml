import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with advanced styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
        50% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.8); }
        100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #5D4037;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 1.2s ease-out;
        background: linear-gradient(45deg, #8D6E63, #A1887F, #BCAAA4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
    }
    
    .main-header::before {
        content: "";
        display: block;
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #8D6E63, transparent);
        margin: 0.5rem auto;
        animation: pulse 2s infinite;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #8D6E63, #A1887F);
        color: white;
        font-weight: 700;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #6D4C41, #8D6E63);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .sidebar {
        animation: slideIn 0.8s ease-out;
        background: linear-gradient(180deg, #EFEBE9, #D7CCC8);
        border-right: 1px solid rgba(141, 110, 99, 0.2);
    }
    
    .model-section {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(141, 110, 99, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid rgba(141, 110, 99, 0.1);
        cursor: pointer;
    }
    
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: rgba(141, 110, 99, 0.3);
    }
    
    .model-card.selected {
        background: linear-gradient(135deg, #EFEBE9, #D7CCC8);
        border-color: #8D6E63;
    }
    
    .model-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #5D4037;
        margin-bottom: 5px;
    }
    
    .model-type {
        font-size: 0.9rem;
        color: #795548;
        margin-bottom: 10px;
    }
    
    .model-info {
        font-size: 0.85rem;
        color: #5D4037;
        background-color: rgba(141, 110, 99, 0.1);
        padding: 8px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    .feature-section {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(141, 110, 99, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #EFEBE9, #D7CCC8);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        animation: fadeIn 1s ease-out, slideIn 0.8s ease-out;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(141, 110, 99, 0.2);
    }
    
    .prediction-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
        animation: glow 3s infinite;
    }
    
    .quality-score {
        font-size: 5rem;
        font-weight: 900;
        margin: 10px 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .quality-excellent { 
        color: #4CAF50; 
        animation: bounce 1s infinite;
    }
    
    .quality-good { 
        color: #8BC34A; 
        animation: pulse 2s infinite;
    }
    
    .quality-average { 
        color: #FFC107; 
        animation: pulse 2.5s infinite;
    }
    
    .quality-poor { 
        color: #FF9800; 
        animation: pulse 3s infinite;
    }
    
    .quality-bad { 
        color: #F44336; 
        animation: pulse 3.5s infinite;
    }
    
    .wine-icon {
        font-size: 4rem;
        margin-bottom: 10px;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .loading-container {
        display: none;
        text-align: center;
        margin: 20px 0;
    }
    
    .loading-spinner {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #8D6E63;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
    }
    
    .gauge-container {
        animation: slideInRight 0.8s ease-out;
    }
    
    .quality-description {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 15px;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .quality-excellent-desc {
        background-color: rgba(76, 175, 80, 0.2);
        color: #2E7D32;
    }
    
    .quality-good-desc {
        background-color: rgba(139, 195, 74, 0.2);
        color: #558B2F;
    }
    
    .quality-average-desc {
        background-color: rgba(255, 193, 7, 0.2);
        color: #F57C00;
    }
    
    .quality-poor-desc {
        background-color: rgba(255, 152, 0, 0.2);
        color: #E65100;
    }
    
    .quality-bad-desc {
        background-color: rgba(244, 67, 54, 0.2);
        color: #B71C1C;
    }
    
    .feature-value {
        font-size: 0.9rem;
        color: #795548;
        margin-top: 5px;
        padding: 5px 10px;
        background-color: #EFEBE9;
        border-radius: 5px;
        display: inline-block;
    }
    
    .feature-card {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #8D6E63;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #5D4037;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    .result-section {
        animation: slideInRight 0.8s ease-out;
    }
    
    .stDataFrame {
        animation: fadeIn 0.5s ease-out;
    }
    
    .stAlert {
        animation: fadeIn 0.5s ease-out;
    }
    
    .info-box {
        background-color: #EFEBE9;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        animation: fadeIn 0.5s ease-out;
        border-left: 4px solid #8D6E63;
    }
    
    .feature-slider {
        margin-bottom: 15px;
    }
    
    .feature-label {
        font-weight: 600;
        color: #5D4037;
        margin-bottom: 5px;
        display: block;
    }
    
    .feature-help {
        font-size: 0.8rem;
        color: #795548;
        margin-top: 2px;
        font-style: italic;
    }
    
    .model-badge {
        display: inline-block;
        background: linear-gradient(45deg, #8D6E63, #A1887F);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .quality-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    
    .quality-excellent-indicator {
        background-color: rgba(76, 175, 80, 0.2);
        color: #2E7D32;
    }
    
    .quality-good-indicator {
        background-color: rgba(139, 195, 74, 0.2);
        color: #558B2F;
    }
    
    .quality-average-indicator {
        background-color: rgba(255, 193, 7, 0.2);
        color: #F57C00;
    }
    
    .quality-poor-indicator {
        background-color: rgba(255, 152, 0, 0.2);
        color: #E65100;
    }
    
    .quality-bad-indicator {
        background-color: rgba(244, 67, 54, 0.2);
        color: #B71C1C;
    }
    
    .feature-importance-chart {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    .wine-pattern {
        position: absolute;
        bottom: 0;
        right: 0;
        font-size: 200px;
        opacity: 0.05;
        z-index: 0;
        animation: float 8s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- File Paths and Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Sidebar ---
st.sidebar.title("🍷 Wine Quality Predictor")
st.markdown("---")

# Model Selection at Top of Sidebar
st.sidebar.subheader("🤖 Model Selection")
try:
    # Define the list of models to be used
    MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "KNeighborsClassifier", "GaussianNB"]
    
    # Check if all required models exist
    missing_models = [name for name in MODEL_NAMES if not os.path.exists(os.path.join(MODEL_DIR, f"{name}.pkl"))]
    
    if missing_models:
        st.sidebar.error(f"Missing models: {', '.join(missing_models)}")
        st.sidebar.info("Please run your training script to create model files.")
        st.stop()
    else:
        # Display model cards
        for model_name in MODEL_NAMES:
            try:
                model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
                model = joblib.load(model_name)
                
                with st.sidebar.container():
                    st.markdown(f"""
                    <div class="model-card">
                        <div class="model-name">
                            {model_name}
                            <span class="model-badge">{type(model).__name__}</span>
                        </div>
                        <div class="model-info">
                            Features: {len(model.coef_[0]) if hasattr(model, 'coef_') else 'N/A'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.sidebar.error(f"Error loading {model_name}: {str(e)}")
                continue
        
        # Create a dropdown for model selection
        model_selection = st.sidebar.selectbox(
            "Select Model for Prediction",
            MODEL_NAMES,
            help="Choose which machine learning model to use for prediction"
        )
        
        # Load the selected model
        model_path = os.path.join(MODEL_DIR, f"{model_selection}.pkl")
        model = joblib.load(model_path)
        
        # Load the scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        st.sidebar.success(f"✅ {model_selection} loaded successfully")
        
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.sidebar.info("Make sure models and scaler are in the models directory")
    st.stop()

# --- Wine Features Section in Sidebar ---
with st.sidebar.expander("🍇 Wine Features", expanded=True):
    st.markdown("---")
    
    # Create two columns for features
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        fixed_acidity = st.slider(
            "Fixed Acidity", 
            min_value=0.0, 
            max_value=20.0, 
            value=7.4, 
            step=0.1,
            help="Tartaric acid - affects taste and preservation"
        )
        volatile_acidity = st.slider(
            "Volatile Acidity", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.7, 
            step=0.01,
            help="Acetic acid - can give vinegar taste at high levels"
        )
        citric_acid = st.slider(
            "Citric Acid", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.01,
            help="Adds freshness and flavor to wines"
        )
        residual_sugar = st.slider(
            "Residual Sugar", 
            min_value=0.0, 
            max_value=20.0, 
            value=1.9, 
            step=0.1,
            help="Natural sugars remaining after fermentation"
        )
        chlorides = st.slider(
            "Chlorides", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.076, 
            step=0.001,
            help="Salt content - affects taste"
        )
        free_sulfur_dioxide = st.slider(
            "Free Sulfur Dioxide", 
            min_value=0, 
            max_value=100, 
            value=11, 
            step=1,
            help="Prevents microbial growth and oxidation"
        )
    
    with feature_col2:
        total_sulfur_dioxide = st.slider(
            "Total Sulfur Dioxide", 
            min_value=0, 
            max_value=300, 
            value=34, 
            step=1,
            help="Total SO2 in wine - preservative"
        )
        density = st.slider(
            "Density", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.9978, 
            step=0.0001,
            help="Similar to water, affected by sugar and alcohol content"
        )
        pH = st.slider(
            "pH", 
            min_value=0.0, 
            max_value=14.0, 
            value=3.51, 
            step=0.01,
            help="Acidity level - affects taste and preservation"
        )
        sulphates = st.slider(
            "Sulphates", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.56, 
            step=0.01,
            help="Additive which can contribute to SO2 levels"
        )
        alcohol = st.slider(
            "Alcohol", 
            min_value=0.0, 
            max_value=20.0, 
            value=9.4, 
            step=0.1,
            help="Ethanol content - affects body and taste"
        )

# --- Main Content Area ---
st.markdown('<div class="main-header">Wine Quality Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

# Create two columns for main content
col1, col2 = st.columns([1, 1])

# Left Column: Input Features Display
with col1:
    st.mark('<div class="section-title"><span class="section-icon">📊</span>Input Features</div>', unsafe_allow_html=True)
    
    # Display current values in a table
    feature_data = {
        'Fixed Acidity': fixed_acidity,
        'Volatile Acidity': volatile_acidity,
        'Citric Acid': citric_acid,
        'Residual Sugar': residual_sugar,
        'Chlorides': chlorides,
        'Free SO₂': free_sulfur_dioxide,
        'Total SO₂': total_sulfur_dioxide,
        'Density': density,
        'pH': pH,
        'Sulphates': sulphates,
        'Alcohol': alcohol
    }
    
    feature_df = pd.DataFrame(list(feature_data.items()), columns=['Feature', 'Value'])
    st.dataframe(feature_df.style.background_gradient(cmap='YlOrBr'), use_container_width=True)
    
    # Feature descriptions
    st.mark('<div class="section-title"><span class="section-icon">📝</span>Feature Descriptions</div>', unsafe_allow_html=True)
    feature_descriptions = {
        'Fixed Acidity': 'Tartaric acid - affects taste and preservation',
        'Volatile Acidity': 'Acetic acid - can give vinegar taste at high levels',
        'Citric Acid': 'Adds freshness and flavor to wines',
        'Residual Sugar': 'Natural sugars remaining after fermentation',
        'Chlorides': 'Salt content - affects taste',
        'Free SO₂': 'Prevents microbial growth and oxidation',
        'Total SO₂': 'Total SO2 in wine - preservative',
        'Density': 'Similar to water, affected by sugar and alcohol content',
        'pH': 'Acidity level - affects taste and preservation',
        'Sulphates': 'Additive which can contribute to SO2 levels',
        'Alcohol': 'Ethanol content - affects body and taste'
    }
    
    for feature, desc in feature_descriptions.items():
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-label">{feature.replace('_', ' ').title()}</div>
            <div class="feature-value">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Right Column: Prediction Results
with col2:
    st.mark(f'<div class="section-title"><span class="section-icon">🎯</span>Prediction Results - {model_selection}</div>', unsafe_allow_html=True)
    
    # Create input data for prediction
    input_data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Standardize the user input before making predictions
    input_scaled = scaler.transform(input_df)
    
    # Prediction button
    if st.button(f"🔮 Predict Wine Quality", type="primary"):
        # Show loading spinner
        loading_container = st.empty()
        loading_container.markdown("""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <p style="margin-top: 10px; color: #795548;">Predicting wine quality...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate processing time
        time.sleep(1.5)
        
        # Remove loading spinner
        loading_container.empty()
        
        # Make prediction
        pred = model.predict(input_scaled)[0]
        
        # Display prediction result with appropriate styling
        st.subheader("🎯 Prediction Result")
        
        # Determine quality category and styling
        if pred >= 7:
            quality_class = "quality-excellent"
            quality_text = "Excellent"
            desc_class = "quality-excellent-desc"
            indicator_class = "quality-excellent-indicator"
        elif pred >= 6:
            quality_class = "quality-good"
            quality_text = "Good"
            desc_class = "quality-good-desc"
            indicator_class = "quality-good-indicator"
        elif pred >= 5:
            quality_class = "quality-average"
            quality_text = "Average"
            desc_class = "quality-average-desc"
            indicator_class = "quality-average-indicator"
        elif pred >= 4:
            quality_class = "quality-poor"
            quality_text = "Poor"
            desc_class = "quality-poor-desc"
            indicator_class = "quality-poor-indicator"
        else:
            quality_class = "quality-bad"
            quality_text = "Bad"
            desc_class = "quality-bad-desc"
            indicator_class = "quality-bad-indicator"
        
        # Create gauge chart for prediction
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Wine Quality Score", 'font': {'size': 16}},
            delta={'reference': 5.0},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 6], 'color': "gray"},
                    {'range': [6, 8], 'color': "lightgreen"},
                    {'range': [8, 10], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': pred
                }
            }
        ))
        
        fig.update_layout(height=300, margin={'l': 0, 'r': 0, 't': 30, 'b': 0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the numerical result with emphasis
        st.markdown(f"""
        <div class="prediction-container">
            <div class="wine-icon">🍷</div>
            <div class="quality-score {quality_class}">{pred:.2f}</div>
            <div class="quality-description {desc_class}">{quality_text} Quality</div>
            <div class="quality-indicator {indicator_class}">
                <span>🏆</span> {quality_text} Wine
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add quality description
        if pred < 4:
            st.error("This wine has poor quality. It may have unpleasant flavors and aromas.")
        elif pred < 6:
            st.warning("This wine has average quality. It's drinkable but not exceptional.")
        elif pred < 8:
            st.success("This wine has good quality. It has balanced flavors and is enjoyable.")
        else:
            st.success("This wine has excellent quality. It has exceptional complexity and balance.")
        
        # Feature contribution analysis if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Contribution")
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            ax.set_title(f'Feature Importance - {model_selection}')
            st.pyplot(fig)

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #795548;'>"
    "Wine Quality Predictor | Powered by Machine Learning"
    "</div>", unsafe_allow_html=True
)