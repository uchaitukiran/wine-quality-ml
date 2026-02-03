import os
import streamlit as st
import pandas as pd
import joblib

# --- File Paths and Setup ---
# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Streamlit App UI ---
st.title("🍷 Wine Quality Prediction")
st.markdown("Enter the wine's features below and select a model to predict its quality.")

st.header("Input Wine Features")
def get_input():
    """Creates a form for user input and returns a DataFrame."""
    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
        volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
        residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
        chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0, 100, 11)
    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0, 300, 34)
        density = st.number_input("Density", 0.0, 2.0, 0.9978)
        pH = st.number_input("pH", 0.0, 14.0, 3.51)
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
        alcohol = st.number_input("Alcohol", 0.0, 20.0, 9.4)
    
    data = {
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
    return pd.DataFrame(data, index=[0])

input_df = get_input()

st.subheader("Input Features")
st.write(input_df)

# --- Model Loading and Prediction ---
try:
    # Load the scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Define the list of models to be used
    MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "KNeighborsClassifier", "GaussianNB"]
    
    # Check if all required models exist
    missing_models = [name for name in MODEL_NAMES if not os.path.exists(os.path.join(MODEL_DIR, f"{name}.pkl"))]
    
    if missing_models:
        st.error(f"Error: The following models were not found: {', '.join(missing_models)}")
        st.info("Please run your `train.py` script to create these model files.")
    else:
        # Create a dropdown for model selection
        model_selection = st.selectbox(
            "Select a Model for Prediction",
            MODEL_NAMES
        )

        # Load the selected model
        model_path = os.path.join(MODEL_DIR, f"{model_selection}.pkl")
        model = joblib.load(model_path)

        # Standardize the user input before making predictions
        input_scaled = scaler.transform(input_df)

        if st.button(f"Predict Quality with {model_selection}"):
            pred = model.predict(input_scaled)[0]
            st.subheader("Prediction Result")
            st.success(f"The predicted quality is: **{int(pred)}**")

except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please ensure your 'models' directory is correctly set up.")
    st.error(f"Missing file: {e}")
    st.info("Make sure you have run the `train.py` script to create the models and scaler.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
