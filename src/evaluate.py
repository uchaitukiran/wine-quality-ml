import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- File Paths and Setup ---
# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality_red.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Main Evaluation Logic ---
try:
    # Load the dataset
    df = pd.read_csv(DATA_PATH, sep=",")
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Check if the 'quality' column exists
    if 'quality' not in df.columns:
        raise KeyError("'quality' column not found in the dataset.")

    # Separate features (X) and target (y)
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Load the scaler used during training
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at '{scaler_path}'.")
        print("Please run your train.py script first to create the scaler.")
        exit()
    scaler = joblib.load(scaler_path)

    # Standardize the features using the loaded scaler
    X_scaled = scaler.transform(X)

    # Evaluate all models in the models directory
    print("\n--- Evaluating Models ---")
    model_count = 0
    
    # Define the list of models to be evaluated
    MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "KNeighborsClassifier", "GaussianNB"]
    
    for name in MODEL_NAMES:
        model_file = f"{name}.pkl"
        model_path = os.path.join(MODEL_DIR, model_file)
        
        if not os.path.exists(model_path):
            print(f"Skipping '{model_file}': File not found.")
            continue
            
        try:
            # Load the model using joblib
            model = joblib.load(model_path)
            
            # Make predictions on the scaled data
            y_pred = model.predict(X_scaled)

            # Print results
            print(f"\n🔹 Model: {name}")
            print("Accuracy:", accuracy_score(y, y_pred))
            print("Classification Report:\n", classification_report(y, y_pred, zero_division=0))
            model_count += 1

        except Exception as e:
            print(f"An error occurred while loading or evaluating model '{model_file}': {e}")
            print("Skipping this file.")

    if model_count == 0:
        print("No models found to evaluate. Please ensure your `train.py` script ran successfully.")

# --- Error Handling ---
except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found.")
    print("Please make sure the 'winequality_red.csv' file is in the 'data' directory.")
except KeyError as e:
    print(f"Error: {e}")
    print("The required column was not found in the dataset.")
    if 'df' in locals():
        print("Available columns are:")
        print(df.columns.tolist())
except Exception as e:
    print(f"An unexpected error occurred: {e}")
