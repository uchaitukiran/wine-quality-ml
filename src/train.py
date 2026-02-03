import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# --- File Paths and Setup ---
# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality_red.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create the models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # Load dataset with the correct separator (comma)
    df = pd.read_csv(DATA_PATH, sep=",")
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Split features (X) and target (y)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Dictionary of models to train
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "GaussianNB": GaussianNB()
    }

    # Train and save each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {name} model at {model_path}")

    # Save the scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved at {scaler_path}")

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
