import os
import sys
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from utils.data_preprocessing import preprocess_data
from utils.feature_extraction import extract_features
from ai_models.neural_networks import SpacecraftSystemNN

def load_data(data_dir):
    # Load spacecraft data from directory
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            data.append(pd.read_csv(os.path.join(data_dir, file)))
    return pd.concat(data, ignore_index=True)

def train_model(model, data, epochs=10):
    # Train neural network model on spacecraft data
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(data, epochs=epochs)
    return model

def diagnose_issue(model, data):
    # Use trained model to diagnose issue on spacecraft
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Load spacecraft data
    data_dir = "data/spacecraft_data"
    data = load_data(data_dir)

    # Preprocess data
    data = preprocess_data(data)

    # Extract features
    features = extract_features(data)

    # Train neural network model
    model = SpacecraftSystemNN()
    model = train_model(model, features)

    # Diagnose issue on spacecraft
    issue = diagnose_issue(model, features)
    print("Diagnosed issue:", issue)
