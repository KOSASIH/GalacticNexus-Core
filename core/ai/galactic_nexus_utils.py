import numpy as np
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Preprocess data using advanced techniques (e.g., tokenization, normalization)
    return data

def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
