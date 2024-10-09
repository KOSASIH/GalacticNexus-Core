import pandas as pd

def preprocess_data(data):
    # Preprocess spacecraft data
    data = data.dropna()  # Remove missing values
    data = data.astype(float)  # Convert data to float
    return data
