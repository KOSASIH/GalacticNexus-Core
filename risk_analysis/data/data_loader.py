# data_loader.py

import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data from CSV file
data = load_data("risk_data.csv")
