import pandas as pd

def load_asset_data(file_path):
    # Load asset data from CSV file
    asset_data = pd.read_csv(file_path)
    return asset_data

def load_market_data(file_path):
    # Loadmarket data from CSV file
    market_data = pd.read_csv(file_path)
    return market_data
