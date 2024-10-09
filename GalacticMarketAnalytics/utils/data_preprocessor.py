import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    scaler = MinMaxScaler()
    df["open"] = scaler.fit_transform(df["open"].values.reshape(-1, 1))
    df["high"] = scaler.fit_transform(df["high"].values.reshape(-1, 1))
    df["low"] = scaler.fit_transform(df["low"].values.reshape(-1, 1))
    df["close"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))
    df["volume"] = scaler.fit_transform(df["volume"].values.reshape(-1, 1))
    return df
