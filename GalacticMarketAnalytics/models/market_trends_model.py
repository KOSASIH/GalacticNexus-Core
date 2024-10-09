import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MarketTrendsModel:
    def __init__(self, config):
        self.config = config
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Market Trends Model MSE: {mse:.2f}")

    def predict(self, X):
        return self.model.predict(X)
