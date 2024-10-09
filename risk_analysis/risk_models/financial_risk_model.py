import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

class FinancialRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Ensemble of models
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42),
            CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42),
            LGBMRegressor(objective='regression', max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42),
            BayesianRidge(),
            SVR(kernel='rbf', C=1e3, gamma=0.1)
        ]

        self.model = models[0]
        self.model.fit(X_train_scaled, y_train)

        # Hyperparameter tuning using Bayesian optimization
        from hyperopt import hp, fmin, tpe, Trials
        space = {
            'n_estimators': hp.quniform('n_estimators', 10, 100, 10),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', -5, 0)
        }
        trials = Trials()
        best = fmin(self.objective, space, algo=tpe.suggest, trials=trials, max_evals=50)
        self.model.set_params(**best)

    def objective(self, params):
        self.model.set_params(**params)
        y_pred = self.model.predict(self.scaler.transform(self.X_test))
        return mean_squared_error(self.y_test, y_pred)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def calculate_risk_score(self, X):
        return self.predict(X)
