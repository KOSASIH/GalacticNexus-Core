import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class TradingStrategy:
    def __init__(self):
        self.rfc = RandomForestClassifier(n_estimators=100)

    def train(self, data: pd.DataFrame):
        self.rfc.fit(data.drop('target', axis=1), data['target'])

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.rfc.predict(data)
