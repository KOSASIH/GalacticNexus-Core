# ai/security/anomaly_detection.py
from sklearn.ensemble import IsolationForest

class ThreatDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100)

    def fit(self, X):
        self.model.fit(X)

    def detect(self, X):
        return self.model.predict(X)  # -1 is anomaly, 1 is normal
