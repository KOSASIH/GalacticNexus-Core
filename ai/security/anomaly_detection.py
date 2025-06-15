# ai/security/anomaly_detection.py

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import logging

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ThreatDetector:
    def __init__(self, algorithm='isolation_forest', model_path='threat_detector_model.pkl', **kwargs):
        self.algorithm = algorithm
        self.model_path = model_path
        self.logger = logging.getLogger("ThreatDetector")
        self.logger.setLevel(logging.INFO)
        if algorithm == 'isolation_forest':
            self.model = IsolationForest(n_estimators=kwargs.get('n_estimators', 100))
        elif algorithm == 'oneclass_svm':
            self.model = OneClassSVM(nu=kwargs.get('nu', 0.05))
        elif algorithm == 'autoencoder' and torch:
            self.input_dim = kwargs.get('input_dim', 16)
            self.model = SimpleAutoEncoder(self.input_dim, encoding_dim=kwargs.get('encoding_dim', 8))
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 1e-3))
        else:
            raise ValueError("Unsupported algorithm or missing torch for autoencoder.")

        self.threshold = kwargs.get('threshold', None)  # for autoencoder

    def fit(self, X):
        self.logger.info(f"Fitting {self.algorithm} model.")
        if self.algorithm in ['isolation_forest', 'oneclass_svm']:
            self.model.fit(X)
        elif self.algorithm == 'autoencoder' and torch:
            X_tensor = torch.FloatTensor(X)
            for epoch in range(100):  # simple training loop
                output = self.model(X_tensor)
                loss = self.criterion(output, X_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Set anomaly threshold as mean + 2*std of training errors
            recon_errors = ((self.model(X_tensor) - X_tensor)**2).mean(dim=1).detach().numpy()
            self.threshold = np.mean(recon_errors) + 2 * np.std(recon_errors)
        self.save()

    def detect(self, X):
        if self.algorithm in ['isolation_forest', 'oneclass_svm']:
            preds = self.model.predict(X)  # -1 is anomaly
            return preds
        elif self.algorithm == 'autoencoder' and torch:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                recon_errors = ((self.model(X_tensor) - X_tensor)**2).mean(dim=1).numpy()
            return np.where(recon_errors > self.threshold, -1, 1)

    def explain(self, X):
        """Returns anomaly score for each instance for explainability."""
        if self.algorithm == 'isolation_forest':
            return self.model.decision_function(X)  # lower = more anomalous
        elif self.algorithm == 'oneclass_svm':
            return self.model.decision_function(X)
        elif self.algorithm == 'autoencoder' and torch:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                recon_errors = ((self.model(X_tensor) - X_tensor)**2).mean(dim=1).numpy()
            return recon_errors

    def save(self):
        if self.algorithm in ['isolation_forest', 'oneclass_svm']:
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        elif self.algorithm == 'autoencoder' and torch:
            torch.save(self.model.state_dict(), self.model_path)
            self.logger.info(f"Autoencoder model saved to {self.model_path}")

    def load(self):
        if self.algorithm in ['isolation_forest', 'oneclass_svm']:
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        elif self.algorithm == 'autoencoderencoder model loadedDetector(algorithm='isolation_forest')
# detector.fit(X_train)
# anomalies = detector.detect(X_test)
# scores = detector.explain(X_test)
