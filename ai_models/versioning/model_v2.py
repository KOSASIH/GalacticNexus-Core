# Galactic Nexus AI Model V2
# Author: KOSASIH
# Date: 10/09/2024

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class GalacticNexusModelV2:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.version = "2.0.0"

    def train(self, dataset):
        X = dataset.drop('target', axis=1)
        y = dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        print("Model V2 Training Report:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, input_data):
        input_data_scaled = self.scaler.transform(input_data)
        return self.model.predict(input_data_scaled)

    def get_version(self):
        return self.version

    def save_model(self, file_path):
        import joblib
        joblib.dump((self.model, self.scaler), file_path)

    def load_model(self, file_path):
        import joblib
        self.model, self.scaler = joblib.load(file_path)
