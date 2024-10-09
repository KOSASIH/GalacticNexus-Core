# Galactic Nexus AI Model V1
# Author: KOSASIH
# Date: 10/09/2024

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GalacticNexusModelV1:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.version = "1.0.0"

    def train(self, dataset):
        X = dataset.drop('target', axis=1)
        y = dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Model V1 Training Report:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, input_data):
        return self.model.predict(input_data)

    def get_version(self):
        return self.version

    def save_model(self, file_path):
        import joblib
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        import joblib
        self.model = joblib.load(file_path)
