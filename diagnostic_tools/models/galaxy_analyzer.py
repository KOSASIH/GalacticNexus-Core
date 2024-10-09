import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class GalaxyAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def analyze_galaxy_data(self):
        # Preprocess data
        X = self.data.drop('galaxy_type', axis=1)
        y = self.data['galaxy_type']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a random forest classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        return clf
