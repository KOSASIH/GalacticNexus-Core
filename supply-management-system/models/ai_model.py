# models/ai_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class AIModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.data = pd.read_csv('supply_chain_data.csv')

    def train_model(self):
        # Train the AI model using the supply chain data
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, input_data):
        # Use the trained AI model to make predictions on new input data
        return self.model.predict(input_data)
