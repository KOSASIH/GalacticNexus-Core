import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ai_ml_models import GalacticNexusAI, GalacticNexusML

class AIMLIntegration:
    def __init__(self):
        self.galactic_nexus_ai = GalacticNexusAI()
        self.galactic_nexus_ml = GalacticNexusML()

    def train_ai_model(self, X, y):
        self.galactic_nexus_ai.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.galactic_nexus_ai.fit(X, y, epochs=10, batch_size=128)

    def train_ml_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.galactic_nexus_ml.fit(X_train, y_train)
        y_pred = self.galactic_nexus_ml.predict(X_test)
        print("ML Model Accuracy:", accuracy_score(y_test, y_pred))

    def integrate_ai_ml(self, X, y):
        self.train_ai_model(X, y)
        self.train_ml_model(X, y)
        ai_pred = self.galactic_nexus_ai.predict(X)
        ml_pred = self.galactic_nexus_ml.predict(X)
        combined_pred = np.concatenate((ai_pred, ml_pred), axis=1)
        return combined_pred

# Example usage
X = np.random.rand(100, 128, 128, 3)
y = np.random.randint(0, 10, 100)
aiml_integration = AIMLIntegration()
combined_pred = aiml_integration.integrate_ai_ml(X, y)
print("Combined Prediction:", combined_pred)
