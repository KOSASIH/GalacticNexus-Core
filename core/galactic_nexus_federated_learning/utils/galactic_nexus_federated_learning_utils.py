import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def federated_learning(models, X_train, y_train, X_val, y_val, num_rounds):
    for i in range(num_rounds):
        for j in range(len(models)):
            X_train_client = X_train[j]
            y_train_client = y_train[j]
            models[j].fit(X_train_client, y_train_client, epochs=10, batch_size=32)

        ensemble_prediction = []
        for model in models:
            y_pred = model.predict(X_val)
            ensemble_prediction.append(y_pred)

        ensemble_prediction = np.mean(ensemble_prediction, axis=0)
        ensemble_prediction_class = np.argmax(ensemble_prediction, axis=1)
        accuracy = accuracy_score(y_val, ensemble_prediction_class)
        print("Round", i, "Accuracy:", accuracy)

    return accuracy
