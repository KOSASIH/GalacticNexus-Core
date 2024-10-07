import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def uncertainty_sampling(predictions):
    uncertainties = np.max(predictions, axis=1)
    return uncertainties

def margin_sampling(predictions):
    margins = np.max(predictions, axis=1) - np.max(predictions, axis=1)[1:]
    return margins

def entropy_sampling(predictions):
    entropies = -np.sum(predictions * np.log(predictions), axis=1)
    return entropies

def select_samples(X_unlabeled, num_samples, uncertainties):
    indices = np.argsort(uncertainties)[-num_samples:]
    return X_unlabeled[indices]

def active_learning(X_train, y_train, X_unlabeled, num_samples, num_iterations):
    for i in range(num_iterations):
        X_selected = select_samples(X_unlabeled, num_samples, uncertainty_sampling(X_unlabeled))
        y_selected = np.argmax(X_selected, axis=1)
        X_train = np.concatenate((X_train, X_selected))
        y_train = np.concatenate((y_train, y_selected))

    return X_train, y_train
