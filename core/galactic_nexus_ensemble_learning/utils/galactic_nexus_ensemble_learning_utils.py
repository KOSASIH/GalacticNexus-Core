import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def bagging(base_estimator, X_train, y_train, X_val, y_val, n_estimators):
    bagging_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def boosting(base_estimator, X_train, y_train, X_val, y_val, n_estimators):
    boosting_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    boosting_model.fit(X_train, y_train)
    y_pred = boosting_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def stacking(base_estimator, X_train, y_train, X_val, y_val, n_estimators):
 [INST] Stacking is a technique that combines multiple models by training a new model on the outputs of the base models. In this case, we can use a Gradient Boosting Classifier as the meta-learner.

Here's an example implementation of stacking in the `galactic_nexus_ensemble_learning_utils.py` file:
```python
def stacking(base_estimator, X_train, y_train, X_val, y_val, n_estimators):
    stacking_model = GradientBoostingClassifier(n_estimators=n_estimators)
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy
