import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class TradingStrategyOptimizer:
    def __init__(self):
        self.rfc = RandomForestClassifier()
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10]
        }

    def optimize(self, data: pd.DataFrame) -> RandomForestClassifier:
        grid_search = GridSearchCV(self.rfc, self.param_grid, cv=5, scoring='accuracy')
        grid_search.fit(data.drop('target', axis=1), data['target'])
        return grid_search.best_estimator_
