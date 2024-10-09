import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def save_model(self, filename):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

# Example usage:
max_depth = 5
decision_tree = DecisionTree(max_depth)

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

decision_tree.train_model(X_train, y_train)
accuracy = decision_tree.evaluate_model(X_test, y_test)
print(f"Accuracy: {accuracy}")

decision_tree.save_model("decision_tree.pkl")
