from galactic_nexus_neuro import GalacticNexusNeuro
from galactic_nexus_neuro_config import config
from sklearn.datasets import load_mnist
from sklearn.model_selection import train_test_split

# Load dataset
(X_train, y_train), (X_test, y_test) = load_mnist()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create neuro instance
neuro = GalacticNexusNeuro(config)

# Perform neuro-feedback
uncertainties = neuro.neuro_feedback(X_val)
print("Neuro-Feedback:", uncertainties)

# Perform neuro-control
controls = neuro.neuro_control(X_val)
print("Neuro-Control:", controls)

# Perform neuro-learning
X_train, y_train = neuro.neuro_learning(X_train, y_train, X_val, config['num_iterations'])
print("Neuro-Learning:", X_train.shape, y_train.shape)

# Train model
history = neuro.train(X_train, y_train, X_val, y_val)
print("Training History:", history.history)

# Evaluate model
accuracy = neuro.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
