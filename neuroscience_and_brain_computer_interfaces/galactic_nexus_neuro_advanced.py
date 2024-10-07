from galactic_nexus_neuro import GalacticNexusNeuro
from galactic_nexus_neuro_config import config
from sklearn.datasets import load_mnist
from sklearn.model_selection import train_test_split

# Load dataset
(X_train, y_train), (X_test, y_test) = load_mnist()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create neuro instance
neuro = GalacticNexusNeuro(config)

# Perform neuro-feedback with multiple iterations
uncertainties = []
for i in range(config['num_iterations']):
    uncertainties.append(neuro.neuro_feedback(X_val))
print("Neuro-Feedback with Multiple Iterations:", uncertainties)

# Perform neuro-control with multiple iterations
controls = []
for i in range(config['num_iterations']):
    controls.append(neuro.neuro_control(X_val))
print("Neuro-Control with Multiple Iterations:", controls)

# Perform neuro-learning with multiple iterations
X_train, y_train = neuro.neuro_learning(X_train, y_train, X_val, config['num_iterations'])
print("Neuro-Learning with Multiple Iterations:", X_train.shape, y_train.shape)

# Train model with multiple iterations
history = []
for i in range(config['num_iterations']):
    history.append(neuro.train(X_train, y_train, X_val, y_val))
print("Training History with Multiple Iterations:", history)

# Evaluate model with multiple iterations
accuracy = []
for i in range(config['num_iterations']):
    accuracy.append(neuro.evaluate(X_test, y_test))
print("Test Accuracy with Multiple Iterations:", accuracy)
