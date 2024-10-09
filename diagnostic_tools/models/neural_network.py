import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(output_shape)
        ])

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def save_model(self, filename):
        self.model.save(filename)

# Example usage:
input_shape = (10,)
output_shape = 1
neural_network = NeuralNetwork(input_shape, output_shape)
neural_network.compile_model(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

X = np.random.rand(100, 10)
y = np.random.rand(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

neural_network.train_model(X_train, y_train, epochs=10, batch_size=32)
loss, accuracy = neural_network.evaluate_model(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

neural_network.save_model("neural_network.h5")
