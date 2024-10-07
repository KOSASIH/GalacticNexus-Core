# Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Define the machine learning functions
def train_model(data):
  model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(data, epochs=100)
  return model

def make_prediction(model, data):
  prediction = model.predict(data)
  return prediction

# Export the machine learning functions
def machine_learning():
  return {'train_model': train_model, 'make_prediction': make_prediction}
