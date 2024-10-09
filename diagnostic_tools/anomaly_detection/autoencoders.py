import tensorflow as tf
from tensorflow import keras

class AnomalyDetector(keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = keras.layers.Dense(64, activation="relu")
        self.decoder = keras.layers.Dense(10, activation="sigmoid")

    def call(self, inputs):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs
