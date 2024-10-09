import tensorflow as tf
from tensorflow import keras

class SpacecraftSystemNN(keras.Model):
    def __init__(self):
        super(SpacecraftSystemNN, self).__init__()
        self.input_layer = keras.layers.InputLayer(input_shape=(10,))
        self.hidden_layer1 = keras.layers.Dense(64, activation="relu")
        self.hidden_layer2 = keras.layers.Dense(32, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        outputs = self.output_layer(x)
        return outputs
