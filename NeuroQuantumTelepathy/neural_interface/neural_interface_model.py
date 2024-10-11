import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

class NeuralInterfaceModel:
  def __init__(self, config):
    self.config = config
    self.build_model()

  def build_model(self):
    input_layer = Input(shape=self.config["input_shape"])
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(self.config["output_shape"][0], activation='softmax')(x)
    self.model = Model(inputs=input_layer, outputs=output_layer)
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    self.model.save(self.config["model_path"])
