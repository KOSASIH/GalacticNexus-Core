import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

class NeuralInterfaceModel:
  def __init__(self, config):
    self.config = config
    self.build_model()

  def build_model(self):
    # Define input layer
    input_layer = Input(shape=self.config["input_shape"])

    # Define convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    # Define flatten layer
    x = Flatten()(x)

    # Define dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(self.config["output_shape"][0], activation='softmax')(x)

    # Define model
    self.model = Model(inputs=input_layer, outputs=x)

    # Compile model
    self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    model_checkpoint = ModelCheckpoint(self.config["model_path"], monitor='val_loss', save_best_only=True, mode='min')

    # Define scaler
    self.scaler = StandardScaler()

  def train(self, X_train, y_train, X_val, y_val):
    # Scale data
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_val_scaled = self.scaler.transform(X_val)

    # Train model
    history = self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping, model_checkpoint])

    return history

  def evaluate(self, X_test, y_test):
    # Scale data
    X_test_scaled = self.scaler.transform(X_test)

    # Evaluate model
    loss, accuracy = self.model.evaluate(X_test_scaled, y_test)

    return loss, accuracy

  def predict(self, X):
    # Scale data
    X_scaled = self.scaler.transform(X)

    # Predict output
    output = self.model.predict(X_scaled)

    return output
