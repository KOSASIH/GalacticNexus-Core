import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from galactic_nexus_utils import *

class GalacticNexusFederatedLearning:
    def __init__(self, config):
        self.config = config
        self.models = self.build_models()

    def build_models(self):
        models = []
        for i in range(self.config['num_clients']):
            model = self.build_model()
            models.append(model)
        return models

    def build_model(self):
        input_layer = Input(shape=(self.config['input_shape'],), name='input_layer')
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(self.config['num_classes'], activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, X_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        histories = []
        for model in self.models:
            history = model.fit(X_train, y_train, 
                                 epochs=self.config['epochs'], 
                                 batch_size=self.config['batch_size'], 
                                 validation_data=(X_val, y_val), 
                                 callbacks=[early_stopping, reduce_lr])

            histories.append(history)

        return histories

    def evaluate(self, X_test, y_test):
        predictions = []
        for model in self.models:
            y_pred = model.predict(X_test)
            predictions.append(y_pred)

        ensemble_prediction = np.mean(predictions, axis=0)
        ensemble_prediction_class = np.argmax(ensemble_prediction, axis=1)
        accuracy = accuracy_score(y_test, ensemble_prediction_class)
        return accuracy

    def federated_learning(self, X_train, y_train, X_val, y_val):
        for i in range(self.config['num_rounds']):
            for j in range(self.config['num_clients']):
                X_train_client = X_train[j]
                y_train_client = y_train[j]
                self.models[j].fit(X_train_client, y_train_client, epochs=self.config['epochs'], batch_size=self.config['batch_size'])

            ensemble_prediction = []
            for model in self.models:
                y_pred = model.predict(X_val)
                ensemble_prediction.append(y_pred)

            ensemble_prediction = np.mean(ensemble_prediction, axis=0)
            ensemble_prediction_class = np.argmax(ensemble_prediction, axis=1)
            accuracy = accuracy_score(y_val, ensemble_prediction_class)
            print("Round", i, "Accuracy:", accuracy)

        return accuracy
