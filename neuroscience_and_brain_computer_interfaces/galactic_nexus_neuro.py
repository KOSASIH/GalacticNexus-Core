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

class GalacticNexusNeuro:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

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

        history = self.model.fit(X_train, y_train, 
                                 epochs=self.config['epochs'], 
                                 batch_size=self.config['batch_size'], 
                                 validation_data=(X_val, y_val), 
                                 callbacks=[early_stopping, reduce_lr])

        return history

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_class)
        return accuracy

    def neuro_feedback(self, X_unlabeled):
        predictions = self.model.predict(X_unlabeled)
        uncertainties = np.max(predictions, axis=1)
        return uncertainties

    def neuro_control(self, X_unlabeled):
        predictions = self.model.predict(X_unlabeled)
        controls = np.argmax(predictions, axis=1)
        return controls

    def neuro_learning(self, X_train, y_train, X_unlabeled):
        for i in range(self.config['num_iterations']):
            X_selected = self.neuro_feedback(X_unlabeled)
            y_selected = self.model.predict(X_selected)
            X_train = np.concatenate((X_train, X_selected))
            y_train = np.concatenate((y_train, y_selected))
            self.model.fit(X_train, y_train, epochs=self.config['epochs'], batch_size=self.config['batch_size'])

        return X_train, y_train
