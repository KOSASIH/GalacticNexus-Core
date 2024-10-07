import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from galactic_nexus_utils import *

class GalacticNexusTransferLearning:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('galactic_nexus_tokenizer')
        self.model = self.build_model()

    def build_model(self):
        # Load pre-trained model
        pre_trained_model = AutoModelForSequenceClassification.from_pretrained('galactic_nexus_pre_trained_model')

        # Freeze pre-trained model layers
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # Add custom layers
        input_layer = Input(shape=(self.config['max_text_length'],), name='input_layer')
        x = pre_trained_model(input_layer)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(self.config['num_classes'], activation='softmax')(x)

        # Compile model
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

    def fine_tune(self, X_train, y_train, X_val, y_val):
        # Unfreeze pre-trained model layers
        for layer in self.model.layers:
            layer.trainable = True

        # Compile model
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        history = self.model.fit(X_train, y_train, 
                                 epochs=self.config['epochs'], 
                                 batch_size=self.config['batch_size'], 
                                 validation_data=(X_val, y_val), 
                                 callbacks=[early_stopping, reduce_lr])

        return history
