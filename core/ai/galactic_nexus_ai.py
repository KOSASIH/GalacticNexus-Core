import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from galactic_nexus_utils import *

class GalacticNexusAI:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('galactic_nexus_tokenizer')
        self.model = self.build_model()

    def build_model(self):
        # Multi-modal Input Layers
        text_input = Input(shape=(self.config['max_text_length'],), name='text_input')
        image_input = Input(shape=(self.config['image_size'], self.config['image_size'], 3), name='image_input')
        audio_input = Input(shape=(self.config['audio_length'],), name='audio_input')

        # Text Encoder
        text_encoder = AutoModelForSequenceClassification.from_pretrained('galactic_nexus_text_encoder')
        text_output = text_encoder(text_input)

        # Image Encoder
        image_encoder = Conv2D(32, (3, 3), activation='relu', input_shape=(self.config['image_size'], self.config['image_size'], 3))
        image_output = image_encoder(image_input)

        # Audio Encoder
        audio_encoder = LSTM(128, return_sequences=True, input_shape=(self.config['audio_length'],))
        audio_output = audio_encoder(audio_input)

        # Fusion Layer
        fusion_output = concatenate([text_output, image_output, audio_output])

        # Dense Layers
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(fusion_output)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = Dropout(0.2)(x)

        # Output Layer
        output = Dense(self.config['num_classes'], activation='softmax')(x)

        # Compile Model
        model = Model(inputs=[text_input, image_input, audio_input], outputs=output)
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

    def explain(self, X_explain):
        # Model Interpretability using SHAP values
        explainer = shap.KernelExplainer(self.model.predict, X_explain)
        shap_values = explainer.shap_values(X_explain)
        return shap_values
