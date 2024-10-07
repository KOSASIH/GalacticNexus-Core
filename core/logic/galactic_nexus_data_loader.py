import numpy as np
import pandas as pd
from galactic_nexus_utils import *

class GalacticNexusDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = load_data(self.file_path)
        return data

    def preprocess_data(self, data):
        text_features, image_features, audio_features, label_features = preprocess_data(data)
        return text_features, image_features, audio_features, label_features

    def split_data(self, data):
        X_train, X_test, y_train, y_test = split_data(data)
        return X_train, X_test, y_train, y_test

    def create_data_generator(self, X_train, y_train):
        generator = create_data_generator(X_train, y_train)
        return generator
