import numpy as np
import pandas as pd
from galactic_nexus_utils import *

class GalacticNexusDataPreprocessor:
    def __init__(self, data):
        self.data = data

    def tokenize_text(self):
        sequences = tokenize_text(self.data['text'])
        return sequences

    def pad_sequences(self, sequences):
        padded_sequences = pad_sequences(sequences)
        return padded_sequences

    def load_image(self, image_path):
        img_array = load_image(image_path)
        return img_array

    def extract_audio_features(self, audio_data):
        audio_features = extract_audio_features(audio_data)
        return audio_features
