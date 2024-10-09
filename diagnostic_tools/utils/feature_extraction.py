import pandas as pd
from sklearn.decomposition import PCA

def extract_features(data):
    # Extract features from spacecraft data
    pca = PCA(n_components=10)
    features = pca.fit_transform(data)
    return features
