import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def engineer_features(df):
    vectorizer = TfidfVectorizer()
    df["text"] = vectorizer.fit_transform(df["text"].values)
    return df
