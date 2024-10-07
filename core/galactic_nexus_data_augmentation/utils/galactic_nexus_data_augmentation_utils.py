import numpy as np
from PIL import Image
from sklearn.preprocessing import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

def image_rotation(image_data, rotation_range):
    datagen = ImageDataGenerator(rotation_range=rotation_range)
    rotated_images = datagen.flow(image_data, batch_size=32)
    return rotated_images

def image_flipping(image_data):
    datagen = ImageDataGenerator(horizontal_flip=True)
    flipped_images = datagen.flow(image_data, batch_size=32)
    return flipped_images

def image_zooming(image_data, zoom_range):
    datagen = ImageDataGenerator(zoom_range=zoom_range)
    zoomed_images = datagen.flow(image_data, batch_size=32)
    return zoomed_images

def text_tokenization(text_data, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    return sequences

def text_padding(sequences, maxlen):
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def text_augmentation(text_data, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectors = vectorizer.fit_transform(text_data)
    return tfidf_vectors

def audio_resampling(audio_data, rate):
    resampled_audio = []
    for audio in audio_data:
        rate, data = wavfile.read(audio)
        resampled_data = resample(data, rate)
        resampled_audio.append(resampled_data)
    return resampled_audio

def audio_feature_extraction(resampled_audio):
    mfcc_features = []
    for audio in resampled_audio:
        mfcc = librosa.feature.mfcc(audio)
        mfcc_features.append(mfcc)
    return mfcc_features
