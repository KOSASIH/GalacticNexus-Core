import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1_l2

def load_pre_trained_model(model_name):
    pre_trained_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pre_trained_model

def freeze_pre_trained_model_layers(pre_trained_model):
    for layer in pre_trained_model.layers:
        layer.trainable = False
    return pre_trained_model

def add_custom_layers(pre_trained_model, input_layer, num_classes):
    x = pre_trained_model(input_layer)
    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return output_layer

def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def fine_tune_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, 
                         epochs=epochs, 
                         batch_size=batch_size, 
                         validation_data=(X_val, y_val), 
                         callbacks=[early_stopping, reduce_lr])

    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_class)
    return accuracy
