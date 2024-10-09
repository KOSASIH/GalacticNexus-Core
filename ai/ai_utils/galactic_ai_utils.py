import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path, delimiter=','):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        delimiter (str, optional): Delimiter used in the CSV file. Defaults to ','.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path, delimiter=delimiter)

def preprocess_data(data, target_column, scaler=StandardScaler()):
    """
    Preprocess data by scaling features and splitting into training and testing sets.

    Args:
        data (pd.DataFrame): Data to be preprocessed.
        target_column (str): Name of the target column.
        scaler (object, optional): Scaler to use for feature scaling. Defaults to StandardScaler.

    Returns:
        tuple: Preprocessed data, including training and testing sets.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def create_data_loader(X, y, batch_size=32, shuffle=True):
    """
    Create a data loader from the given data.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        batch_size (int, optional): Batch size for the data loader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: Created data loader.
    """
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def train_model(model, device, data_loader, optimizer, criterion, epochs=5):
    """
    Train a model using the given data loader and optimizer.

    Args:
        model (nn.Module): Model to be trained.
        device (torch.device): Device to use for training.
        data_loader (DataLoader): Data loader for training.
        optimizer (optim.Optimizer): Optimizer to use for training.
        criterion (nn.Module): Loss function to use for training.
        epochs (int, optional): Number of epochs to train for. Defaults to 5.
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

def evaluate_model(model, device, data_loader, criterion):
    """
    Evaluate a model using the given data loader and loss function.

    Args:
        model (nn.Module): Model to be evaluated.
        device (torch.device): Device to use for evaluation.
        data_loader (DataLoader): Data loader for evaluation.
        criterion (nn.Module): Loss function to use for evaluation.

    Returns:
        tuple: Evaluation metrics, including accuracy, classification report, and confusion matrix.
    """
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    matrix = confusion_matrix(labels, predictions)

    return accuracy, report, matrix
