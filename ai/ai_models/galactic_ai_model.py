import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GalacticAIModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GalacticAIModel, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.fc_layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.fc_layers) - 1):
            x = torch.relu(self.fc_layers[i](x))
            x = self.dropout_layers[i](x)
        x = self.fc_layers[-1](x)
        return x

class GalacticAIDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def train_galactic_ai_model(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch Loss: {total_loss / len(train_loader)}')

def evaluate_galactic_ai_model(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    matrix = confusion_matrix(labels, predictions)

    print(f'Test Loss: {total_loss / len(test_loader)}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{matrix}')

if __name__ == '__main__':
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and data loader
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = GalacticAIDataset(data, labels)
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model, optimizer, and criterion
    input_dim = 10
    hidden_dim = 128
    output_dim = 2
    num_layers = 3
    dropout = 0.1
    galactic_ai_model = GalacticAIModel(input_dim, hidden_dim, output_dim, num_layers, dropout)
    optimizer = optim.Adam(galactic_ai_model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train model
    num_epochs = 5
    for epoch in range(num_epochs):
        train_galactic_ai_model(galactic_ai_model, device, data_loader, optimizer, criterion)

    # Evaluate model
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate_galactic_ai_model(galactic_ai_model, device, test_loader, criterion)
