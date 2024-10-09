import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GraphNorm
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GalacticGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GalacticGNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norm_layers.append(GraphNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = self.norm_layers[i](x)
            x = self.dropout_layers[i](x)
            x = torch.relu(x)

        x = self.fc(x)
        return x

class GalacticGNNTrainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f'Epoch Loss: {total_loss / len(data_loader)}')

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()
                _, predicted = torch.max(out, dim=1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(batch.y.cpu().numpy())

        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        matrix = confusion_matrix(labels, predictions)

        print(f'Test Loss: {total_loss / len(data_loader)}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Classification Report:\n{report}')
        print(f'Confusion Matrix:\n{matrix}')

if __name__ == '__main__':
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and data loader
    num_nodes = 100
    num_edges = 200
    num_features = 10
    num_classes = 5
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, num_classes, (num_nodes,))
    data = Data(x=x, edge_index=edge_index, y=y)
    batch_size = 32
    data_loader = DataLoader([data], batch_size=batch_size, shuffle=True)

    # Create model, optimizer, and criterion
    input_dim = num_features
    hidden_dim = 128
    output_dim = num_classes
    num_layers = 3
    dropout = 0.1
    galactic_gnn = GalacticGNN(input_dim, hidden_dim, output_dim, num_layers, dropout)
    optimizer = optim.Adam(galactic_gnn.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train model
    num_epochs = 5
    trainer = GalacticGNNTrainer(galactic_gnn, device, optimizer, criterion)
    for epoch in range(num_epochs):
        trainer.train(data_loader)

    # Evaluate model
    trainer.evaluate(data_loader)
