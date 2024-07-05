import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkTrainer:
    def __init__(self):
        self.model = NeuralNetwork()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data: pd.DataFrame) -> None:
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, data['target'])
            loss.backward()
            self.optimizer.step()
