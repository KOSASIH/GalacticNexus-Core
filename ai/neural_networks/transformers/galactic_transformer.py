import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GalacticTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout):
        super(GalacticTransformer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class GalacticTransformerDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_len):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_galactic_transformer(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch Loss: {total_loss / len(train_loader)}')

def evaluate_galactic_transformer(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, label)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(label.cpu().numpy())

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

    # Load pre-trained model and tokenizer
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataset and data loader
    data = ['This is a sample text.', 'This is another sample text.']
    labels = [0, 1]
    dataset = GalacticTransformerDataset(data, labels, tokenizer, max_len=512)
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model, optimizer, and criterion
    input_dim = 768
    hidden_dim = 128
    output_dim = 8
    num_heads = 8
    dropout = 0.1
    galactic_transformer = GalacticTransformer(input_dim, hidden_dim, output_dim, num_heads, dropout)
    optimizer = optim.Adam(galactic_transformer.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train model
    num_epochs = 5
    for epoch in range(num_epochs):
        train_galactic_transformer(galactic_transformer, device, train_loader, optimizer, criterion)

    # Evaluate model
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate_galactic_transformer(galactic_transformer, device, test_loader, criterion)
