import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score

class SentimentAnalysisModel:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def train(self, X, y):
        inputs = self.tokenizer(X, return_tensors='pt', max_length=100, padding='max_length', truncation=True)
        labels = torch.tensor(y)
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        print(f"Sentiment Analysis Model Loss: {loss:.2f}")

    def predict(self, X):
        inputs = self.tokenizer(X, return_tensors='pt', max_length=100, padding='max_length', truncation=True)
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return torch.argmax(outputs.logits, dim=1)
