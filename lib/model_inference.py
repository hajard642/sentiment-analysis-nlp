import json
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    """Sentiment classification model using DistilBERT"""
    def __init__(self, num_classes=2, dropout=0.1, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        dropped = self.dropout(pooled)
        logits = self.linear(dropped)
        return logits


class SentimentAnalyzer:
    """Inference wrapper for sentiment analysis"""
    
    def __init__(self, model_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load config
        with open(self.model_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load tokenizer
        with open(self.model_dir / 'tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load model
        self.model = SentimentClassifier(
            num_classes=2,
            model_name=self.metadata['config']['model_name']
        )
        
        # Find best checkpoint
        checkpoints = list(self.model_dir.glob('model_best_epoch*.pt'))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, 
                                      key=lambda x: int(x.stem.split('epoch')[1]))[-1]
            self.model.load_state_dict(torch.load(latest_checkpoint, 
                                                  map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str):
        """Predict sentiment for a given text"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'sentiment': 'positive' if pred_class == 1 else 'negative',
            'confidence': confidence,
            'probabilities': {
                'negative': probs[0, 0].item(),
                'positive': probs[0, 1].item()
            }
        }
    
    def get_metadata(self):
        """Return model metadata"""
        return self.metadata
