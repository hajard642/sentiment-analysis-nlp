import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import pickle

# ML libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# NLP libraries
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': SEED,
    'early_stopping_patience': 2,
    'early_stopping_delta': 0.001,
}

print(f"[INFO] Using device: {CONFIG['device']}")
print(f"[INFO] Random seeds set to {SEED} for reproducibility")


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier(nn.Module):
    """Sentiment classification model using DistilBERT"""
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CONFIG['model_name'])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        dropped = self.dropout(pooled)
        logits = self.linear(dropped)
        return logits


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic movie review data for demonstration"""
    print(f"\n[INFO] Generating {n_samples} synthetic movie reviews...")
    
    positive_words = ['excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 
                     'perfect', 'outstanding', 'superb', 'incredible', 'masterpiece']
    negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'bad',
                     'poor', 'dreadful', 'boring', 'waste', 'pathetic']
    neutral_words = ['movie', 'film', 'watched', 'saw', 'acting', 'plot', 
                    'characters', 'story', 'scenes', 'direction']
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        sentiment = np.random.choice([0, 1])  # 0: negative, 1: positive
        
        if sentiment == 1:
            review_words = np.random.choice(positive_words, size=np.random.randint(3, 6))
        else:
            review_words = np.random.choice(negative_words, size=np.random.randint(3, 6))
        
        neutral = np.random.choice(neutral_words, size=np.random.randint(2, 4))
        review = ' '.join(np.concatenate([review_words, neutral]))
        
        texts.append(review)
        labels.append(sentiment)
    
    return texts, labels


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels
    }


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("MOVIE REVIEW SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("="*60)
    
    # Create output directory
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    texts, labels = generate_synthetic_data(n_samples=1000)
    print(f"[INFO] Dataset: {len(texts)} samples")
    print(f"[INFO] Class distribution: {np.bincount(labels)}")
    
    # Step 2: Split data
    print("\n[STEP 2] Splitting data (train/val/test: 70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 3: Tokenize
    print("\n[STEP 3] Tokenizing texts...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Step 4: Initialize model
    print("\n[STEP 4] Initializing model...")
    model = SentimentClassifier(num_classes=2)
    model.to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Step 5: Training loop with early stopping
    print("\n[STEP 5] Training with early stopping (patience={})...".format(
        CONFIG['early_stopping_patience']
    ))
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'])
        val_metrics = validate(model, val_loader, CONFIG['device'])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"\n[Epoch {epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss - CONFIG['early_stopping_delta']:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            checkpoint_path = output_dir / f'model_best_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [BEST] Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n[INFO] Early stopping triggered (patience={CONFIG['early_stopping_patience']})")
                break
    
    # Step 6: Test evaluation
    print("\n[STEP 6] Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / f'model_best_epoch{epoch}.pt'))
    test_metrics = validate(model, test_loader, CONFIG['device'])
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
    print("\nConfusion Matrix:")
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    precision = precision_score(test_metrics['labels'], test_metrics['preds'])
    recall = recall_score(test_metrics['labels'], test_metrics['preds'])
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save metadata
    metadata = {
        'config': CONFIG,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_loss': float(test_metrics['loss']),
        'history': history,
        'data_split': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)}
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(output_dir / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
