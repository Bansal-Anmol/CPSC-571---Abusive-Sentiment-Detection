# mBERT Model for Multilingual Hate Speech Detection

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import clean_text, load_datasets

# File paths
ENGLISH_PATH = 'data/davidson_english.csv'
HINDI_PATH = 'data/hasoc_hindi.csv'
HINGLISH_PATH = 'data/heot_hinglish.csv'

# Hyperparameters
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("=" * 50)
print("mBERT - MULTILINGUAL HATE SPEECH DETECTION")
print("=" * 50)

# Load data
df = load_datasets(ENGLISH_PATH, HINDI_PATH, HINGLISH_PATH)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded")


# Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Create datasets and dataloaders
train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_LENGTH)
test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))

# Load model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
print("Model loaded")

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training
print("\n" + "=" * 50)
print("TRAINING STARTED")
print("=" * 50)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} done. Average Loss: {avg_loss:.4f}\n")

print("Training complete!")

# Evaluation
print("\n" + "=" * 50)
print("EVALUATING MODEL")
print("=" * 50)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            annot_kws={'size': 14})
axes[0].set_title('Confusion Matrix - mBERT', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('Actual Label', fontsize=12)

# Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = axes[1].bar(metrics, values, color=colors)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('mBERT Performance Metrics', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val*100:.2f}%', ha='center', fontsize=11, fontweight='bold')

# Prediction Distribution
pred_counts = pd.Series(all_preds).value_counts().sort_index()
actual_counts = pd.Series(all_labels).value_counts().sort_index()
x = np.arange(2)
width = 0.35
axes[2].bar(x - width/2, actual_counts.values, width, label='Actual', color='#3498db')
axes[2].bar(x + width/2, pred_counts.values, width, label='Predicted', color='#2ecc71')
axes[2].set_xticks(x)
axes[2].set_xticklabels(['Non-Hate', 'Hate'])
axes[2].set_title('Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=12)
axes[2].legend()

plt.tight_layout()
plt.savefig('results/mbert_results.png', dpi=200, bbox_inches='tight')
plt.show()

# Save results to file
results = f"""mBERT RESULTS
====================

Model: {MODEL_NAME}
Max Length: {MAX_LENGTH}
Batch Size: {BATCH_SIZE}
Epochs: {EPOCHS}
Learning Rate: {LEARNING_RATE}

Dataset: {len(df)} samples
- English: {len(df[df['language']=='english'])}
- Hindi: {len(df[df['language']=='hindi'])}
- Hinglish: {len(df[df['language']=='hinglish'])}

Train: {len(X_train)}
Test: {len(X_test)}

Results:
Accuracy:  {accuracy*100:.2f}%
Precision: {precision*100:.2f}%
Recall:    {recall*100:.2f}%
F1-Score:  {f1*100:.2f}%

Confusion Matrix:
TN={cm[0][0]}, FP={cm[0][1]}
FN={cm[1][0]}, TP={cm[1][1]}
"""

with open('results/mbert_results.txt', 'w') as f:
    f.write(results)

print("\nResults saved to results/mbert_results.txt")
print("Plot saved to results/mbert_results.png")
