# Logistic Regression Model f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import clean_text, load_datasets, load_profanity_list, count_profanity, get_profanity_score

# File paths
ENGLISH_PATH = 'data/davidson_english.csv'
HINDI_PATH = 'data/hasoc_hindi.csv'
HINGLISH_PATH = 'data/heot_hinglish.csv'
PROFANITY_PATH = 'data/Hinglish_Profanity_List.csv'

print("="*60)
print("LOGISTIC REGRESSION - MULTILINGUAL HATE SPEECH DETECTION")
print("="*60)

# Load data
df = load_datasets(ENGLISH_PATH, HINDI_PATH, HINGLISH_PATH)
profanity_words, profanity_df = load_profanity_list(PROFANITY_PATH)

# Add profanity features
print("\nExtracting profanity features...")
df['profanity_count'] = df['text'].apply(lambda x: count_profanity(x, profanity_words))
df['profanity_score'] = df['text'].apply(lambda x: get_profanity_score(x, profanity_df))

print(f"\nProfanity Analysis:")
print(f"  Non-Hate avg: {df[df['label']==0]['profanity_count'].mean():.3f}")
print(f"  Hate avg:     {df[df['label']==1]['profanity_count'].mean():.3f}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

train_idx = X_train.index
test_idx = X_test.index
prof_train = df.loc[train_idx, ['profanity_count', 'profanity_score']].values
prof_test = df.loc[test_idx, ['profanity_count', 'profanity_score']].values

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# TF-IDF Vectorization
print("\nVectorizing with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Combine TF-IDF + Profanity features
X_train_combined = hstack([X_train_tfidf, prof_train])
X_test_combined = hstack([X_test_tfidf, prof_test])

print(f"Features: {X_train_combined.shape[1]} (TF-IDF + Profanity)")

# Train Model
print("\nTraining Logistic Regression...")
model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
model.fit(X_train_combined, y_train)
print("Training complete!")

# Evaluate
y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            annot_kws={'size': 14})
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('Actual Label', fontsize=12)

# Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = axes[1].bar(metrics, values, color=colors)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val*100:.2f}%', ha='center', fontsize=11, fontweight='bold')

# Prediction Distribution
pred_counts = pd.Series(y_pred).value_counts().sort_index()
actual_counts = y_test.value_counts().sort_index()
x = np.arange(2)
width = 0.35
axes[2].bar(x - width/2, actual_counts.values, width, label='Actual', color='#3498db')
axes[2].bar(x + width/2, pred_counts.values, width, label='Predicted', color='#e74c3c')
axes[2].set_xticks(x)
axes[2].set_xticklabels(['Non-Hate', 'Hate'])
axes[2].set_title('Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=12)
axes[2].legend()

plt.tight_layout()
plt.savefig('results/lr_results.png', dpi=200, bbox_inches='tight')
plt.show()

# Save results to file
results = f"""LOGISTIC REGRESSION RESULTS
{'='*40}

Dataset: {len(df)} samples
  - English:  {len(df[df['language']=='english'])}
  - Hindi:    {len(df[df['language']=='hindi'])}
  - Hinglish: {len(df[df['language']=='hinglish'])}

Features: TF-IDF (5000) + Profanity (2)
Train: {len(X_train)} | Test: {len(X_test)}

Results:
  Accuracy:  {accuracy*100:.2f}%
  Precision: {precision*100:.2f}%
  Recall:    {recall*100:.2f}%
  F1-Score:  {f1*100:.2f}%

Confusion Matrix:
  TN={cm[0][0]}, FP={cm[0][1]}
  FN={cm[1][0]}, TP={cm[1][1]}
"""

with open('results/lr_results.txt', 'w') as f:
    f.write(results)


