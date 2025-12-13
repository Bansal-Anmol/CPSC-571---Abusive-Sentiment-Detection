"""
Shared preprocessing for all models
"""
import re
import pandas as pd

def clean_text(text):
    """Standard text preprocessing for hate speech detection"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)      # Remove URLs
    text = re.sub(r'@\w+', '', text)                          # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)                     # Keep hashtag words
    text = re.sub(r'rt\s+', '', text)                         # Remove RT
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)          # Remove punctuation, keep Devanagari
    text = re.sub(r'\d+', '', text)                           # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()                  # Normalize whitespace
    return text

def load_datasets(english_path, hindi_path, hinglish_path):
    """Load and combine all datasets with standardized labels"""
    all_data = []
    
    # English - Davidson
    eng_df = pd.read_csv(english_path)
    eng_df = eng_df.rename(columns={'tweet': 'text'})
    eng_df['label'] = eng_df['class'].apply(lambda x: 0 if x == 2 else 1)
    eng_df['language'] = 'english'
    eng_df = eng_df[['text', 'label', 'language']]
    all_data.append(eng_df)
    
    # Hindi - HASOC
    hindi_df = pd.read_csv(hindi_path)
    hindi_df['label'] = hindi_df['hate'].astype(int)
    hindi_df['language'] = 'hindi'
    hindi_df = hindi_df[['text', 'label', 'language']]
    all_data.append(hindi_df)
    
    # Hinglish - HEOT
    hing_df = pd.read_csv(hinglish_path)
    hing_df['label'] = hing_df['hate'].apply(lambda x: 0 if x == 0 else 1)
    hing_df['language'] = 'hinglish'
    hing_df = hing_df[['text', 'label', 'language']]
    all_data.append(hing_df)
    
    # Combine
    df = pd.concat(all_data, ignore_index=True)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df['clean_text'] = df['text'].apply(clean_text)
    
    return df

def load_profanity_list(path):
    """Load profanity word list"""
    profanity_df = pd.read_csv(path)
    profanity_words = set(profanity_df['word'].str.lower().dropna().tolist())
    return profanity_words, profanity_df
