# Preprocessing Module for Multilingual Hate Speech Detection

import re
import pandas as pd


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)      # remove urls
    text = re.sub(r'@\w+', '', text)                          # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)                     # keep hashtag words
    text = re.sub(r'rt\s+', '', text)                         # remove RT
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)          # remove punctuation, keep devanagari
    text = re.sub(r'\d+', '', text)                           # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()                  # normalize whitespace
    return text


def load_datasets(english_path, hindi_path, hinglish_path):
    all_data = []
    
    # English - Davidson
    print("[1/3] Loading English (Davidson)...")
    eng_df = pd.read_csv(english_path)
    eng_df = eng_df.rename(columns={'tweet': 'text'})
    eng_df['label'] = eng_df['class'].apply(lambda x: 0 if x == 2 else 1)
    eng_df['language'] = 'english'
    eng_df = eng_df[['text', 'label', 'language']]
    all_data.append(eng_df)
    print(f"   English: {len(eng_df)} samples")
    
    # Hindi - HASOC
    print("[2/3] Loading Hindi (HASOC)...")
    hindi_df = pd.read_csv(hindi_path)
    hindi_df['label'] = hindi_df['hate'].astype(int)
    hindi_df['language'] = 'hindi'
    hindi_df = hindi_df[['text', 'label', 'language']]
    all_data.append(hindi_df)
    print(f"   Hindi: {len(hindi_df)} samples")
    
    # Hinglish - HEOT
    print("[3/3] Loading Hinglish (HEOT)...")
    hing_df = pd.read_csv(hinglish_path)
    hing_df['label'] = hing_df['hate'].apply(lambda x: 0 if x == 0 else 1)
    hing_df['language'] = 'hinglish'
    hing_df = hing_df[['text', 'label', 'language']]
    all_data.append(hing_df)
    print(f"   Hinglish: {len(hing_df)} samples")
    
    # Combine all
    df = pd.concat(all_data, ignore_index=True)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df['clean_text'] = df['text'].apply(clean_text)
    
    print(f"\nTotal: {len(df)} samples")
    return df


def load_profanity_list(path):
    profanity_df = pd.read_csv(path)
    profanity_words = set(profanity_df['word'].str.lower().dropna().tolist())
    print(f"Loaded {len(profanity_words)} profanity words")
    return profanity_words, profanity_df


def count_profanity(text, profanity_words):
    text = str(text).lower()
    words = text.split()
    return sum(1 for word in words if word in profanity_words)


def get_profanity_score(text, profanity_df):
    text = str(text).lower()
    words = text.split()
    total_score = 0
    for word in words:
        match = profanity_df[profanity_df['word'].str.lower() == word]
        if len(match) > 0:
            total_score += match['score'].values[0]
    return total_score
