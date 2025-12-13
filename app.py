# Simple Hate Speech Detector

import streamlit as st
import pickle
import re
from scipy.sparse import hstack

# load models
@st.cache_resource
def load_model():
    model = pickle.load(open('models/lr_model.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    profanity_words = pickle.load(open('models/profanity_words.pkl', 'rb'))
    profanity_df = pickle.load(open('models/profanity_df.pkl', 'rb'))
    return model, tfidf, profanity_words, profanity_df

# clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# count profanity
def count_profanity(text, profanity_words):
    words = str(text).lower().split()
    return sum(1 for w in words if w in profanity_words)

# get profanity score
def get_profanity_score(text, profanity_df):
    words = str(text).lower().split()
    score = 0
    for w in words:
        match = profanity_df[profanity_df['word'].str.lower() == w]
        if len(match) > 0:
            score += match['score'].values[0]
    return score

# predict
def predict(text, model, tfidf, profanity_words, profanity_df):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    prof_count = count_profanity(text, profanity_words)
    prof_score = get_profanity_score(text, profanity_df)
    features = hstack([vec, [[prof_count, prof_score]]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return pred, proba


# app
st.title("Hate Speech Detector")
st.write("Detects hate speech in English, Hindi, and Hinglish")

text = st.text_area("Enter text:", height=100)

if st.button("Analyze"):
    if text.strip():
        model, tfidf, profanity_words, profanity_df = load_model()
        pred, proba = predict(text, model, tfidf, profanity_words, profanity_df)
        
        confidence = max(proba) * 100
        
        if pred == 1:
            st.error(f"HATE SPEECH DETECTED")
        else:
            st.success(f"NOT HATE SPEECH")
        
        st.write(f"Confidence: {confidence:.1f}%")
    else:
        st.warning("Please enter some text")
