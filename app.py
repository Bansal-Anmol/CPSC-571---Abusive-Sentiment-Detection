# Hate Speech Detector - Streamlit App

import streamlit as st
import pickle
import re
from scipy.sparse import hstack

# page config
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="centered"
)

# custom css for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .hate {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .not-hate {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .stats-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# preprocessing function
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

# load models
@st.cache_resource
def load_models():
    model = pickle.load(open('models/lr_model.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    profanity_words = pickle.load(open('models/profanity_words.pkl', 'rb'))
    profanity_df = pickle.load(open('models/profanity_df.pkl', 'rb'))
    return model, tfidf, profanity_words, profanity_df

# profanity functions
def count_profanity(text, profanity_words):
    words = str(text).lower().split()
    return sum(1 for word in words if word in profanity_words)

def get_profanity_score(text, profanity_df):
    words = str(text).lower().split()
    total = 0
    for word in words:
        match = profanity_df[profanity_df['word'].str.lower() == word]
        if len(match) > 0:
            total += match['score'].values[0]
    return total

# header
st.markdown("<h1 class='main-title'>🛡️ Hate Speech Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Multilingual detection for English, Hindi, and Hinglish</p>", unsafe_allow_html=True)

# text input
text_input = st.text_area(
    "Enter text to analyze:",
    height=120,
    placeholder="Type or paste text here..."
)

# example buttons
st.write("**Try these examples:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("I love this! 💚"):
        text_input = "I love this so much"
        st.rerun()

with col2:
    if st.button("You're an idiot 😠"):
        text_input = "You're such an idiot"
        st.rerun()

with col3:
    if st.button("तू बेवकूफ है"):
        text_input = "तू बेवकूफ है"
        st.rerun()

# analyze button
st.write("")
analyze = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

# prediction
if analyze:
    if text_input.strip():
        try:
            model, tfidf, profanity_words, profanity_df = load_models()
            
            # preprocess and predict
            clean = clean_text(text_input)
            text_vec = tfidf.transform([clean])
            prof_count = count_profanity(text_input, profanity_words)
            prof_score = get_profanity_score(text_input, profanity_df)
            features = hstack([text_vec, [[prof_count, prof_score]]])
            
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = max(proba) * 100
            
            # display result
            st.write("")
            st.write("---")
            st.write("")
            
            if pred == 1:
                st.markdown("""
                <div class='result-box hate'>
                    <h2>🚨 HATE SPEECH DETECTED</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-box not-hate'>
                    <h2>✅ NOT HATE SPEECH</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # confidence bar
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.progress(confidence / 100)
            
            # details
            with st.expander("See details"):
                st.write(f"**Original text:** {text_input}")
                st.write(f"**Cleaned text:** {clean}")
                st.write(f"**Profanity count:** {prof_count}")
                st.write(f"**Profanity score:** {prof_score}")
                st.write(f"**Model:** Logistic Regression with TF-IDF")
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("Please enter some text to analyze")

# footer with stats
st.write("")
st.write("---")
st.markdown("""
<div class='stats-box'>
    <b>📊 Model Stats:</b> Trained on 37,302 samples | 3 Languages | 88.47% Accuracy
    <br><br>
    <small>CPSC 571 - University of Calgary</small>
</div>
""", unsafe_allow_html=True)
