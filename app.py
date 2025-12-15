# HateGuard - Multilingual Hate Speech Detection

import streamlit as st # web app
import pickle # loads model files
import re # text pattern matching
import torch # run mBERT 
from scipy.sparse import hstack # combine TF-IDF features with profanity features
from transformers import BertTokenizer, BertForSequenceClassification # loads BERT tokenizer and model

# page setup
st.set_page_config(
    page_title="HateGuard",
    page_icon="🛡️",
    layout="centered"
)

# styling - dark purple theme
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 700px;
    }
    
    .header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .logo {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .tagline {
        color: #8888aa;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea {
        background: #1e1e3f !important;
        border: 1px solid #333366 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stRadio > div {
        background: #1e1e3f;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333366;
    }
    .stRadio label {
        color: #ffffff !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 30px;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .result-hate {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.3);
    }
    
    .result-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3);
    }
    
    .result-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .result-text {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .confidence-box {
        background: #1e1e3f;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border: 1px solid #333366;
        margin: 1rem 0;
    }
    .confidence-label {
        color: #8888aa;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .confidence-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .footer {
        text-align: center;
        color: #555577;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        border-top: 1px solid #333366;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# cleaning the text - same function from preprocessing.py
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) #urls
    text = re.sub(r'@\w+', '', text) #mentions
    text = re.sub(r'#(\w+)', r'\1', text) #hashtags
    text = re.sub(r'rt\s+', '', text) #retweets
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text) # keep letters + hindi
    text = re.sub(r'\d+', '', text) # numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# load LR models and vectorizer from pickle files
@st.cache_resource
def load_lr_models():
    model = pickle.load(open('models/lr_model.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    profanity_words = pickle.load(open('models/profanity_words.pkl', 'rb'))
    profanity_df = pickle.load(open('models/profanity_df.pkl', 'rb'))
    return model, tfidf, profanity_words, profanity_df

# load mBERT model from HuggingFace - trained and uploaded there
@st.cache_resource
def load_mbert_model():
    model_name = "roshanp1923/mbert-hate-speech"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

# counts how many profanity words are in the text
def count_profanity(text, profanity_words):
    words = str(text).lower().split()
    count = 0
    for word in words:
        if word in profanity_words:
            count += 1
    return count

# gets the profanity scores based on our hinglish list
def get_profanity_score(text, profanity_df):
    words = str(text).lower().split()
    total = 0
    for word in words:
        match = profanity_df[profanity_df['word'].str.lower() == word]
        if len(match) > 0:
            total += match['score'].values[0]
    return total

# predicting with logistic regression
def predict_lr(text, model, tfidf, profanity_words, profanity_df):
    clean = clean_text(text)
    text_vec = tfidf.transform([clean])
    prof_count = count_profanity(text, profanity_words)
    prof_score = get_profanity_score(text, profanity_df)

    # combining tfidf with profanity features
    features = hstack([text_vec, [[prof_count, prof_score]]])
    
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba) * 100
    return pred, confidence

# predicting with mBERT
def predict_mbert(text, model, tokenizer):
    clean = clean_text(text)
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100
    
    return pred, confidence

# ---------- APP STARTS HERE ----------

# header
st.markdown("""
<div class="header">
    <div class="logo">🛡️ HateGuard</div>
    <div class="tagline">Multilingual Hate Speech Detection</div>
</div>
""", unsafe_allow_html=True)

# text input box
text_input = st.text_area(
    "Enter text to analyze:",
    height=100,
    placeholder="Type or paste text here..."
)

# choosing which model to use
st.write("")
model_choice = st.radio(
    "Select Model:",
    ["Logistic Regression", "mBERT"],
    horizontal=True
)

# button to run prediction
detect = st.button("🔍 Detect Hate Speech")

# when button is clicked
if detect:
    if text_input.strip():
        try:
            with st.spinner("Analyzing..."):
                if model_choice == "Logistic Regression":
                    model, tfidf, profanity_words, profanity_df = load_lr_models()
                    pred, confidence = predict_lr(text_input, model, tfidf, profanity_words, profanity_df)
                else:
                    model, tokenizer = load_mbert_model()
                    pred, confidence = predict_mbert(text_input, model, tokenizer)
            
            # result display
            if pred == 1:
                st.markdown("""
                <div class="result-hate">
                    <div class="result-icon">🚨</div>
                    <div class="result-text">HATE SPEECH DETECTED</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    <div class="result-icon">✅</div>
                    <div class="result-text">NOT HATE SPEECH</div>
                </div>
                """, unsafe_allow_html=True)
            
            # confidence
            st.markdown(f"""
            <div class="confidence-box">
                <div class="confidence-label">CONFIDENCE ({model_choice})</div>
                <div class="confidence-value">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(confidence / 100)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text")

# footer
st.markdown("""
<div class="footer">
    CPSC 571 Final Project • University of Calgary<br>
    <a href="https://github.com/Bansal-Anmol/CPSC-571---Abusive-Sentiment-Detection" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
