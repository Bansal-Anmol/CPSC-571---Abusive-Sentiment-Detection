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

# custom css
st.markdown("""
<style>
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* input card */
    .input-card {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    /* example buttons */
    .example-btn {
        background: #2D2D2D;
        border: 1px solid #444;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #fff;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-btn:hover {
        background: #3D3D3D;
        border-color: #667eea;
    }
    
    /* result boxes */
    .result-hate {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-safe {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* confidence section */
    .confidence-container {
        background: #1E1E1E;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-label {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .confidence-value {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* stats footer */
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stat-label {
        color: #888;
        font-size: 0.8rem;
    }
    
    /* footer */
    .footer {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.85rem;
    }
    
    /* button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 25px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
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
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ Hate Speech Detector</div>
    <div class="header-subtitle">Multilingual detection for English, Hindi & Hinglish</div>
</div>
""", unsafe_allow_html=True)

# text input
text_input = st.text_area(
    "Enter text to analyze:",
    height=100,
    placeholder="Type or paste any text here..."
)

# example texts
st.markdown("**Quick Examples:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("👍 Positive"):
        st.session_state.example = "I love this community"
with col2:
    if st.button("😠 Offensive"):
        st.session_state.example = "You are such an idiot"
with col3:
    if st.button("🇮🇳 Hindi"):
        st.session_state.example = "तू बेवकूफ है"
with col4:
    if st.button("🔀 Hinglish"):
        st.session_state.example = "tu chutiya hai"

# check if example button was clicked
if 'example' in st.session_state:
    text_input = st.session_state.example
    del st.session_state.example
    st.rerun()

# analyze button
st.write("")
analyze = st.button("🔍 Analyze Text")

# prediction
if analyze and text_input.strip():
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
                <div class="result-text">NO HATE SPEECH</div>
            </div>
            """, unsafe_allow_html=True)
        
        # confidence display
        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-label">CONFIDENCE SCORE</div>
            <div class="confidence-value">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # progress bar
        st.progress(confidence / 100)
        
        # details expander
        with st.expander("📋 View Details"):
            st.write(f"**Original:** {text_input}")
            st.write(f"**Cleaned:** {clean}")
            st.write(f"**Profanity Count:** {prof_count}")
            st.write(f"**Profanity Score:** {prof_score}")
            
    except Exception as e:
        st.error(f"Error: {e}")

elif analyze:
    st.warning("Please enter some text to analyze")

# stats section
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <div class="stat-value">37,302</div>
        <div class="stat-label">Training Samples</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">3</div>
        <div class="stat-label">Languages</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">88.5%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">TF-IDF</div>
        <div class="stat-label">+ Profanity</div>
    </div>
</div>
""", unsafe_allow_html=True)

# footer
st.markdown("""
<div class="footer">
    CPSC 571 Final Project • University of Calgary
</div>
""", unsafe_allow_html=True)
