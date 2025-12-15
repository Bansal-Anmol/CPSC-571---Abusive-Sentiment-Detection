# HateGuard - Multilingual Hate Speech Detection

A machine learning project that detects hate speech in **English**, **Hindi**, and **Hinglish** (code-mixed) text using two different approaches: Logistic Regression and mBERT.

**CPSC 571 Final Project - University of Calgary**

---

## Project Overview

Social media platforms face challenges in moderating hate speech across multiple languages. This project addresses the problem by building a multilingual hate speech detection system that can identify offensive content in English, Hindi, and Hinglish.

We implemented and compared two approaches:
- Logistic Regression with TF-IDF and profanity features
- mBERT (Multilingual BERT) fine-tuned for hate speech classification

---

## Features

- Detects hate speech in 3 languages (English, Hindi, Hinglish)
- Two model options with different accuracy-speed tradeoffs
- Clean web interface built with Streamlit
- Real-time predictions with confidence scores

---

## Dataset

| Source | Language | Samples |
|--------|----------|---------|
| Davidson et al. | English | 24,783 |
| HASOC 2019 | Hindi | 9,330 |
| HEOT | Hinglish | 3,189 |
| **Total** | | **37,302** |

---

## Models

### 1. Logistic Regression
- TF-IDF vectorizer (5000 features)
- Hinglish profanity word count
- Hinglish profanity score
- **Accuracy: 88.47%**

### 2. mBERT (Multilingual BERT)
- Pre-trained `bert-base-multilingual-cased`
- Fine-tuned for 3 epochs
- **Accuracy: 94.21%**

| Metric | Logistic Regression | mBERT |
|--------|---------------------|-------|
| Accuracy | 88.47% | 94.21% |
| Precision | 90.85% | 96.12% |
| Recall | 93.68% | 96.06% |
| F1-Score | 92.24% | 96.09% |

---

## mBERT Model

The trained mBERT model is hosted on HuggingFace:

https://huggingface.co/roshanp1923/mbert-hate-speech

The app automatically downloads and loads the model from HuggingFace.

---

## Files

| File | Description |
|------|-------------|
| `CPSC571_Final_Notebook.ipynb` | Main notebook with all training code |
| `app.py` | Streamlit web application |
| `preprocessing.py` | Text cleaning functions |
| `logistic_regression.py` | LR model training code |
| `mbert_model.py` | mBERT model training code |
| `lr_model.pkl` | Trained Logistic Regression model |
| `profanity_df.pkl` | Profanity words with scores |
| `profanity_words.pkl` | List of Hinglish profanity words |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (5000 features) |

## Set up Instruction (Run it on your local computer -> terminal):

Step 1: Clone the repository
git clone hhttps://github.com/Bansal-Anmol/CPSC-571---Abusive-Sentiment-Detection.git
cd CPSC-571---Abusive-Sentiment-Detection

Step 2: Install dependencies
pip install streamlit pandas numpy scikit-learn scipy torch transformers

Step 3: Run the app
Streamlit run app.py (App will automatically open in your browser) 

Usage: 
1. Enter text in the input box
2. Select model (Logistic Regression or mBERT)
3. Click "Detect Hate Speech"
4. View result and confidence score
