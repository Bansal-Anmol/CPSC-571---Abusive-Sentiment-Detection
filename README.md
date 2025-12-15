# 🛡️ HateGuard - Multilingual Hate Speech Detection

A machine learning project that detects hate speech in **English**, **Hindi**, and **Hinglish** (code-mixed) text using two different approaches: Logistic Regression and mBERT.

**CPSC 571 Final Project - University of Calgary**

---

## 📌 Project Overview

Social media platforms face challenges in moderating hate speech across multiple languages. This project addresses the problem by building a multilingual hate speech detection system that can identify offensive content in English, Hindi, and Hinglish.

We implemented and compared two approaches:
- **Logistic Regression** with TF-IDF and profanity features
- **mBERT** (Multilingual BERT) fine-tuned for hate speech classification

---

## 🎯 Features

- Detects hate speech in 3 languages (English, Hindi, Hinglish)
- Two model options with different accuracy-speed tradeoffs
- Clean web interface built with Streamlit
- Real-time predictions with confidence scores

---

## 📊 Dataset

| Source | Language | Samples |
|--------|----------|---------|
| Davidson et al. | English | 24,783 |
| HASOC 2019 | Hindi | 9,330 |
| HEOT | Hinglish | 3,189 |
| **Total** | | **37,302** |

---

## 🤖 Models

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

## 📁 Repository Structure

```
CPSC-571---Abusive-Sentiment-Detection/
│
├── data/
│   ├── davidson_english.csv
│   ├── hasoc_hindi.csv
│   ├── heot_hinglish.csv
│   └── Hinglish_Profanity_List.csv
│
├── models/
│   ├── lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── profanity_words.pkl
│   └── profanity_df.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── logistic_regression.py
│   └── mbert_model.py
│
├── CPSC571_Final_Notebook.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Bansal-Anmol/CPSC-571---Abusive-Sentiment-Detection.git
cd CPSC-571---Abusive-Sentiment-Detection
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn scipy torch transformers
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 💻 Usage

1. Enter text in the input box
2. Select model (Logistic Regression or mBERT)
3. Click "Detect Hate Speech"
4. View result and confidence score

---

## 🔗 mBERT Model

The trained mBERT model is hosted on HuggingFace:

**https://huggingface.co/roshanp1923/mbert-hate-speech**

The app automatically downloads and loads the model from HuggingFace.

---

## 📸 Screenshots

### App Interface
- Dark purple themed UI
- Model selector (LR / mBERT)
- Real-time hate speech detection
- Confidence score display

---

## 📝 Files Description

| File | Description |
|------|-------------|
| `CPSC571_Final_Notebook.ipynb` | Main notebook with all training code |
| `app.py` | Streamlit web application |
| `preprocessing.py` | Text cleaning functions |
| `logistic_regression.py` | LR model training code |
| `mbert_model.py` | mBERT model training code |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Technical Details

### Preprocessing
- Convert to lowercase
- Remove URLs, mentions, hashtags
- Remove special characters (keep Hindi unicode)
- Remove numbers
- Remove extra whitespace

### Logistic Regression Pipeline
1. Clean text using preprocessing
2. Convert to TF-IDF features (5000 dimensions)
3. Add profanity count feature
4. Add profanity score feature
5. Combine features and predict

### mBERT Pipeline
1. Clean text using preprocessing
2. Tokenize with BERT tokenizer
3. Pass through fine-tuned mBERT
4. Apply softmax for prediction

---

## 📚 References

- Davidson, T., et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language
- HASOC 2019: Hate Speech and Offensive Content Identification
- HEOT: Hindi-English Offensive Tweet Dataset
- HuggingFace Transformers Library

---

## 👥 Team

CPSC 571 - University of Calgary

---

## 📄 License

This project is for educational purposes as part of CPSC 571 coursework.
