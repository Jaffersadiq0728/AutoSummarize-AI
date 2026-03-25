# 📰 Automated Text Summarization System

> An end-to-end NLP project that automatically generates concise, meaningful summaries from long documents using both **Extractive** (TF-IDF) and **Abstractive** (T5 Transformer) techniques — with a sleek Streamlit interface.

---

## 📁 Project Structure

```
Automated Text Summarization System/
├── app.py              ← Streamlit UI (main entry point)
├── model.py            ← Extractive & Abstractive summarization models
├── preprocessing.py    ← Text cleaning, tokenization, stopwords
├── utils.py            ← ROUGE evaluation + visualizations
├── dataset.py          ← Kaggle News Summary dataset loader
├── data/
│   └── news_summary.csv   ← Dataset (download via Kaggle API)
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
cd "Automated Text Summarization System"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (~2 GB) and HuggingFace T5-small (~242 MB) will be downloaded the first time the abstractive model is used. They are automatically cached.

---

## 📦 Dataset Setup (Kaggle)

The app works out-of-the-box with the **3 built-in sample articles**.  
To unlock the **Dataset Explorer** tab with 5,000 real news articles:

### Step 1 — Get your Kaggle API token
1. Go to [kaggle.com](https://www.kaggle.com) → **Settings** → **API** → **Create New Token**
2. A `kaggle.json` file is downloaded

### Step 2 — Place the token
```
Windows:  C:\Users\<YourUsername>\.kaggle\kaggle.json
Linux/Mac: ~/.kaggle/kaggle.json
```

### Step 3 — Download the dataset
```bash
python dataset.py
```
This downloads [sunnysai12345/news-summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) into `data/news_summary.csv`.

---

## 🚀 Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ✨ Features

| Feature | Details |
|---|---|
| **Extractive Summarization** | TF-IDF sentence scoring, top-N sentences selected |
| **Abstractive Summarization** | Google T5-small transformer (beam search) |
| **ROUGE Evaluation** | ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1) |
| **Word Count Reduction %** | Shown for both methods |
| **Download Summary** | .txt download for both extractive & abstractive |
| **Dataset Explorer** | Load, view, and visualize Kaggle news dataset |
| **Length Distribution Charts** | Article vs summary word count histograms |
| **Summary Comparison Chart** | Bar chart with reduction percentages |
| **Dark Glassmorphism UI** | Modern Streamlit design with custom CSS |

---

## 🧠 How It Works

### Extractive (TF-IDF)
1. Split document into sentences
2. Vectorize each sentence with `TfidfVectorizer`
3. Score each sentence by mean TF-IDF weight
4. Return top-N sentences in original order

### Abstractive (T5-small)
1. Prepend `"summarize: "` task prefix
2. Pass through T5 encoder-decoder transformer
3. Beam-search decode novel summary sentences

### ROUGE Scoring
Custom pure-Python implementation of:
- **ROUGE-1** — unigram overlap
- **ROUGE-2** — bigram overlap
- **ROUGE-L** — Longest Common Subsequence

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **NLTK** — tokenization, stopwords, lemmatization
- **Scikit-learn** — TF-IDF vectorization
- **HuggingFace Transformers** — T5-small model
- **PyTorch** — deep learning backend
- **Streamlit** — web interface
- **Matplotlib** — visualizations
- **Pandas / NumPy** — data handling
- **Kaggle API** — dataset download

---

## 📊 Evaluation

| Metric | Description |
|---|---|
| ROUGE-1 F1 | Unigram overlap between generated and reference summary |
| ROUGE-2 F1 | Bigram overlap |
| ROUGE-L F1 | Longest common subsequence |

Scores range from 0 to 1 (higher is better).

---

## 📸 Screenshots

Run the app to see the live interface with:
- Side-by-side extractive and abstractive summaries
- Interactive ROUGE bar charts
- Word count comparison visualization

---

## 📄 License

MIT License — free to use, modify, and distribute.
