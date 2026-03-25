# 📰 Walkthrough: Automated Text Summarization System

The **Automated Text Summarization System** is now complete and ready for use. This system implements a dual approach to summarization: **Extractive** (using TF-IDF sentence scoring) and **Abstractive** (using the Google T5-small transformer).

## 🚀 How to Run the App

### Step 1: Activate the Environment
Open your terminal in the project directory and run:
```bash
venv\Scripts\activate
```

### Step 2: Install Dependencies (One-time)
If you haven't already installed everything, run:
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Launch Streamlit
```bash
streamlit run app.py
```

### Step 4: Access the UI
The app will automatically open in your browser, or you can go to:
- **Local URL:** [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Project Architecture

| File | Purpose |
|---|---|
| [`app.py`](app.py) | **Main UI.** Streamlit web app with custom dark-mode CSS and interactive charts. |
| [`model.py`](model.py) | **Core Models.** `ExtractiveSummarizer` (TF-IDF) and `AbstractiveSummarizer` (T5 Transformer). |
| [`preprocessing.py`](preprocessing.py) | **Cleaning.** Handles tokenization, stopword removal, and lemmatization. |
| [`utils.py`](utils.py) | **Scoring & Charts.** Custom ROUGE implementation and Matplotlib viz. |
| [`dataset.py`](dataset.py) | **Kaggle Loader.** Cleanup and loading of the "News Summary" dataset. |
| [`requirements.txt`](requirements.txt) | **Dependencies.** Fixed to avoid Windows build errors. |

---

## ✨ System Features

### 1. Dual Summarization Methods
- **Extractive:** Fast, selects key sentences from the original text based on TF-IDF relevance.
- **Abstractive:** Uses a pre-trained T5 model to rewrite a novel summary. (The first run downloads the ~242 MB model into the local `./hf_cache` folder).

### 2. Live ROUGE Evaluation
When a reference summary is provided (automatically loaded for sample articles), the system calculates:
- **ROUGE-1 / 2 / L:** Standard metrics for text summarization quality.

### 3. Interactive Visualizations
The app includes high-quality dark-themed charts:
- **Word Count Comparison:** Original vs Extractive vs Abstractive.
- **Reduction Percentage:** Percent of text saved per method.

---

## 📦 How to Load the Kaggle Dataset
To unlock the "Dataset Explorer" tab with 5,000 news articles:
1. Place your `kaggle.json` in `C:\Users\<YourUser>\.kaggle\kaggle.json`.
2. Run `python dataset.py` in the terminal.
3. The app will detect `data/news_summary.csv` automatically.

---

## ✅ Verification Status
- **File Structure:** All files created.
- **Syntax:** All Python files passed syntax check.
- **Cache Fix:** Removed the permission error by using a local `./hf_cache` folder.
- **Dependencies:** Installed and verified in `venv`.
