"""
preprocessing.py
----------------
Handles all text preprocessing steps:
  - Lowercasing
  - Removing special characters / punctuation
  - Tokenization
  - Stopword removal
  - Stemming / Lemmatization
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (runs once)
def download_nltk_resources():
    """Download all required NLTK resources silently."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# ── Constants ────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ── Core cleaning helpers ────────────────────────────────────────────────────
def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_special_characters(text: str) -> str:
    """Remove URLs, HTML tags, and non-alphanumeric characters."""
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"<.*?>", "", text)                    # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?]", "", text)    # Keep sentence markers
    text = re.sub(r"\s+", " ", text).strip()             # Collapse whitespace
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize_words(text: str) -> list:
    """Tokenize text into individual words."""
    return word_tokenize(text)


def tokenize_sentences(text: str) -> list:
    """Tokenize text into sentences."""
    return sent_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove English stopwords from a token list."""
    return [t for t in tokens if t.lower() not in STOP_WORDS and t.isalpha()]


def stem_tokens(tokens: list) -> list:
    """Apply Porter stemming to token list."""
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list) -> list:
    """Apply WordNet lemmatization to token list."""
    return [lemmatizer.lemmatize(t) for t in tokens]


# ── High-level pipeline ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      lowercase → remove special characters → collapse whitespace
    Preserves sentence structure (keeps . ! ?).
    """
    text = lowercase(text)
    text = remove_special_characters(text)
    return text


def preprocess_for_tfidf(text: str, use_lemma: bool = True) -> str:
    """
    Preprocess text for TF-IDF vectorization.
    Returns a single cleaned string without stopwords.
    """
    text = clean_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_words(text)
    tokens = remove_stopwords(tokens)
    if use_lemma:
        tokens = lemmatize_tokens(tokens)
    else:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def get_sentence_list(text: str) -> list:
    """
    Return a cleaned list of sentences from original text.
    Used by the extractive summarizer.
    """
    # Keep original sentences for readability but clean lightly
    text = remove_special_characters(text)
    sentences = tokenize_sentences(text)
    # Drop very short sentences (< 5 words)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    return sentences


def word_count(text: str) -> int:
    """Return total word count of a text string."""
    return len(text.split())


def reduction_percentage(original: str, summary: str) -> float:
    """Calculate the word count reduction percentage."""
    orig_wc = word_count(original)
    summ_wc = word_count(summary)
    if orig_wc == 0:
        return 0.0
    return round((1 - summ_wc / orig_wc) * 100, 2)
