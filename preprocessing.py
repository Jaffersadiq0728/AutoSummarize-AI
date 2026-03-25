import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def download_nltk_resources():
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

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def lowercase(text: str) -> str:
    return text.lower()


def remove_special_characters(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def tokenize_words(text: str) -> list:
    return word_tokenize(text)

def tokenize_sentences(text: str) -> list:
    return sent_tokenize(text)

def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t.lower() not in STOP_WORDS and t.isalpha()]

def stem_tokens(tokens: list) -> list:
    return [stemmer.stem(t) for t in tokens]

def lemmatize_tokens(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


def clean_text(text: str) -> str:
    text = lowercase(text)
    text = remove_special_characters(text)
    return text


def preprocess_for_tfidf(text: str, use_lemma: bool = True) -> str:
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
    text = remove_special_characters(text)
    sentences = tokenize_sentences(text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    return sentences


def word_count(text: str) -> int:
    return len(text.split())

def reduction_percentage(original: str, summary: str) -> float:
    orig_wc = word_count(original)
    summ_wc = word_count(summary)
    if orig_wc == 0:
        return 0.0
    return round((1 - summ_wc / orig_wc) * 100, 2)
