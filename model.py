import math
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from preprocessing import (
    clean_text,
    get_sentence_list,
    preprocess_for_tfidf,
    remove_stopwords,
    tokenize_words,
)

def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except (LookupError, AttributeError):
            print(f"Downloading NLTK resource: {resource}...")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error downloading NLTK resource {resource}: {e}")

# Ensure resources are available
download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class ExtractiveSummarizer:
    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def _score_sentences(self, sentences: list) -> np.ndarray:
        processed = [preprocess_for_tfidf(s) for s in sentences]
        valid_mask = [bool(p.strip()) for p in processed]
        valid_processed = [p for p, v in zip(processed, valid_mask) if v]

        if len(valid_processed) < 2:
            return np.ones(len(sentences))

        try:
            tfidf_matrix = self.vectorizer.fit_transform(valid_processed)
        except ValueError:
            return np.ones(len(sentences))

        tfidf_dense = tfidf_matrix.toarray()
        scores = np.zeros(len(sentences))
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                scores[i] = float(np.mean(tfidf_dense[valid_idx]))
                valid_idx += 1

        return scores

    def summarize(self, text: str, num_sentences: int | None = None) -> str:
        n = num_sentences or self.num_sentences
        sentences = get_sentence_list(text)

        if not sentences:
            return "Unable to extract sentences from the provided text."

        if len(sentences) <= n:
            return " ".join(sentences)

        scores = self._score_sentences(sentences)
        ranked_indices = np.argsort(scores)[::-1][:n]
        selected_indices = sorted(ranked_indices)
        summary = " ".join(sentences[i] for i in selected_indices)
        return summary


import os
from pathlib import Path

CACHE_DIR = Path("hf_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR.absolute())
os.environ["HF_HOME"] = str(CACHE_DIR.absolute())

class AbstractiveSummarizer:
    MODEL_NAME = "t5-small"
    _model = None
    _tokenizer = None

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer, logging as hf_logging
                hf_logging.set_verbosity_error()
                
                print(f"[model] Loading T5-small into {CACHE_DIR.absolute()} …")
                
                try:
                    self._tokenizer = T5Tokenizer.from_pretrained(
                        self.MODEL_NAME, 
                        cache_dir=str(CACHE_DIR.absolute()),
                        legacy=False
                    )
                    self._model = T5ForConditionalGeneration.from_pretrained(
                        self.MODEL_NAME, 
                        cache_dir=str(CACHE_DIR.absolute())
                    )
                    print("[model] T5-small loaded successfully.")
                except Exception as e:
                    print(f"[model] ERROR loading T5-small: {e}")
                    raise RuntimeError(f"Failed to load T5 model: {e}")
            except ImportError as e:
                print(f"[model] ImportError: {e}")
                raise e
        return self._model, self._tokenizer

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        do_sample: bool = False,
    ) -> str:
        model, tokenizer = self._load_model()
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])

        input_text = "summarize: " + text
        inputs = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            do_sample=do_sample
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

_extractive_model = ExtractiveSummarizer(num_sentences=3)
_abstractive_model = AbstractiveSummarizer()

def get_extractive_summary(text: str, num_sentences: int = 3) -> str:
    return _extractive_model.summarize(text, num_sentences=num_sentences)

def get_abstractive_summary(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
) -> str:
    return _abstractive_model.summarize(
        text, max_length=max_length, min_length=min_length
    )
