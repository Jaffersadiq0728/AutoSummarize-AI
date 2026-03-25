"""
model.py
--------
Two summarization approaches:

1. EXTRACTIVE  – TF-IDF + sentence scoring (TextRank-style)
   • Fast, no GPU required
   • Selects the most informative sentences from the original text

2. ABSTRACTIVE – HuggingFace T5-small (transfer learning)
   • Generates brand-new sentences not in the original text
   • Requires ~240 MB model download on first run (cached automatically)
"""

import math
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import (
    clean_text,
    get_sentence_list,
    preprocess_for_tfidf,
    remove_stopwords,
    tokenize_words,
)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  EXTRACTIVE SUMMARIZER  (TF-IDF sentence scoring)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractiveSummarizer:
    """
    TF-IDF based extractive summarizer.

    Algorithm:
      1. Split text into sentences.
      2. Compute TF-IDF matrix over sentences.
      3. Score each sentence as the mean of its TF-IDF values.
      4. Return the top-N sentences in their original order.
    """

    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def _score_sentences(self, sentences: list) -> np.ndarray:
        """Compute a relevance score for each sentence using TF-IDF."""
        # Pre-process each sentence for TF-IDF (no stopwords, lemmatized)
        processed = [preprocess_for_tfidf(s) for s in sentences]

        # Filter out sentences that become empty after preprocessing
        valid_mask = [bool(p.strip()) for p in processed]
        valid_processed = [p for p, v in zip(processed, valid_mask) if v]

        if len(valid_processed) < 2:
            # Fallback: return uniform scores if not enough content
            return np.ones(len(sentences))

        try:
            tfidf_matrix = self.vectorizer.fit_transform(valid_processed)
        except ValueError:
            return np.ones(len(sentences))

        # Map valid sentence scores back to original indices
        tfidf_dense = tfidf_matrix.toarray()
        scores = np.zeros(len(sentences))
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                scores[i] = float(np.mean(tfidf_dense[valid_idx]))
                valid_idx += 1

        return scores

    def summarize(self, text: str, num_sentences: int = None) -> str:
        """
        Generate an extractive summary.

        Parameters
        ----------
        text : str
            The original long text.
        num_sentences : int, optional
            Number of sentences to include (overrides instance default).

        Returns
        -------
        str
            The extractive summary.
        """
        n = num_sentences or self.num_sentences
        sentences = get_sentence_list(text)

        if not sentences:
            return "Unable to extract sentences from the provided text."

        if len(sentences) <= n:
            return " ".join(sentences)

        # Score and pick top-N
        scores = self._score_sentences(sentences)
        ranked_indices = np.argsort(scores)[::-1][:n]
        # Preserve original sentence order for readability
        selected_indices = sorted(ranked_indices)
        summary = " ".join(sentences[i] for i in selected_indices)
        return summary


import os
from pathlib import Path

# ── Cache Configuration (Fixes PermissionError on Windows) ───────────────────
CACHE_DIR = Path("hf_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR.absolute())
os.environ["HF_HOME"] = str(CACHE_DIR.absolute())

# ─────────────────────────────────────────────────────────────────────────────
#  2.  ABSTRACTIVE SUMMARIZER  (T5-small via HuggingFace Transformers)
# ─────────────────────────────────────────────────────────────────────────────

class AbstractiveSummarizer:
    """
    Abstractive summarizer using Google's T5-small model.

    The model is ~242 MB and is downloaded once to the local ./hf_cache folder.
    """

    MODEL_NAME = "t5-small"
    _model = None
    _tokenizer = None

    def _load_model(self):
        """Lazy-load the T5 model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer, logging as hf_logging
                hf_logging.set_verbosity_error()
                
                print(f"[model] Loading T5-small into {CACHE_DIR.absolute()} …")
                
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
            except ImportError:
                raise ImportError("transformers or sentencepiece package not found.")
        return self._model, self._tokenizer

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        do_sample: bool = False,
    ) -> str:
        """Generate an abstractive summary using T5 model directly."""
        model, tokenizer = self._load_model()

        # T5-small handles up to 512 tokens
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

        # Generate summary
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



# ─────────────────────────────────────────────────────────────────────────────
#  3.  COMBINED  (convenience wrapper used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singletons (avoids reloading on every function call)
_extractive_model = ExtractiveSummarizer(num_sentences=3)
_abstractive_model = AbstractiveSummarizer()


def get_extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Return an extractive summary with a given number of sentences."""
    return _extractive_model.summarize(text, num_sentences=num_sentences)


def get_abstractive_summary(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
) -> str:
    """Return an abstractive (T5-generated) summary."""
    return _abstractive_model.summarize(
        text, max_length=max_length, min_length=min_length
    )
