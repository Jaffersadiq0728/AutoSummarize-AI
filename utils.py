"""
utils.py
--------
Utility functions for:
  • ROUGE evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
  • Visualization (length distributions, summary comparison chart)
  • Miscellaneous helpers
"""

from __future__ import annotations

import io
from collections import Counter

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  ROUGE SCORER
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens: list, n: int) -> Counter:
    """Return a Counter of n-grams from a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list, b: list) -> int:
    """Compute length of the Longest Common Subsequence of two token lists."""
    m, n = len(a), len(b)
    # Use only 1-D DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_n(hypothesis: str, reference: str, n: int) -> dict:
    """Compute ROUGE-N precision, recall, and F1."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    overlap = sum((hyp_ngrams & ref_ngrams).values())
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall    = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _rouge_l(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-L using the LCS approach."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / max(len(hyp_tokens), 1)
    recall    = lcs / max(len(ref_tokens), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Parameters
    ----------
    hypothesis : Generated / predicted summary.
    reference  : Ground-truth / reference summary.

    Returns
    -------
    dict with keys 'rouge1', 'rouge2', 'rougeL', each containing
    sub-keys 'precision', 'recall', 'f1'.
    """
    return {
        "rouge1": _rouge_n(hypothesis, reference, 1),
        "rouge2": _rouge_n(hypothesis, reference, 2),
        "rougeL": _rouge_l(hypothesis, reference),
    }


def rouge_scores_to_df(rouge_dict: dict) -> pd.DataFrame:
    """Convert ROUGE score dict to a tidy DataFrame for display."""
    rows = []
    for metric, scores in rouge_dict.items():
        rows.append(
            {
                "Metric": metric.upper(),
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1 Score": scores["f1"],
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Shared color palette
PALETTE = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "accent":     "#43B89C",
    "bg":         "#1E1E2E",
    "surface":    "#2A2A3E",
    "text":       "#E0E0FF",
    "muted":      "#888AAA",
}


def _apply_dark_style(fig: plt.Figure, ax):
    """Apply a consistent dark theme to a figure/axis."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])


def plot_text_length_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the word-count distribution of articles and summaries.

    Parameters
    ----------
    df : DataFrame with 'text' and 'summary' columns.

    Returns
    -------
    matplotlib Figure object.
    """
    article_lengths = df["text"].str.split().str.len()
    summary_lengths = df["summary"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Word Count Distribution", fontsize=16, color=PALETTE["text"], fontweight="bold")

    for ax, lengths, label, color in zip(
        axes,
        [article_lengths, summary_lengths],
        ["Article Word Count", "Summary Word Count"],
        [PALETTE["primary"], PALETTE["secondary"]],
    ):
        ax.hist(lengths, bins=40, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(lengths.mean(), color="white", linestyle="--", linewidth=1.5,
                   label=f"Mean = {lengths.mean():.0f}")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=10, facecolor=PALETTE["surface"],
                  labelcolor=PALETTE["text"], edgecolor=PALETTE["muted"])
        _apply_dark_style(fig, ax)

    fig.tight_layout()
    return fig


def plot_summary_length_comparison(
    original_text: str,
    extractive_summary: str,
    abstractive_summary: str,
) -> plt.Figure:
    """
    Bar chart comparing word counts of original text, extractive, and abstractive
    summaries — plus reduction percentages.
    """
    labels   = ["Original Text", "Extractive\nSummary", "Abstractive\nSummary"]
    counts   = [
        len(original_text.split()),
        len(extractive_summary.split()),
        len(abstractive_summary.split()),
    ]
    colors   = [PALETTE["muted"], PALETTE["primary"], PALETTE["secondary"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="none")

    # Annotate bars with word count
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,} words",
            ha="center", va="bottom",
            color=PALETTE["text"], fontsize=11, fontweight="bold",
        )

    # Reduction % labels for summaries
    orig = counts[0]
    for i in [1, 2]:
        pct = round((1 - counts[i] / orig) * 100, 1) if orig else 0
        ax.text(
            bars[i].get_x() + bars[i].get_width() / 2,
            bars[i].get_height() / 2,
            f"↓ {pct}%",
            ha="center", va="center",
            color="white", fontsize=12, fontweight="bold", alpha=0.9,
        )

    ax.set_ylabel("Word Count", fontsize=12)
    ax.set_title("Summary Length Comparison", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_rouge_scores(rouge_dict: dict) -> plt.Figure:
    """
    Grouped bar chart of ROUGE precision / recall / F1
    for both extractive and abstractive summaries.

    Parameters
    ----------
    rouge_dict : {
        'extractive': { 'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...} },
        'abstractive': { ... }
    }
    """
    metrics   = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    sub_keys  = ["rouge1", "rouge2", "rougeL"]
    methods   = list(rouge_dict.keys())
    method_colors = [PALETTE["primary"], PALETTE["secondary"]]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (method, color) in enumerate(zip(methods, method_colors)):
        f1_scores = [rouge_dict[method][k]["f1"] for k in sub_keys]
        bars = ax.bar(
            x + (i - 0.5) * width,
            f1_scores,
            width,
            label=method.capitalize(),
            color=color,
            alpha=0.9,
            edgecolor="none",
        )
        for bar, score in zip(bars, f1_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.2f}",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=10, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("ROUGE F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, facecolor=PALETTE["surface"],
              labelcolor=PALETTE["text"], edgecolor=PALETTE["muted"])
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Serialize a matplotlib Figure to PNG bytes (for Streamlit download)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
