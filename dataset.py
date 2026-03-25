"""
dataset.py
----------
Handles Kaggle dataset download and loading.

Dataset: "News Summary" by Kondalarao Vonteru
Kaggle: https://www.kaggle.com/datasets/sunnysai12345/news-summary

Columns used:
  - 'text'     → full article  (mapped from 'Complete Article')
  - 'summary'  → headline/short summary (mapped from 'Headlines')

Setup (one-time):
  1. Go to https://www.kaggle.com/settings → Account → Create New Token
  2. Place the downloaded kaggle.json in:
       Windows: C:\\Users\\<YourUser>\\.kaggle\\kaggle.json
  3. Run:  python dataset.py
"""

import os
import zipfile
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATASET_CSV = DATA_DIR / "news_summary.csv"
KAGGLE_DATASET = "sunnysai12345/news-summary"   # <owner>/<dataset-slug>
KAGGLE_FILE = "news_summary.csv"

# Columns in the downloaded CSV
COL_ARTICLE = "Complete Article"
COL_HEADLINE = "Headlines"
COL_SHORT = "Short text"


# ── Download helpers ──────────────────────────────────────────────────────────
def download_kaggle_dataset():
    """
    Download the News Summary dataset from Kaggle using the Kaggle API.
    Requires kaggle.json to be configured.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if DATASET_CSV.exists():
        print(f"[dataset] Dataset already exists at '{DATASET_CSV}'. Skipping download.")
        return

    try:
        import kaggle  # noqa: F401 – ensures package is installed
    except ImportError:
        raise ImportError(
            "Kaggle package not found. Install it with:  pip install kaggle\n"
            "Then place your kaggle.json API token in C:\\Users\\<you>\\.kaggle\\"
        )

    print(f"[dataset] Downloading '{KAGGLE_DATASET}' from Kaggle …")
    os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip'
    )
    print("[dataset] Download complete.")


# ── Loading ───────────────────────────────────────────────────────────────────
def load_dataset(max_rows: int = 5000) -> pd.DataFrame:
    """
    Load and clean the News Summary CSV.

    Returns a DataFrame with columns:
        'text'    – full article text
        'summary' – reference summary (headline or short description)
    """
    if not DATASET_CSV.exists():
        download_kaggle_dataset()

    print(f"[dataset] Loading data from '{DATASET_CSV}' …")
    df = pd.read_csv(DATASET_CSV, encoding="latin-1")

    # ── Basic info ─────────────────────────────────────────────────────────────
    print(f"[dataset] Raw shape: {df.shape}")
    print(f"[dataset] Columns  : {list(df.columns)}")

    # ── Rename to standard column names ────────────────────────────────────────
    rename_map = {}
    if COL_ARTICLE in df.columns:
        rename_map[COL_ARTICLE] = "text"
    if COL_HEADLINE in df.columns:
        rename_map[COL_HEADLINE] = "summary"
    elif COL_SHORT in df.columns:
        rename_map[COL_SHORT] = "summary"

    df = df.rename(columns=rename_map)

    # ── Keep only the two relevant columns ────────────────────────────────────
    keep = [c for c in ["text", "summary"] if c in df.columns]
    df = df[keep].copy()

    # ── Clean nulls & duplicates ──────────────────────────────────────────────
    df.dropna(subset=["text", "summary"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df[df["text"].str.split().str.len() > 20]   # drop very short articles
    df = df[df["summary"].str.split().str.len() > 2]  # drop trivial summaries

    # ── Reset and cap ─────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    print(f"[dataset] Clean shape: {df.shape}")
    return df


def get_sample_texts() -> list:
    """
    Return a small list of hard-coded sample articles for quick demo
    when the Kaggle dataset has not yet been downloaded.
    """
    return [
        {
            "text": (
                "Scientists at NASA have confirmed the discovery of water ice on the surface of the Moon "
                "near its south pole. This groundbreaking finding was made using data collected by the "
                "Lunar Reconnaissance Orbiter and several other spacecraft. The presence of water ice "
                "could be crucial for future human missions to the Moon, as it can potentially be used "
                "for drinking water, oxygen production, and even rocket fuel. Researchers believe the "
                "ice has been stable in permanently shadowed craters for billions of years due to "
                "extremely low temperatures. This discovery opens up new possibilities for establishing "
                "a sustainable human presence on the Moon in the coming decades. NASA's Artemis program "
                "aims to return humans to the Moon by 2025 and leverage these resources for long-duration "
                "missions. Scientists from multiple countries are collaborating to study the extent and "
                "purity of the water ice deposits."
            ),
            "summary": "NASA confirms water ice on Moon's south pole, boosting future mission prospects.",
        },
        {
            "text": (
                "The global electric vehicle market experienced record-breaking growth in 2024, with sales "
                "surpassing 14 million units worldwide. China remained the dominant market, accounting for "
                "nearly 60 percent of all EV sales. Europe saw a 25 percent increase in EV adoption, driven "
                "by stricter emission regulations and generous government subsidies. In the United States, "
                "the Inflation Reduction Act continued to incentivize consumers with tax credits of up to "
                "$7,500 for qualifying vehicles. Major automakers, including Ford, General Motors, and "
                "Volkswagen, announced multi-billion-dollar investments to expand their EV lineups. Battery "
                "technology is also advancing rapidly, with new solid-state batteries promising greater range "
                "and faster charging times. Analysts predict that EVs will account for over 30 percent of "
                "global car sales by 2030 as infrastructure continues to expand."
            ),
            "summary": "Global EV sales hit record 14 million units in 2024 with China leading growth.",
        },
        {
            "text": (
                "Artificial intelligence has transformed the healthcare industry in remarkable ways over the "
                "past few years. Machine learning algorithms are now capable of detecting diseases such as "
                "cancer, diabetic retinopathy, and cardiovascular conditions with accuracy comparable to or "
                "even exceeding that of human specialists. Hospitals are deploying AI-powered diagnostic "
                "tools to analyze medical images, pathology slides, and patient records to identify "
                "conditions earlier and more accurately. Natural language processing is being used to "
                "extract insights from clinical notes and streamline medical documentation. Drug discovery "
                "has also been accelerated by AI, with companies like DeepMind and BioNTech using machine "
                "learning to identify potential drug candidates in weeks rather than years. Personalized "
                "medicine is becoming a reality as AI systems analyze genetic data to recommend tailored "
                "treatment plans for individual patients. Despite these advances, concerns around data "
                "privacy, algorithmic bias, and regulatory approval remain significant challenges."
            ),
            "summary": "AI is revolutionizing healthcare with advanced diagnostics, drug discovery, and personalized medicine.",
        },
    ]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_kaggle_dataset()
    df = load_dataset(max_rows=1000)
    print(df.head())
    print(f"\nText word count stats:\n{df['text'].str.split().str.len().describe()}")
    print(f"\nSummary word count stats:\n{df['summary'].str.split().str.len().describe()}")
