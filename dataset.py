import os
import pandas as pd

def get_sample_texts():
    """
    Returns a list of sample articles for the UI.
    Matches the labels in app.py: Moon Water Ice, EV Market Growth, AI in Healthcare.
    """
    return [
        {
            "text": (
                "Scientists have confirmed the presence of water ice on the Moon's surface, "
                "particularly in the permanently shadowed regions of the lunar poles. This discovery, "
                "made using data from NASA's Lunar Reconnaissance Orbiter and other missions, has "
                "significant implications for future human exploration. Water ice could be used to "
                "produce drinking water, breathable oxygen, and even rocket fuel. The south pole is "
                "of particular interest due to its large craters that never see sunlight, keeping "
                "temperatures low enough for ice to remain stable for billions of years. Researchers "
                "are now planning missions to drill into these icy deposits to understand their "
                "composition and accessibility."
            ),
            "summary": (
                "NASA has confirmed water ice on the Moon's poles, which could provide essential "
                "resources like water, oxygen, and fuel for future human missions."
            )
        },
        {
            "text": (
                "The global electric vehicle (EV) market experienced unprecedented growth in 2025, "
                "with sales increasing by 40% compared to the previous year. This surge is driven by "
                "improved battery technology, expanded charging infrastructure, and government incentives "
                "aimed at reducing carbon emissions. Major automakers are pivoting their entire lineups "
                "toward electric models, with several announcing plans to phase out internal combustion "
                "engines by 2035. Despite challenges like high upfront costs and supply chain issues for "
                "critical minerals, the transition to sustainable transportation appears to be accelerating "
                "worldwide. Consumer preference is also shifting as EVs become more affordable and offer "
                "better performance."
            ),
            "summary": (
                "The EV market grew by 40% in 2025 due to better technology and incentives, as the industry "
                "shifts towards sustainable transportation and phases out gas engines."
            )
        },
        {
            "text": (
                "Artificial Intelligence is transforming healthcare by enabling faster and more accurate "
                "diagnoses. Machine learning algorithms can analyze medical images, such as X-rays and "
                "MRIs, to detect anomalies that may be missed by the human eye. In addition to diagnostics, "
                "AI is being used for drug discovery, identifying potential treatments in a fraction of "
                "the time traditional methods take. Telehealth platforms integrated with AI-driven chatbots "
                "are also improving patient access to care and triaging symptoms. However, the use of AI "
                "in medicine raises important questions about data privacy, algorithmic bias, and the need "
                "for human oversight to ensure patient safety and ethical standards."
            ),
            "summary": (
                "AI is revolutionizing healthcare through improved diagnostics and drug discovery, though "
                "it also presents challenges regarding privacy and ethics."
            )
        }
    ]

def load_kaggle_dataset(path="data/news_summary.csv"):
    """
    Placeholder for the Kaggle dataset loader mentioned in WALKTHROUGH.md.
    Expects news_summary.csv to be present in the data/ directory.
    """
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    # Test the module
    print("Testing dataset module...")
    samples = get_sample_texts()
    print(f"Loaded {len(samples)} samples.")
    for i, s in enumerate(samples):
        print(f"Sample {i+1} length: {len(s['text'])} characters.")
