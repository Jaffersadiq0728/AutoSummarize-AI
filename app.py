import io
import textwrap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import word_count, reduction_percentage
from model import get_extractive_summary, get_abstractive_summary
from utils import (
    compute_rouge,
    rouge_scores_to_df,
    plot_text_length_distribution,
    plot_summary_length_comparison,
    plot_rouge_scores,
    fig_to_bytes,
    PALETTE,
)
from dataset import get_sample_texts

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoSummarize AI | Professional NLP Toolkit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern CSS Styling ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #FF6584;
        --bg-dark: #0f0f1a;
        --surface-dark: #1a1a2e;
        --text-main: #e0e0ff;
        --text-muted: #888aaa;
    }

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    
    .stApp { 
        background: radial-gradient(circle at top right, #1a1a2e, #0f0f1a);
        color: var(--text-main); 
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] { 
        background: rgba(30, 30, 50, 0.95); 
        border-right: 1px solid rgba(108, 99, 255, 0.2); 
    }

    /* Card Styling */
    .summary-card { 
        background: rgba(42, 42, 62, 0.6); 
        border: 1px solid rgba(108, 99, 255, 0.3); 
        border-radius: 20px; 
        padding: 2rem; 
        margin-bottom: 2rem; 
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: transform 0.3s ease;
    }
    .summary-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary-color);
    }

    /* Professional Headings */
    .hero-container {
        padding: 3rem 0;
        text-align: center;
    }
    .hero-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #6C63FF, #FF6584, #43B89C); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem; 
    }
    .hero-sub { 
        color: var(--text-muted); 
        font-size: 1.2rem; 
        max-width: 700px;
        margin: 0 auto 2rem auto;
    }

    /* Badges */
    .badge { 
        display: inline-block; 
        padding: 0.4rem 1rem; 
        border-radius: 12px; 
        font-size: 0.75rem; 
        font-weight: 700; 
        text-transform: uppercase;
        margin-bottom: 1rem; 
    }
    .badge-extractive  { background: rgba(108,99,255,0.15); color: #6C63FF; border: 1px solid #6C63FF; }
    .badge-abstractive { background: rgba(255,101,132,0.15); color: #FF6584; border: 1px solid #FF6584; }

    /* Custom Input */
    .stTextArea textarea {
        background: rgba(20, 20, 35, 0.8) !important;
        border: 1px solid rgba(108, 99, 255, 0.2) !important;
        border-radius: 15px !important;
        color: white !important;
    }
    
    /* Metrics */
    .metric-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .metric-item {
        background: rgba(255,255,255,0.03);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    .metric-lab {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
    }

    hr { border-color: rgba(108,99,255,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/6C63FF/brain.png", width=60)
    st.markdown("### Model Configuration")
    st.markdown("---")

    with st.expander("🛠️ Summarization Parameters", expanded=True):
        st.markdown("**Extractive**")
        num_sentences = st.slider("Sentence Count", 1, 10, 3)
        
        st.markdown("**Abstractive (T5)**")
        max_len = st.slider("Max Length", 60, 300, 150)
        min_len = st.slider("Min Length", 20, 100, 40)

    st.markdown("---")
    st.markdown("### Quick Samples")
    samples = get_sample_texts()
    sample_labels = ["🌕 Moon Habitat", "🚗 EV Revolution", "🏥 AI Medical"]
    selected_sample = st.selectbox("Load Example", ["None"] + sample_labels)

    st.markdown("---")
    st.info("💡 **Tip:** Abstractive summarization creates new sentences, while extractive selects the most relevant original ones.")

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">AutoSummarize AI</div>
        <div class="hero-sub">Enter your long-form text below and let our dual-engine NLP transform it into concise, high-impact summaries.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main Content Loop ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✨ Summarize", "📈 Analytics & Evaluation"])

default_text = ""
if selected_sample != "None":
    idx = sample_labels.index(selected_sample)
    default_text = samples[idx]["text"]

with tab1:
    input_col, output_col = st.columns([1, 1.2], gap="large")

    with input_col:
        st.subheader("Input Text")
        user_text = st.text_area(
            "Article content",
            value=default_text,
            height=400,
            placeholder="Paste your content here (articles, reports, essays)...",
            label_visibility="collapsed"
        )
        
        # Reference summary for ROUGE
        with st.expander("Optional: Reference Summary (for ROUGE scoring)"):
            ref_summary = st.text_area(
                "Ground truth summary",
                height=100,
                placeholder="Paste original summary if available..."
            )
            if selected_sample != "None" and not ref_summary:
                idx = sample_labels.index(selected_sample)
                ref_summary = samples[idx]["summary"]
                st.caption(f"✅ Loaded reference for '{selected_sample}'")

        process_btn = st.button("🚀 Process Summaries", use_container_width=True)

    with output_col:
        st.subheader("Results")
        if not process_btn and not default_text:
            st.info("Waiting for input text... Paste an article and click process.")
        
        if process_btn:
            if not user_text.strip():
                st.warning("Please provide input text first.")
            else:
                progress_bar = st.progress(0)
                
                # 1. Extractive
                try:
                    progress_bar.progress(20, "Running Extractive Engine...")
                    ext_summary = get_extractive_summary(user_text, num_sentences=num_sentences)
                except Exception as e:
                    st.error(f"Extractive Error: {e}")
                    ext_summary = "Error generating extractive summary."

                # 2. Abstractive
                try:
                    progress_bar.progress(50, "Running Neural Abstractive Engine (T5)...")
                    abs_summary = get_abstractive_summary(user_text, max_length=max_len, min_length=min_len)
                except Exception as e:
                    st.error(f"Abstractive Error: {e}. Check if dependencies are installed correctly.")
                    abs_summary = "Error generating abstractive summary."

                progress_bar.progress(100, "Finalizing...")
                
                # Calculations
                ext_wc = word_count(ext_summary)
                abs_wc = word_count(abs_summary)
                ext_red = reduction_percentage(user_text, ext_summary)
                abs_red = reduction_percentage(user_text, abs_summary)

                # Rendering
                # Extractive Card
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <span class="badge badge-extractive">Extractive Engine</span>
                        <p style="font-size: 1.05rem; line-height: 1.6;">{ext_summary}</p>
                        <div class="metric-container">
                            <div class="metric-item">
                                <div class="metric-val">{ext_wc}</div>
                                <div class="metric-lab">Word Count</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-val" style="color: #43B89C">-{ext_red}%</div>
                                <div class="metric-lab">Reduction</div>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # Abstractive Card
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <span class="badge badge-abstractive">Neural Abstractive Engine</span>
                        <p style="font-size: 1.05rem; line-height: 1.6;">{abs_summary}</p>
                        <div class="metric-container">
                            <div class="metric-item">
                                <div class="metric-val" style="color: #FF6584">{abs_wc}</div>
                                <div class="metric-lab">Word Count</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-val" style="color: #43B89C">-{abs_red}%</div>
                                <div class="metric-lab">Reduction</div>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # Store in session state for tab2
                st.session_state['ext_summary'] = ext_summary
                st.session_state['abs_summary'] = abs_summary
                st.session_state['user_text'] = user_text
                st.session_state['ref_summary'] = ref_summary

with tab2:
    if 'ext_summary' not in st.session_state:
        st.info("Run a summary first to view detailed analytics.")
    else:
        st.subheader("System Performance & Comparison")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Visual Distribution")
            fig_len = plot_summary_length_comparison(
                st.session_state['user_text'], 
                st.session_state['ext_summary'], 
                st.session_state['abs_summary']
            )
            st.pyplot(fig_len)
        
        with c2:
            st.markdown("#### Download Export")
            combined = f"--- EXTRACTIVE ---\n{st.session_state['ext_summary']}\n\n--- ABSTRACTIVE ---\n{st.session_state['abs_summary']}"
            st.download_button("📩 Download Summaries (.txt)", combined, "summaries.txt")
            st.download_button("🖼️ Download Length Chart (.png)", fig_to_bytes(fig_len), "chart.png")

        if st.session_state['ref_summary'].strip():
            st.markdown("---")
            st.markdown("#### 📐 ROUGE Score Comparison")
            r_ext = compute_rouge(st.session_state['ext_summary'], st.session_state['ref_summary'])
            r_abs = compute_rouge(st.session_state['abs_summary'], st.session_state['ref_summary'])
            
            fig_rouge = plot_rouge_scores({"Extractive": r_ext, "Abstractive": r_abs})
            st.pyplot(fig_rouge)
            
            cols = st.columns(2)
            with cols[0]:
                st.caption("Extractive Scores")
                st.dataframe(rouge_scores_to_df(r_ext), use_container_width=True)
            with cols[1]:
                st.caption("Abstractive Scores")
                st.dataframe(rouge_scores_to_df(r_abs), use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888aaa; font-size: 0.8rem;'>"
    "AutoSummarize AI v2.0 | Powered by HuggingFace T5 & NLTK | Professional Edition"
    "</div>", 
    unsafe_allow_html=True
)
