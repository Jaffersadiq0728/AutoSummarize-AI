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

st.set_page_config(
    page_title="AutoSummarize AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); color: #e0e0ff; }
    section[data-testid="stSidebar"] { background: rgba(30, 30, 50, 0.95); border-right: 1px solid rgba(108, 99, 255, 0.3); }
    .summary-card { background: rgba(42, 42, 62, 0.85); border: 1px solid rgba(108, 99, 255, 0.4); border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; backdrop-filter: blur(8px); box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
    .metric-row { display: flex; gap: 1rem; margin: 0.8rem 0; }
    .metric-box { flex: 1; background: rgba(108, 99, 255, 0.12); border: 1px solid rgba(108, 99, 255, 0.35); border-radius: 12px; padding: 0.9rem 1rem; text-align: center; }
    .metric-label { font-size: 0.75rem; color: #888aaa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #6C63FF; }
    .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .badge-extractive  { background: rgba(108,99,255,0.25); color: #6C63FF; border: 1px solid #6C63FF; }
    .badge-abstractive { background: rgba(255,101,132,0.25); color: #FF6584; border: 1px solid #FF6584; }
    .hero-title { font-size: 2.8rem; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #FF6584, #43B89C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
    .hero-sub { color: #888aaa; font-size: 1.05rem; margin-bottom: 2rem; }
    .stButton > button { background: linear-gradient(90deg, #6C63FF, #FF6584); color: white; border: none; border-radius: 12px; padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600; transition: opacity 0.2s; }
    .stButton > button:hover { opacity: 0.88; }
    textarea { background: rgba(30, 30, 50, 0.9) !important; border: 1px solid rgba(108,99,255,0.4) !important; border-radius: 12px !important; color: #e0e0ff !important; font-size: 0.95rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: rgba(42,42,62,0.7); border-radius: 10px 10px 0 0; border: 1px solid rgba(108,99,255,0.2); color: #888aaa; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: rgba(108,99,255,0.2) !important; color: #e0e0ff !important; border-color: #6C63FF !important; }
    hr { border-color: rgba(108,99,255,0.2); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### Extractive")
    num_sentences = st.slider(
        "Number of sentences", min_value=1, max_value=10, value=3, step=1
    )

    st.markdown("### Abstractive (T5)")
    max_len = st.slider("Max summary length (tokens)", 60, 300, 150, 10)
    min_len = st.slider("Min summary length (tokens)", 20, 100, 40, 5)

    st.markdown("---")
    st.markdown("### Example Articles")
    samples = get_sample_texts()
    sample_labels = ["🌕 Moon Water Ice", "🚗 EV Market Growth", "🏥 AI in Healthcare"]
    selected_sample = st.selectbox("Load a sample article", ["None"] + sample_labels)

    st.markdown("---")
    st.markdown(
        "<small style='color:#888aaa'>Model: T5-small (HuggingFace)<br>"
        "Dataset: Kaggle News Summary<br>"
        "ROUGE: custom implementation</small>",
        unsafe_allow_html=True,
    )


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📰 AutoSummarize AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Extractive & Abstractive NLP summarization — powered by TF-IDF and T5</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_summarize, = st.tabs(["✨ Summarize"])

with tab_summarize:
    default_text = ""
    if selected_sample != "None":
        idx = sample_labels.index(selected_sample)
        default_text = samples[idx]["text"]

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### 📝 Input Article")
        user_text = st.text_area(
            "Paste your long article here …",
            value=default_text,
            height=320,
            label_visibility="collapsed",
            placeholder="Paste your article text here. The longer the better!",
        )

        # Optional reference summary for ROUGE evaluation
        with st.expander("📌 Add Reference Summary (for ROUGE evaluation)"):
            ref_summary = st.text_area(
                "Reference / ground-truth summary (optional)",
                height=100,
                placeholder="Paste the original summary to compute ROUGE scores …",
            )
            if selected_sample != "None" and not ref_summary:
                idx = sample_labels.index(selected_sample)
                st.caption(
                    f"✅ Reference auto-loaded: *{samples[idx]['summary']}*"
                )
                ref_summary = samples[idx]["summary"]

        generate_btn = st.button("🚀 Generate Summary", use_container_width=True)

    with col_output:
        st.markdown("#### 📋 Generated Summaries")

        if generate_btn:
            if not user_text.strip():
                st.warning("Please enter or paste some text first.")
            else:
                # ── Extractive ───────────────────────────────────────────────
                with st.spinner("Running extractive summarizer …"):
                    ext_summary = get_extractive_summary(
                        user_text, num_sentences=num_sentences
                    )

                # ── Abstractive ──────────────────────────────────────────────
                with st.spinner("Running T5 abstractive summarizer (may take ~15s first run) …"):
                    abs_summary = get_abstractive_summary(
                        user_text, max_length=max_len, min_length=min_len
                    )

                # ── Word counts ──────────────────────────────────────────────
                orig_wc = word_count(user_text)
                ext_wc  = word_count(ext_summary)
                abs_wc  = word_count(abs_summary)
                ext_red = reduction_percentage(user_text, ext_summary)
                abs_red = reduction_percentage(user_text, abs_summary)

                # ── Display extractive ────────────────────────────────────────
                st.markdown(
                    f"""
                    <div class="summary-card">
                      <span class="badge badge-extractive">EXTRACTIVE</span>
                      <p style="color:#e0e0ff; font-size:0.97rem; line-height:1.7">{ext_summary}</p>
                      <div class="metric-row">
                        <div class="metric-box">
                          <div class="metric-label">Words</div>
                          <div class="metric-value">{ext_wc}</div>
                        </div>
                        <div class="metric-box">
                          <div class="metric-label">Reduction</div>
                          <div class="metric-value" style="color:#43B89C">{ext_red}%</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Display abstractive ───────────────────────────────────────
                st.markdown(
                    f"""
                    <div class="summary-card">
                      <span class="badge badge-abstractive">ABSTRACTIVE (T5)</span>
                      <p style="color:#e0e0ff; font-size:0.97rem; line-height:1.7">{abs_summary}</p>
                      <div class="metric-row">
                        <div class="metric-box">
                          <div class="metric-label">Words</div>
                          <div class="metric-value" style="color:#FF6584">{abs_wc}</div>
                        </div>
                        <div class="metric-box">
                          <div class="metric-label">Reduction</div>
                          <div class="metric-value" style="color:#43B89C">{abs_red}%</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Download buttons ──────────────────────────────────────────
                dl_col1, dl_col2 = st.columns(2)
                combined = (
                    f"=== EXTRACTIVE SUMMARY ===\n{ext_summary}\n\n"
                    f"=== ABSTRACTIVE SUMMARY (T5) ===\n{abs_summary}\n"
                )
                with dl_col1:
                    st.download_button(
                        "⬇️ Download Extractive",
                        data=ext_summary,
                        file_name="extractive_summary.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.download_button(
                        "⬇️ Download Abstractive",
                        data=abs_summary,
                        file_name="abstractive_summary.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                st.download_button(
                    "📦 Download Both Summaries",
                    data=combined,
                    file_name="summaries.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                # ── ROUGE scores ──────────────────────────────────────────────
                if ref_summary.strip():
                    st.markdown("---")
                    st.markdown("#### 📐 ROUGE Evaluation")
                    rouge_ext = compute_rouge(ext_summary, ref_summary)
                    rouge_abs = compute_rouge(abs_summary, ref_summary)

                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        st.markdown("**Extractive ROUGE**")
                        st.dataframe(rouge_scores_to_df(rouge_ext), use_container_width=True)
                    with r_col2:
                        st.markdown("**Abstractive ROUGE**")
                        st.dataframe(rouge_scores_to_df(rouge_abs), use_container_width=True)

                    fig_rouge = plot_rouge_scores(
                        {"extractive": rouge_ext, "abstractive": rouge_abs}
                    )
                    st.pyplot(fig_rouge, use_container_width=True)
                    plt.close(fig_rouge)

                # ── Length comparison chart ───────────────────────────────────
                st.markdown("---")
                st.markdown("#### 📊 Summary Length Comparison")
                fig_len = plot_summary_length_comparison(user_text, ext_summary, abs_summary)
                st.pyplot(fig_len, use_container_width=True)

                st.download_button(
                    "⬇️ Download Chart (PNG)",
                    data=fig_to_bytes(fig_len),
                    file_name="length_comparison.png",
                    mime="image/png",
                    use_container_width=True,
                )
                plt.close(fig_len)

        else:
            st.info("👈 Enter your text and press **Generate Summary** to begin.")
