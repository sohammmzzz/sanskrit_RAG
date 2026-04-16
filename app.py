"""
Sanskrit RAG — Streamlit App
Run: streamlit run app.py

Dependencies (pipeline/ must be in the same directory):
  pipeline/parser.py        — LlamaParse v1 agentic_plus multilingual parser
  pipeline/chunker.py       — Semantic Sanskrit / Hindi / English multilingual chunker
  pipeline/embedder.py
  pipeline/vector_store.py
  pipeline/translator.py
  pipeline/reranker.py
  pipeline/generator.py
  config.py

Requires: streamlit >= 1.36 (for @st.dialog)
"""

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from collections import Counter

load_dotenv()

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Sanskrit RAG",
    page_icon="🕉",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.toolbarMode", "viewer")


# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=Noto+Serif+Devanagari:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
  --bg:          #0b0a08;
  --surface:     #111009;
  --card:        #19180f;
  --card-hover:  #1f1e14;
  --border:      #2c2a1e;
  --border-soft: #221f14;
  --gold:        #c8922a;
  --gold-light:  #dba84a;
  --gold-dim:    #7a5a1a;
  --vermil:      #8b3030;
  --text:        #e2dcc8;
  --text-soft:   #b8b09a;
  --muted:       #6a6452;
  --sanskrit:    #d4a84a;
  --success-bg:  #0f1f12;
  --success-fg:  #5aad6a;
  --error-bg:    #1f0f0f;
  --error-fg:    #ad5a5a;
  --info-bg:     #0f1520;
  --info-fg:     #5a80ad;
  --hindi-fg:    #7aadad;
  --eng-fg:      #8a9aad;
}

/* ── Reset & Base ── */
.stApp { background: var(--bg) !important; color: var(--text); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Main content padding ── */
.block-container { padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--text-soft) !important; font-size: 0.82rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Crimson Pro', serif !important; color: var(--text) !important; font-weight: 300 !important; }
h1 { font-size: 2rem !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.2rem !important; }
p  { color: var(--text-soft); line-height: 1.7; font-size: 0.9rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border);
  gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem !important; font-weight: 500 !important;
  letter-spacing: 0.06em !important; text-transform: uppercase !important;
  color: var(--muted) !important; background: transparent !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  padding: 0.85rem 1.75rem !important; transition: color 0.2s, border-color 0.2s;
}
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important; font-weight: 500 !important;
  letter-spacing: 0.07em !important; text-transform: uppercase !important;
  background: transparent !important; border: 1px solid var(--gold-dim) !important;
  color: var(--gold) !important; border-radius: 1px !important;
  padding: 0.55rem 1.4rem !important; transition: all 0.15s !important;
}
.stButton > button:hover { background: var(--gold) !important; border-color: var(--gold) !important; color: var(--bg) !important; }
.stButton > button[kind="primary"] { background: var(--gold) !important; border-color: var(--gold) !important; color: var(--bg) !important; font-weight: 600 !important; }
.stButton > button[kind="primary"]:hover { background: var(--gold-light) !important; border-color: var(--gold-light) !important; }

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 1px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus { border-color: var(--gold-dim) !important; box-shadow: 0 0 0 1px var(--gold-dim) !important; }
.stTextArea textarea { line-height: 1.65 !important; }
label[data-testid="stWidgetLabel"] p {
  font-size: 0.75rem !important; letter-spacing: 0.06em !important;
  text-transform: uppercase !important; color: var(--muted) !important; font-weight: 500 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important; border: 1px dashed var(--border) !important;
  border-radius: 2px !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold-dim) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Progress ── */
.stProgress > div { background: var(--border) !important; border-radius: 1px !important; height: 2px !important; }
.stProgress > div > div {
  background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
  border-radius: 1px !important; transition: width 0.4s ease !important;
}

/* ── Alerts ── */
[data-testid="stAlert"][data-baseweb="notification"] {
  border-radius: 1px !important; border-width: 0 0 0 2px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 0.83rem !important;
}
div[data-testid="stAlert"] > div { font-size: 0.83rem !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 1px !important; margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
  color: var(--text-soft) !important; padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--gold) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Checkboxes ── */
.stCheckbox label p {
  color: var(--text-soft) !important; font-size: 0.83rem !important;
  letter-spacing: 0 !important; text-transform: none !important; font-weight: 400 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }
[data-testid="stStatusWidget"] { color: var(--gold) !important; }

/* ── Code ── */
code {
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.77rem !important;
  background: var(--surface) !important; color: var(--gold-light) !important;
  border: 1px solid var(--border) !important; border-radius: 1px !important;
  padding: 1px 5px !important;
}

/* ── Chat Interface ── */
.stChatMessage {
  background: transparent !important;
  padding: 0.4rem 0 !important;
}
[data-testid="stChatMessageContent"] {
  background: transparent !important;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 2px 2px 0 !important;
  padding: 0.85rem 1.1rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.92rem !important;
  color: var(--text) !important;
  margin: 0 !important;
}
/* Status / thinking widget */
[data-testid="stStatusContainer"],
div[class*="StatusWidget"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-left: 2px solid var(--gold-dim) !important;
  border-radius: 0 1px 1px 0 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
}
[data-testid="stStatusContainer"] p { font-size: 0.78rem !important; color: var(--text-soft) !important; }
/* Chat input */
[data-testid="stChatInput"] > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 1px !important;
}
[data-testid="stChatInput"] textarea {
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
  background: transparent !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--gold-dim) !important;
}
[data-testid="stChatInput"] button {
  background: var(--gold) !important;
  border: none !important;
  color: var(--bg) !important;
  border-radius: 1px !important;
}
[data-testid="stChatInput"] button:hover {
  background: var(--gold-light) !important;
}
/* Dialog modal */
[data-testid="stModal"] > div,
[data-testid="stDialog"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Custom UI Components
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Page Header */
.page-header {
  padding: 2.5rem 0 1.5rem 0; border-bottom: 1px solid var(--border);
  margin-bottom: 2rem; display: flex; align-items: flex-end; gap: 1.25rem;
}
.page-header-glyph {
  font-family: 'Noto Serif Devanagari', serif; font-size: 3rem;
  color: var(--gold); opacity: 0.85; line-height: 1; flex-shrink: 0;
}
.page-header-text h1 {
  font-family: 'Crimson Pro', serif !important; font-size: 2.1rem !important;
  font-weight: 300 !important; color: var(--text) !important;
  margin: 0 0 0.2rem 0 !important; letter-spacing: -0.02em; line-height: 1.1;
}
.page-header-text .sub {
  font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
}

/* Section label */
.section-label {
  font-family: 'DM Sans', sans-serif; font-size: 0.68rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.75rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Sidebar model tag */
.model-tag {
  display: inline-block; background: var(--border-soft); border: 1px solid var(--border);
  color: var(--text-soft); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
  padding: 2px 6px; border-radius: 1px; margin-top: 3px; word-break: break-all;
}

/* Pipeline step row */
.step-row {
  display: flex; align-items: flex-start; gap: 0.9rem;
  padding: 0.9rem 0; border-bottom: 1px solid var(--border-soft);
}
.step-row:last-child { border-bottom: none; }
.step-number {
  width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
  color: var(--muted); flex-shrink: 0; margin-top: 1px;
}
.step-number.done { border-color: var(--success-fg); color: var(--success-fg); background: var(--success-bg); }
.step-info { flex: 1; }
.step-title { font-family: 'DM Sans', sans-serif; font-size: 0.83rem; font-weight: 500; color: var(--text); margin-bottom: 0.15rem; }
.step-desc  { font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: var(--muted); }

/* Script-type badge */
.script-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 1px 6px; border-radius: 1px; border: 1px solid;
}
.script-sanskrit { color: var(--gold);     border-color: var(--gold-dim);  background: rgba(200,146,42,0.07); }
.script-hindi    { color: var(--hindi-fg); border-color: #2a5a5a;          background: rgba(100,180,180,0.07); }
.script-english  { color: var(--eng-fg);   border-color: #2a3a5a;          background: rgba(100,130,180,0.07); }
.script-mixed    { color: var(--muted);    border-color: var(--border);    background: transparent; }

/* Chunk card */
.chunk-card {
  background: var(--card); border: 1px solid var(--border);
  border-left: 2px solid var(--gold-dim); border-radius: 0 1px 1px 0;
  padding: 1.2rem 1.4rem; margin-bottom: 0.6rem;
}
.chunk-card.hindi   { border-left-color: #2a7a7a; }
.chunk-card.english { border-left-color: #2a3a7a; }
.chunk-card.mixed   { border-left-color: var(--border); }
.chunk-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.7rem; flex-wrap: wrap; }
.chunk-num {
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--gold);
  background: rgba(200,146,42,0.1); border: 1px solid var(--gold-dim);
  padding: 1px 7px; border-radius: 1px; letter-spacing: 0.05em;
}
.chunk-score  { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); }
.chunk-source { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); margin-left: auto; opacity: 0.7; }
.chunk-text   { font-family: 'Noto Serif Devanagari','Crimson Pro',serif; font-size: 1rem; color: var(--text); line-height: 1.9; }

/* Answer block */
.answer-wrap {
  background: var(--card); border: 1px solid var(--border);
  border-top: 1px solid var(--gold-dim); border-radius: 0 0 2px 2px;
  padding: 2rem 2rem 1.75rem 2rem;
}
.answer-label {
  font-family: 'DM Sans', sans-serif; font-size: 0.65rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--gold);
  margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
}
.answer-label::before { content: ''; display: inline-block; width: 3px; height: 3px; background: var(--gold); border-radius: 50%; }
.answer-text { font-family: 'Crimson Pro', serif; font-size: 1.12rem; color: var(--text); line-height: 1.95; }

/* Query detail box */
.query-detail {
  background: var(--surface); border: 1px solid var(--border); border-radius: 1px;
  padding: 1rem 1.25rem; margin-top: 1rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
}
.query-detail-row   { margin-bottom: 0.35rem; }
.query-detail-label { color: var(--muted); margin-right: 0.5rem; }
.query-detail-value { color: var(--text-soft); }

/* Stats row */
.stat-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--border); border: 1px solid var(--border);
  border-radius: 1px; overflow: hidden; margin-bottom: 1.5rem;
}
.stat-cell { background: var(--card); padding: 1rem 1.25rem; }
.stat-label { font-family: 'DM Sans', sans-serif; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.2rem; }
.stat-value { font-family: 'Crimson Pro', serif; font-size: 1.8rem; color: var(--gold); font-weight: 300; line-height: 1; }

/* Ornament divider */
.orn-divider {
  text-align: center; font-family: 'Noto Serif Devanagari', serif;
  color: var(--gold-dim); font-size: 0.9rem; letter-spacing: 0.3em;
  margin: 1.25rem 0; opacity: 0.5;
}

/* Chunk sidebar card */
.chunk-sidebar-card {
  background: var(--card); border: 1px solid var(--border);
  border-left: 2px solid var(--gold-dim); border-radius: 0 1px 1px 0;
  padding: 0.7rem 0.9rem; margin-bottom: 0.4rem; cursor: pointer;
  transition: border-left-color 0.15s, background 0.15s;
}
.chunk-sidebar-card:hover { background: var(--card-hover); border-left-color: var(--gold); }
.chunk-sidebar-card.hindi   { border-left-color: #2a7a7a; }
.chunk-sidebar-card.english { border-left-color: #2a3a7a; }

/* Chat empty state */
.chat-empty {
  text-align: center; padding: 5rem 2rem;
  pointer-events: none;
}
.chat-empty-glyph {
  font-family: 'Noto Serif Devanagari', serif;
  font-size: 4.5rem; color: var(--gold-dim); opacity: 0.35;
  line-height: 1; margin-bottom: 1.5rem;
}
.chat-empty-title {
  font-family: 'Crimson Pro', serif; font-size: 1.5rem; font-weight: 300;
  color: var(--text-soft); margin-bottom: 0.5rem;
}
.chat-empty-sub {
  font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
  letter-spacing: 0.07em; color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "page":               "ingest",
    "chat_history":       [],
    "pipeline_results":   None,
    "last_chunks":        None,
    "last_pages":         None,
    "last_script_counts": {},
    "last_chunks_data":   [],
    "translate_query":    True,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Pipeline imports (lazy — after page config) ───────────────────────────────
from pipeline.parser import parse_document
from pipeline.chunker import chunk_document
from pipeline.embedder import embed_texts, embed_query
from pipeline.vector_store import get_client, ensure_collection, upsert_chunks, hybrid_search
from pipeline.translator import translate_to_sanskrit
from pipeline.reranker import rerank
from pipeline.generator import generate_answer
from config import TOP_K_RETRIEVE, TOP_K_RERANK


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_LABELS = {
    "sanskrit": ("संस्कृत", "script-sanskrit"),
    "hindi":    ("हिन्दी",  "script-hindi"),
    "english":  ("English", "script-english"),
    "mixed":    ("Mixed",   "script-mixed"),
}

def _script_badge(script_type: str) -> str:
    label, cls = _SCRIPT_LABELS.get(script_type, ("?", "script-mixed"))
    return f'<span class="script-badge {cls}">{label}</span>'

def _card_class(script_type: str) -> str:
    return {"hindi": "hindi", "english": "english", "mixed": "mixed"}.get(script_type, "")


# ─────────────────────────────────────────────────────────────────────────────
#  Chunk Modal Dialog  (requires Streamlit >= 1.36)
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("Source Chunk", width="large")
def show_chunk_modal(text: str, script_type: str, source: str, score: float):
    badge = _script_badge(script_type)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap;">'
        f'  {badge}'
        f'  <code>score&nbsp;{score:.3f}</code>'
        f'  <code style="margin-left:auto;opacity:0.65;">{source}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:var(--border);margin:0.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:\'Noto Serif Devanagari\',\'Crimson Pro\',serif;'
        f'font-size:1.05rem;line-height:2.15;color:var(--text);padding:0.5rem 0;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ══════════════════════════════════════════════════════════════════════════
    #  INGEST SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state["page"] == "ingest":

        st.markdown("""
        <div style="padding:0.5rem 0 1.5rem 0;border-bottom:1px solid var(--border);margin-bottom:1.25rem;">
          <div style="font-family:'Noto Serif Devanagari',serif;font-size:1.5rem;color:var(--gold);margin-bottom:0.25rem;">संस्कृत RAG</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);">Knowledge Retrieval System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
        for _label, _tag in [
            ("Parser",      "agentic_plus · LlamaParse v1"),
            ("Embedding",   "BAAI/bge-m3"),
            ("Reranker",    "nvidia/llama-3.2-nv-rerankqa-1b-v2"),
            ("Generator",   "GPT OSS 120b · Groq"),
            ("Fallback",    "gemma-3-27b · OpenRouter"),
            ("Vector DB",   "Qdrant Cloud"),
            ("Translation", "Sarvam AI"),
        ]:
            st.markdown(
                f'<div style="margin-bottom:0.6rem;">'
                f'  <div style="font-size:0.68rem;color:var(--muted);letter-spacing:0.06em;'
                f'       text-transform:uppercase;font-family:\'DM Sans\',sans-serif;">{_label}</div>'
                f'  <div class="model-tag">{_tag}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Chunker</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.63rem;color:var(--muted);line-height:2.15;">
          <span style="color:var(--gold);font-size:0.68rem;letter-spacing:0.04em;">Semantic Chunker</span><br>
          ⎘&nbsp; atomic split (॥ · । · ¶)<br>
          ⇹&nbsp; context pad (±1 neighbor)<br>
          🧠 batch embed (NVIDIA NIM)<br>
          ∿&nbsp; semantic break (85th %ile dist)<br>
          ⌗&nbsp; sliding window cap (1200 chars)<br>
          ⚠&nbsp; fixed fallback (3-shloka ↔ 1 overlap)
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  QUERY SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # Back button
        if st.button("← Back to Ingest", use_container_width=True):
            st.session_state["page"] = "ingest"
            st.rerun()

        st.markdown('<br>', unsafe_allow_html=True)

        # Query options
        st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
        st.checkbox(
            "Translate query to Sanskrit",
            value=True,
            key="translate_query",
            help="Uses Sarvam AI to translate before embedding — improves Sanskrit chunk recall",
        )

        st.markdown('<br>', unsafe_allow_html=True)

        pr = st.session_state["pipeline_results"]

        if pr:
            # ── Pipeline cards ────────────────────────────────────────────────
            st.markdown('<div class="section-label">Pipeline · Last Run</div>', unsafe_allow_html=True)

            with st.expander("🔄  Translation", expanded=True):
                st.markdown(
                    f'<div class="query-detail" style="margin-top:0;">'
                    f'  <div class="query-detail-row">'
                    f'    <span class="query-detail-label">original&nbsp;</span>'
                    f'    <span class="query-detail-value">{pr.get("query", "")}</span>'
                    f'  </div>'
                    f'  <div class="query-detail-row">'
                    f'    <span class="query-detail-label">sanskrit&nbsp;</span>'
                    f'    <span class="query-detail-value">{pr.get("sanskrit_query", "")}</span>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("⚡  Embedding · BGE-M3", expanded=False):
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.71rem;'
                    'color:var(--muted);line-height:1.9;">'
                    'Dense + sparse vectors computed<br>'
                    'Dim: <span style="color:var(--gold);">1024</span> (dense)</div>',
                    unsafe_allow_html=True,
                )

            with st.expander(f"🔍  Hybrid Search · top {TOP_K_RETRIEVE}", expanded=False):
                _cnt = pr.get("search_count", 0)
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.71rem;'
                    f'color:var(--muted);line-height:1.9;">'
                    f'Retrieved <span style="color:var(--gold);">{_cnt}</span> candidates<br>'
                    f'RRF fusion applied</div>',
                    unsafe_allow_html=True,
                )

            with st.expander(f"📊  Reranking · top {TOP_K_RERANK}", expanded=False):
                _ts = pr.get("top_score", 0)
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.71rem;'
                    f'color:var(--muted);line-height:1.9;">'
                    f'NVIDIA NIM reranker<br>'
                    f'Top score: <span style="color:var(--gold);">{_ts:.3f}</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Retrieved Chunks ──────────────────────────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Retrieved Sources</div>', unsafe_allow_html=True)

            _top_results = pr.get("top_results", [])
            _top_scores  = pr.get("top_scores",  [])

            for _i, (_result, _score) in enumerate(zip(_top_results, _top_scores)):
                _payload = _result.payload
                _script  = _payload.get("script_type", "mixed")
                _label, _badge_cls = _SCRIPT_LABELS.get(_script, ("?", "script-mixed"))
                _preview = _payload["text"][:90].strip()

                with st.expander(f"Chunk {_i+1}  ·  {_score:.3f}", expanded=False):
                    st.markdown(
                        f'<span class="script-badge {_badge_cls}" style="margin-bottom:0.5rem;'
                        f'display:inline-block;">{_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="font-family:\'Noto Serif Devanagari\',\'Crimson Pro\',serif;'
                        f'font-size:0.82rem;color:var(--text-soft);line-height:1.8;margin:0.4rem 0 0.6rem 0;">'
                        f'{_preview}…</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "Open Full Chunk →",
                        key=f"open_chunk_{_i}",
                        use_container_width=True,
                    ):
                        show_chunk_modal(
                            _payload["text"],
                            _script,
                            _payload.get("source", "unknown"),
                            _score,
                        )

        else:
            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:var(--muted);
                        text-align:center;padding:2rem 0.75rem;border:1px dashed var(--border);
                        border-radius:1px;margin-top:0.5rem;line-height:1.9;">
              Pipeline details &amp; retrieved sources will appear here after your first query.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  INGEST PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["page"] == "ingest":

    st.markdown("""
    <div class="page-header">
      <div class="page-header-glyph">ॐ</div>
      <div class="page-header-text">
        <h1>Sanskrit RAG System</h1>
        <div class="sub">Parse · Chunk · Embed · Retrieve · Answer</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    # ── Left: upload + ingestion pipeline ────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop a PDF here",
            type=["pdf", "doc", "docx"],
            key="pdf_uploader",
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;margin:0.75rem 0 1rem 0;">'
                f'  <div style="width:32px;height:32px;background:var(--card);border:1px solid var(--border);'
                f'       border-radius:1px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;">📄</div>'
                f'  <div>'
                f'    <div style="font-family:\'DM Sans\',sans-serif;font-size:0.85rem;color:var(--text);'
                f'         font-weight:500;">{uploaded_file.name}</div>'
                f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:var(--muted);">'
                f'         {uploaded_file.size // 1024} KB</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            ingest_btn = st.button("Start Ingestion", type="primary", use_container_width=True)

            if ingest_btn:
                progress_bar = st.progress(0)
                status_box   = st.empty()

                # ── Step 1: Save ──────────────────────────────────────────────
                status_box.info("Saving uploaded file…")
                _suffix = Path(uploaded_file.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(suffix=_suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                progress_bar.progress(5)

                # ── Step 2: Parse ─────────────────────────────────────────────
                status_box.info("Starting LlamaParse v1 (agentic_plus · multilingual)…")
                try:
                    def _ocr_status(msg: str):
                        status_box.info(msg)

                    pages = parse_document(tmp_path, status_callback=_ocr_status)
                    full_text = "\n\n".join(p for p in pages if p.strip())
                    progress_bar.progress(30)
                    status_box.success(
                        f"Parse complete — {len(pages)} pages · {len(full_text):,} characters"
                    )
                except Exception as e:
                    status_box.error(f"Parse failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

                with st.expander("Raw parsed text — first 800 chars"):
                    st.text(full_text[:800] + ("…" if len(full_text) > 800 else ""))

                # ── Step 3: Chunk ─────────────────────────────────────────────
                status_box.info(
                    "Semantic chunking: atomic split → context pad → batch embed → semantic break…"
                )
                chunks = chunk_document(full_text, source=uploaded_file.name)
                progress_bar.progress(40)

                script_counts = Counter(c.get("script_type", "mixed") for c in chunks)
                status_box.success(
                    f"Created {len(chunks)} chunks  ·  "
                    f"Sanskrit {script_counts.get('sanskrit', 0)}  "
                    f"Hindi {script_counts.get('hindi', 0)}  "
                    f"English {script_counts.get('english', 0)}  "
                    f"Mixed {script_counts.get('mixed', 0)}"
                )

                st.session_state["last_chunks"]        = len(chunks)
                st.session_state["last_pages"]         = len(pages)
                st.session_state["last_script_counts"] = dict(script_counts)
                st.session_state["last_chunks_data"]   = chunks

                with st.expander(f"Preview all {len(chunks)} chunks"):
                    for i, c in enumerate(chunks):
                        script   = c.get("script_type", "mixed")
                        card_cls = _card_class(script)
                        preview  = c["text"][:260] + ("…" if len(c["text"]) > 260 else "")
                        meta = (
                            f'shlokas {c["shloka_indices"]}'
                            if c.get("shloka_indices")
                            else f'{c.get("char_count", len(c["text"]))} chars'
                        )
                        st.markdown(
                            f'<div class="chunk-card {card_cls}">'
                            f'  <div class="chunk-header">'
                            f'    <span class="chunk-num">CHUNK {i+1}</span>'
                            f'    {_script_badge(script)}'
                            f'    <span class="chunk-score">{meta}</span>'
                            f'  </div>'
                            f'  <div class="chunk-text">{preview}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Step 4: Embed ─────────────────────────────────────────────
                status_box.info("Loading BGE-M3 (first run: ~30 s download)…")
                embed_progress = st.progress(0)
                all_texts      = [c["text"] for c in chunks]
                batch_size     = 8
                all_embeddings = []

                for i in range(0, len(all_texts), batch_size):
                    batch = all_texts[i : i + batch_size]
                    all_embeddings.extend(embed_texts(batch))
                    frac = min(1.0, (i + batch_size) / len(all_texts))
                    embed_progress.progress(frac)
                    status_box.info(
                        f"Embedding chunks {i+1}–{min(i+batch_size, len(all_texts))} / {len(all_texts)}"
                    )

                embed_progress.progress(1.0)
                progress_bar.progress(75)
                status_box.success(
                    f"Embedded {len(all_embeddings)} chunks (dense 1024-dim + sparse)"
                )

                # ── Step 5: Upsert ────────────────────────────────────────────
                status_box.info("Upserting to Qdrant Cloud…")
                try:
                    qdrant_client = get_client()
                    ensure_collection(qdrant_client)
                    upsert_chunks(qdrant_client, chunks, all_embeddings)
                    progress_bar.progress(100)
                    status_box.success(
                        f"{len(chunks)} chunks stored in Qdrant · collection `sanskrit_texts`"
                    )
                except Exception as e:
                    status_box.error(f"Qdrant upsert failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

                os.unlink(tmp_path)

                st.markdown(
                    '<div class="orn-divider">॥ &nbsp; ॥ &nbsp; ॥</div>',
                    unsafe_allow_html=True,
                )
                st.success("Ingestion complete — switch to the Query interface to ask questions.")
                st.balloons()

    # ── Right: pipeline overview + last-run stats ─────────────────────────────
    with col_right:
        st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)

        steps_html = "".join(
            f'<div class="step-row">'
            f'  <div class="step-number">{num}</div>'
            f'  <div class="step-info">'
            f'    <div class="step-title">{title}</div>'
            f'    <div class="step-desc">{desc}</div>'
            f'  </div>'
            f'</div>'
            for num, title, desc in [
                ("01", "Upload Document",   "PDF, DOC, DOCX — Sanskrit, Hindi, or English"),
                ("02", "LlamaParse v1",     "agentic_plus tier · Devanagari custom prompt"),
                ("03", "Semantic Chunk",    "Atomic split → context pad → embed → semantic break"),
                ("04", "BGE-M3 Embed",      "Dense 1024-dim + sparse via NVIDIA NIM"),
                ("05", "Qdrant Upsert",     "Hybrid-ready named vectors in the cloud"),
            ]
        )
        st.markdown(
            f'<div style="background:var(--card);border:1px solid var(--border);'
            f'border-radius:1px;padding:0.75rem 1.25rem;">{steps_html}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("last_chunks") is not None:
            sc = st.session_state.get("last_script_counts", {})
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Last Run</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="stat-row">'
                f'  <div class="stat-cell"><div class="stat-label">Pages</div>'
                f'    <div class="stat-value">{st.session_state.get("last_pages", "—")}</div></div>'
                f'  <div class="stat-cell"><div class="stat-label">Chunks</div>'
                f'    <div class="stat-value">{st.session_state.get("last_chunks", "—")}</div></div>'
                f'  <div class="stat-cell"><div class="stat-label">Dim</div>'
                f'    <div class="stat-value">1024</div></div>'
                f'  <div class="stat-cell"><div class="stat-label">Scripts</div>'
                f'    <div class="stat-value">{len([k for k, v in sc.items() if v > 0])}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Script Mix section intentionally removed

    # ── Navigate to Query ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="orn-divider">· · ·</div>', unsafe_allow_html=True)
    _, _cta_col, _ = st.columns([2, 3, 2])
    with _cta_col:
        if st.button("Open Query Interface →", type="primary", use_container_width=True):
            st.session_state["page"] = "query"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY / CHATBOT PAGE
# ─────────────────────────────────────────────────────────────────────────────
else:

    # Compact header
    st.markdown("""
    <div style="padding:1.25rem 0 1rem 0;border-bottom:1px solid var(--border);
                margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between;">
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <div style="font-family:'Noto Serif Devanagari',serif;font-size:1.7rem;
                    color:var(--gold);line-height:1;flex-shrink:0;">ॐ</div>
        <div>
          <div style="font-family:'Crimson Pro',serif;font-size:1.35rem;font-weight:300;
                      color:var(--text);margin:0;line-height:1.2;">Sanskrit RAG</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:0.67rem;letter-spacing:0.12em;
                      text-transform:uppercase;color:var(--muted);">Query Interface</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear chat
    _clr_col, _ = st.columns([1, 7])
    with _clr_col:
        if st.button("Clear Chat", key="clear_chat_btn"):
            st.session_state["chat_history"]     = []
            st.session_state["pipeline_results"] = None
            st.rerun()

    # ── Render chat history ───────────────────────────────────────────────────
    for _msg in st.session_state["chat_history"]:
        with st.chat_message(_msg["role"]):
            if _msg["role"] == "assistant":
                st.markdown(
                    f'<div class="answer-wrap">'
                    f'  <div class="answer-label">Generated Response</div>'
                    f'  <div class="answer-text">{_msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(_msg["content"])

    # ── Empty state ───────────────────────────────────────────────────────────
    if not st.session_state["chat_history"]:
        st.markdown("""
        <div class="chat-empty">
          <div class="chat-empty-glyph">॥</div>
          <div class="chat-empty-title">Ask anything about the ingested texts</div>
          <div class="chat-empty-sub">Sanskrit · Hindi · English queries all supported</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    if _query := st.chat_input(
        "Ask about the texts — Sanskrit, Hindi, or English…",
        key="main_chat_input",
    ):
        st.session_state["chat_history"].append({"role": "user", "content": _query})

        with st.chat_message("user"):
            st.markdown(_query)

        _translate_q   = st.session_state.get("translate_query", True)
        _pipeline_log  = {"query": _query}

        with st.chat_message("assistant"):

            with st.status("Thinking…", expanded=True) as _proc_status:

                # ── 1. Translate ──────────────────────────────────────────────
                if _translate_q:
                    st.write("🔄 Translating query to Sanskrit via Sarvam AI…")
                    try:
                        _sanskrit_query = translate_to_sanskrit(_query)
                    except Exception as _te:
                        st.write(f"⚠ Translation unavailable ({_te}), using original")
                        _sanskrit_query = _query
                    st.write(f"Sanskrit query: `{_sanskrit_query}`")
                else:
                    _sanskrit_query = _query

                _pipeline_log["sanskrit_query"] = _sanskrit_query

                # ── 2. Embed ──────────────────────────────────────────────────
                st.write("⚡ Embedding query with BGE-M3…")
                _q_emb = embed_query(_sanskrit_query)
                st.write("Dense + sparse vectors computed")

                # ── 3. Hybrid search ──────────────────────────────────────────
                st.write(f"🔍 Hybrid search in Qdrant — top {TOP_K_RETRIEVE}…")
                try:
                    _qdrant_client = get_client()
                    _results = hybrid_search(
                        _qdrant_client,
                        query_dense=_q_emb["dense"],
                        query_sparse=_q_emb["sparse"],
                        top_k=TOP_K_RETRIEVE,
                    )
                    st.write(f"Retrieved {len(_results)} candidates (RRF fusion)")
                    _pipeline_log["search_count"] = len(_results)
                except Exception as _se:
                    st.write(f"⚠ Search failed: {_se}")
                    _results = []
                    _pipeline_log["search_count"] = 0

                if not _results:
                    _proc_status.update(
                        label="No results — have you ingested a document?",
                        state="error",
                        expanded=True,
                    )
                    st.stop()

                # ── 4. Rerank ─────────────────────────────────────────────────
                st.write(f"📊 Reranking with NVIDIA NIM — keeping top {TOP_K_RERANK}…")
                _passages    = [r.payload["text"] for r in _results]
                _ranked      = rerank(_sanskrit_query, _passages, top_k=TOP_K_RERANK)
                _top_results = [_results[idx]  for idx, _ in _ranked]
                _top_scores  = [score          for _,   score in _ranked]
                st.write(f"Top score: {_top_scores[0]:.3f}")

                _pipeline_log.update({
                    "top_score":   _top_scores[0] if _top_scores else 0,
                    "top_results": _top_results,
                    "top_scores":  _top_scores,
                })

                _proc_status.update(
                    label="✓ Pipeline complete",
                    state="complete",
                    expanded=False,
                )

            # ── 5. Generate answer ────────────────────────────────────────────
            _context_texts = [r.payload["text"] for r in _top_results]
            with st.spinner("Generating answer…"):
                _answer = generate_answer(_query, _context_texts)

            st.markdown(
                f'<div class="answer-wrap">'
                f'  <div class="answer-label">Generated Response</div>'
                f'  <div class="answer-text">{_answer}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Persist & refresh ─────────────────────────────────────────────────
        st.session_state["chat_history"].append({"role": "assistant", "content": _answer})
        st.session_state["pipeline_results"] = _pipeline_log
        # Rerun to update the sidebar with the new pipeline data
        st.rerun()
