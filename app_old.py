"""
Sanskrit RAG — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sanskrit RAG",
    page_icon="🕉",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
}

/* ── Reset & Base ── */
.stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Main content padding ── */
.block-container {
  padding: 0 2.5rem 3rem 2.5rem !important;
  max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1.25rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: var(--text-soft) !important;
  font-size: 0.82rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--text) !important;
}

/* ── Typography ── */
h1, h2, h3 {
  font-family: 'Crimson Pro', serif !important;
  color: var(--text) !important;
  font-weight: 300 !important;
}
h1 { font-size: 2rem !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.2rem !important; }
p  { color: var(--text-soft); line-height: 1.7; font-size: 0.9rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border);
  gap: 0;
  padding: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.85rem 1.75rem !important;
  transition: color 0.2s, border-color 0.2s;
}
.stTabs [aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom-color: var(--gold) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  border: 1px solid var(--gold-dim) !important;
  color: var(--gold) !important;
  border-radius: 1px !important;
  padding: 0.55rem 1.4rem !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
  color: var(--bg) !important;
}
.stButton > button[kind="primary"] {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
  color: var(--bg) !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--gold-light) !important;
  border-color: var(--gold-light) !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 1px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
  border-color: var(--gold-dim) !important;
  box-shadow: 0 0 0 1px var(--gold-dim) !important;
}
.stTextArea textarea { line-height: 1.65 !important; }
label[data-testid="stWidgetLabel"] p {
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  font-weight: 500 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 2px !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--gold-dim) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
}

/* ── Progress ── */
.stProgress > div {
  background: var(--border) !important;
  border-radius: 1px !important;
  height: 2px !important;
}
.stProgress > div > div {
  background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
  border-radius: 1px !important;
  transition: width 0.4s ease !important;
}

/* ── Alerts ── */
[data-testid="stAlert"][data-baseweb="notification"] {
  border-radius: 1px !important;
  border-width: 0 0 0 2px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.83rem !important;
}
div[data-testid="stAlert"] > div {
  font-size: 0.83rem !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 1px !important;
  margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--text-soft) !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--gold) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Checkboxes ── */
.stCheckbox label p {
  color: var(--text-soft) !important;
  font-size: 0.83rem !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  font-weight: 400 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }
[data-testid="stStatusWidget"] { color: var(--gold) !important; }

/* ── Code ── */
code {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.77rem !important;
  background: var(--surface) !important;
  color: var(--gold-light) !important;
  border: 1px solid var(--border) !important;
  border-radius: 1px !important;
  padding: 1px 5px !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Custom UI Components
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Page Header */
.page-header {
  padding: 2.5rem 0 1.5rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
  display: flex;
  align-items: flex-end;
  gap: 1.25rem;
}
.page-header-glyph {
  font-family: 'Noto Serif Devanagari', serif;
  font-size: 3rem;
  color: var(--gold);
  opacity: 0.85;
  line-height: 1;
  flex-shrink: 0;
}
.page-header-text h1 {
  font-family: 'Crimson Pro', serif !important;
  font-size: 2.1rem !important;
  font-weight: 300 !important;
  color: var(--text) !important;
  margin: 0 0 0.2rem 0 !important;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.page-header-text .sub {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}

/* Section label */
.section-label {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* Sidebar key row */
.key-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.6rem;
  border-radius: 1px;
  margin-bottom: 0.25rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
}
.key-ok   { background: var(--success-bg); color: var(--success-fg); }
.key-miss { background: var(--error-bg);   color: var(--error-fg);   }
.key-dot  { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.key-ok   .key-dot { background: var(--success-fg); }
.key-miss .key-dot { background: var(--error-fg);   }

/* Sidebar model tag */
.model-tag {
  display: inline-block;
  background: var(--border-soft);
  border: 1px solid var(--border);
  color: var(--text-soft);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  padding: 2px 6px;
  border-radius: 1px;
  margin-top: 3px;
  word-break: break-all;
}

/* Pipeline step row */
.step-row {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  padding: 0.9rem 0;
  border-bottom: 1px solid var(--border-soft);
}
.step-row:last-child { border-bottom: none; }
.step-number {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  flex-shrink: 0;
  margin-top: 1px;
}
.step-number.active {
  border-color: var(--gold-dim);
  color: var(--gold);
  background: rgba(200, 146, 42, 0.08);
}
.step-number.done {
  border-color: var(--success-fg);
  color: var(--success-fg);
  background: var(--success-bg);
}
.step-info { flex: 1; }
.step-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.83rem;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 0.15rem;
}
.step-desc {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.75rem;
  color: var(--muted);
}

/* Chunk card */
.chunk-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 2px solid var(--gold-dim);
  border-radius: 0 1px 1px 0;
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.6rem;
}
.chunk-header {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.7rem;
}
.chunk-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--gold);
  background: rgba(200, 146, 42, 0.1);
  border: 1px solid var(--gold-dim);
  padding: 1px 7px;
  border-radius: 1px;
  letter-spacing: 0.05em;
}
.chunk-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
}
.chunk-source {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  margin-left: auto;
  opacity: 0.7;
}
.chunk-text {
  font-family: 'Noto Serif Devanagari', 'Crimson Pro', serif;
  font-size: 1rem;
  color: var(--text);
  line-height: 1.9;
}

/* Answer block */
.answer-wrap {
  background: var(--card);
  border: 1px solid var(--border);
  border-top: 1px solid var(--gold-dim);
  border-radius: 0 0 2px 2px;
  padding: 2rem 2rem 1.75rem 2rem;
}
.answer-label {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.answer-label::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 3px;
  background: var(--gold);
  border-radius: 50%;
}
.answer-text {
  font-family: 'Crimson Pro', serif;
  font-size: 1.12rem;
  color: var(--text);
  line-height: 1.95;
}

/* Query detail box */
.query-detail {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 1px;
  padding: 1rem 1.25rem;
  margin-top: 1rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
}
.query-detail-row { margin-bottom: 0.35rem; }
.query-detail-label { color: var(--muted); margin-right: 0.5rem; }
.query-detail-value { color: var(--text-soft); }

/* Stats row */
.stat-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 1px;
  overflow: hidden;
  margin-bottom: 1.5rem;
}
.stat-cell {
  background: var(--card);
  padding: 1rem 1.25rem;
}
.stat-label {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.2rem;
}
.stat-value {
  font-family: 'Crimson Pro', serif;
  font-size: 1.8rem;
  color: var(--gold);
  font-weight: 300;
  line-height: 1;
}

/* Ornament divider */
.orn-divider {
  text-align: center;
  font-family: 'Noto Serif Devanagari', serif;
  color: var(--gold-dim);
  font-size: 0.9rem;
  letter-spacing: 0.3em;
  margin: 1.25rem 0;
  opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# ── Lazy imports (only after page config) ─────────────────────────────────────
from pipeline.parser import parse_pdf
from pipeline.chunker import chunk_document
from pipeline.embedder import embed_texts, embed_query
from pipeline.vector_store import get_client, ensure_collection, upsert_chunks, hybrid_search
from pipeline.translator import translate_to_sanskrit
from pipeline.reranker import rerank
from pipeline.generator import generate_answer
from config import TOP_K_RETRIEVE, TOP_K_RERANK

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.25rem;">
      <div style="font-family: 'Noto Serif Devanagari', serif; font-size: 1.5rem; color: var(--gold); margin-bottom: 0.25rem;">संस्कृत RAG</div>
      <div style="font-family: 'DM Sans', sans-serif; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);">Knowledge Retrieval System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">API Keys</div>', unsafe_allow_html=True)

    keys = {
        "LLAMA_CLOUD_API_KEY": "LlamaParse",
        "QDRANT_URL":          "Qdrant Cloud",
        "QDRANT_API_KEY":      "Qdrant Key",
        "SARVAM_API_KEY":      "Sarvam AI",
        "GROQ_API_KEY":        "Groq",
        "OPENROUTER_API_KEY":  "OpenRouter",
    }
    for env_key, label in keys.items():
        ok = bool(os.getenv(env_key))
        cls = "key-ok" if ok else "key-miss"
        icon = "✓" if ok else "✗"
        st.markdown(
            f'<div class="key-row {cls}"><span class="key-dot"></span>{icon}&nbsp;{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)

    models_info = [
        ("Embedding",  "BAAI/bge-m3"),
        ("Reranker",   "bge-reranker-v2-m3"),
        ("Generator",  "compound-beta · Groq"),
        ("Fallback",   "gemma-3-27b · OpenRouter"),
        ("Vector DB",  "Qdrant Cloud"),
        ("Translation","Sarvam AI"),
    ]
    for label, tag in models_info:
        st.markdown(
            f"""<div style="margin-bottom:0.6rem;">
              <div style="font-size:0.68rem;color:var(--muted);letter-spacing:0.06em;text-transform:uppercase;font-family:'DM Sans',sans-serif;">{label}</div>
              <div class="model-tag">{tag}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-header-glyph">ॐ</div>
  <div class="page-header-text">
    <h1>Sanskrit RAG System</h1>
    <div class="sub">Parse · Embed · Retrieve · Answer</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ingest, tab_query = st.tabs(["Ingest Document", "Query"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop a Sanskrit PDF here",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;margin:0.75rem 0 1rem 0;">
              <div style="width:32px;height:32px;background:var(--card);border:1px solid var(--border);border-radius:1px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;">📄</div>
              <div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:var(--text);font-weight:500;">{uploaded_file.name}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--muted);">{uploaded_file.size // 1024} KB</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            ingest_btn = st.button("Start Ingestion", type="primary", use_container_width=True)

            if ingest_btn:
                progress_bar = st.progress(0)
                status_box   = st.empty()

                # ── Step 1: Save ──────────────────────────────────────────────
                status_box.info("Saving uploaded file…")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                progress_bar.progress(5)

                # ── Step 2: Parse ─────────────────────────────────────────────
                status_box.info("Starting Sanskrit OCR (browser will open briefly for Cloudflare)…")
                try:
                    def _ocr_status(msg: str):
                        status_box.info(msg)

                    pages     = parse_pdf(tmp_path, status_callback=_ocr_status)
                    full_text = "\n\n".join(p for p in pages if p)
                    progress_bar.progress(30)
                    status_box.success(f"OCR complete — {len(pages)} pages, {len(full_text):,} characters")
                except Exception as e:
                    status_box.error(f"Parse failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

                with st.expander("Raw parsed text — first 500 chars"):
                    st.text(full_text[:500] + ("…" if len(full_text) > 500 else ""))

                # ── Step 3: Chunk ─────────────────────────────────────────────
                status_box.info("Chunking by daṇḍa (॥) boundaries…")
                chunks = chunk_document(full_text, source=uploaded_file.name)
                progress_bar.progress(40)
                status_box.success(f"Created {len(chunks)} shloka-chunks")

                # Store for stats display
                st.session_state["last_chunks"] = len(chunks)
                st.session_state["last_pages"]  = len(pages)

                with st.expander(f"View all {len(chunks)} chunks"):
                    for i, c in enumerate(chunks):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<div class="chunk-header">'
                            f'<span class="chunk-num">CHUNK {i+1}</span>'
                            f'<span class="chunk-score">shlokas {c["shloka_indices"]}</span>'
                            f'</div>'
                            f'<div class="chunk-text">{c["text"][:220] + "…" if len(c["text"]) > 220 else c["text"]}</div>'
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
                status_box.success(f"Embedded {len(all_embeddings)} chunks (dense 1024-dim + sparse)")

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

                st.markdown('<div class="orn-divider">॥ &nbsp; ॥ &nbsp; ॥</div>', unsafe_allow_html=True)
                st.success("Ingestion complete. Switch to the Query tab to ask questions.")
                st.balloons()

    with col_right:
        st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)

        steps = [
            ("01", "Upload PDF", "Sanskrit manuscript, scripture, or commentary"),
            ("02", "LlamaParse OCR", "Devanagari-aware extraction, language='sa'"),
            ("03", "Daṇḍa Chunking", "Split on ॥ · group 2–4 shlokas with overlap"),
            ("04", "BGE-M3 Embed", "Dense 1024-dim + sparse lexical weights"),
            ("05", "Qdrant Upsert", "Hybrid-ready named vectors in the cloud"),
        ]
        steps_html = ""
        for num, title, desc in steps:
            steps_html += f"""
            <div class="step-row">
              <div class="step-number">{num}</div>
              <div class="step-info">
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>"""

        st.markdown(
            f'<div style="background:var(--card);border:1px solid var(--border);border-radius:1px;padding:0.75rem 1.25rem;">{steps_html}</div>',
            unsafe_allow_html=True,
        )

        if "last_chunks" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Last Run</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-cell">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{st.session_state.get("last_pages", "—")}</div>
              </div>
              <div class="stat-cell">
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{st.session_state.get("last_chunks", "—")}</div>
              </div>
              <div class="stat-cell">
                <div class="stat-label">Dim</div>
                <div class="stat-value">1024</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:

    col_q, col_opts = st.columns([5, 2], gap="large")

    with col_q:
        st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
        query = st.text_area(
            "question",
            placeholder="What does the text say about dharma and duty?\nआत्मा के बारे में क्या कहा गया है?\nॐ तत् सत्",
            height=120,
            label_visibility="collapsed",
        )
        search_btn = st.button("Search & Answer", type="primary", use_container_width=True)

    with col_opts:
        st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
        translate_query = st.checkbox(
            "Translate query to Sanskrit",
            value=True,
            help="Translates via Sarvam AI before embedding — improves recall on Sanskrit chunks",
        )
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        st.markdown(f"""
        <div style="margin-top:1rem;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--muted);">
          Retrieve top <span style="color:var(--gold);">{TOP_K_RETRIEVE}</span> →
          Rerank to <span style="color:var(--gold);">{TOP_K_RERANK}</span>
        </div>
        """, unsafe_allow_html=True)

    if search_btn and query.strip():
        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Step 1: Translate ─────────────────────────────────────────────────
        sanskrit_query = query
        if translate_query:
            with st.status("Translating query to Sanskrit via Sarvam AI…", expanded=False):
                sanskrit_query = translate_to_sanskrit(query)
                st.write(f"**Sanskrit query:** {sanskrit_query}")

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        with st.status("Embedding query with BGE-M3…", expanded=False):
            q_emb = embed_query(sanskrit_query)
            st.write("Dense + sparse vectors computed")

        # ── Step 3: Hybrid Search ─────────────────────────────────────────────
        with st.status(f"Hybrid search in Qdrant — top {TOP_K_RETRIEVE}…", expanded=False):
            qdrant_client = get_client()
            results = hybrid_search(
                qdrant_client,
                query_dense=q_emb["dense"].tolist(),
                query_sparse=q_emb["sparse"],
                top_k=TOP_K_RETRIEVE,
            )
            st.write(f"Retrieved {len(results)} candidates (RRF fusion)")

        if not results:
            st.warning("No results found. Have you ingested a document yet?")
            st.stop()

        # ── Step 4: Rerank ────────────────────────────────────────────────────
        with st.status(f"Reranking with BGE-reranker-v2-m3 — keeping top {TOP_K_RERANK}…", expanded=False):
            passages    = [r.payload["text"] for r in results]
            ranked      = rerank(sanskrit_query, passages, top_k=TOP_K_RERANK)
            top_results = [results[idx]  for idx, _ in ranked]
            top_scores  = [score         for _,   score in ranked]
            st.write(f"Top score: {top_scores[0]:.3f}")

        # ── Retrieved chunks ──────────────────────────────────────────────────
        if show_chunks:
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Retrieved Sources</div>', unsafe_allow_html=True)
            for i, (result, score) in enumerate(zip(top_results, top_scores), 1):
                source = result.payload.get("source", "unknown")
                text   = result.payload["text"]
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div class="chunk-header">'
                    f'<span class="chunk-num">CHUNK {i}</span>'
                    f'<span class="chunk-score">score {score:.3f}</span>'
                    f'<span class="chunk-source">{source}</span>'
                    f'</div>'
                    f'<div class="chunk-text">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Step 5: Generate ──────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Answer</div>', unsafe_allow_html=True)
        context_texts = [r.payload["text"] for r in top_results]

        with st.spinner("Generating answer…"):
            answer = generate_answer(query, context_texts)

        st.markdown(
            f'<div class="answer-wrap">'
            f'<div class="answer-label">Generated Response</div>'
            f'<div class="answer-text">{answer}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Query details ─────────────────────────────────────────────────────
        if translate_query and sanskrit_query != query:
            st.markdown(
                f'<div class="query-detail">'
                f'<div class="query-detail-row"><span class="query-detail-label">original &nbsp;</span><span class="query-detail-value">{query}</span></div>'
                f'<div class="query-detail-row"><span class="query-detail-label">sanskrit &nbsp;</span><span class="query-detail-value">{sanskrit_query}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    elif search_btn:
        st.warning("Please enter a question before searching.")
