

## 🎯 Project Goal

Build a **Streamlit web application** for a Sanskrit RAG (Retrieval-Augmented Generation) pipeline. The app must allow a user to:

1. **Upload a PDF** (Sanskrit manuscript, scripture, or text)
2. Watch it get **parsed → chunked → embedded → stored in Qdrant** live in the UI with progress feedback
3. **Query the knowledge base** in English, Hindi, Hinglish, or Sanskrit
4. Receive a **grounded answer** with retrieved source chunks displayed

---

## 📦 Complete Tech Stack (All Free-Tier)

| Component | Tool | Free Tier |
|---|---|---|
| **PDF Parser** | LlamaParse v2 (`llama-parse` Python SDK) | 1,000 pages/day free |
| **Sentence Boundary** | Custom daṇḍa-splitter (no external model needed) | Free |
| **Embedding Model** | `BAAI/bge-m3` via `FlagEmbedding` | Free, runs in-memory (~1.1 GB RAM) |
| **Vector DB** | Qdrant Cloud | Free 1GB cluster |
| **Query Translation** | Sarvam AI `sarvam-translate:v1` | Free ₹1000 credits |
| **Generator LLM** | Groq `openai/gpt-oss-120b` | Free tier (14,400 req/day) |
| **Fallback LLM** | OpenRouter `google/gemma-4-31b-it:free` | Free (200 req/day) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` via `FlagEmbedding` | Free, runs in-memory |
| **UI** | Streamlit | Free |

---

## 🔑 API Keys Needed (All Free to Get)

```
LLAMA_CLOUD_API_KEY     → https://cloud.llamaindex.ai/api-key
QDRANT_URL              → https://cloud.qdrant.io (free cluster URL)
QDRANT_API_KEY          → from Qdrant Cloud dashboard
SARVAM_API_KEY          → https://dashboard.sarvam.ai
GROQ_API_KEY            → https://console.groq.com/keys
OPENROUTER_API_KEY      → https://openrouter.ai/keys
```

Store all keys in a `.env` file and load with `python-dotenv`.

---

## 📁 Project Structure

```
sanskrit_rag/
├── .env                    # All API keys
├── requirements.txt
├── app.py                  # Streamlit entry point
├── config.py               # Constants, model names, collection settings
├── pipeline/
│   ├── __init__.py
│   ├── parser.py           # LlamaParse integration
│   ├── chunker.py          # Daṇḍa-aware chunking logic
│   ├── embedder.py         # BGE-M3 embedding wrapper
│   ├── vector_store.py     # Qdrant upsert/search
│   ├── translator.py       # Sarvam AI query translation
│   ├── reranker.py         # BGE-reranker-v2-m3
│   └── generator.py        # Groq / OpenRouter LLM calls
└── utils/
    ├── __init__.py
    └── helpers.py          # Language detection, text cleaning
```

---

## 📦 requirements.txt

```
streamlit>=1.35.0
llama-parse>=0.6.0
FlagEmbedding>=1.2.11
qdrant-client>=1.9.0
groq>=0.9.0
openai>=1.30.0         # used for OpenRouter (OpenAI-compatible)
requests>=2.31.0
python-dotenv>=1.0.0
langdetect>=1.0.9
torch>=2.0.0           # required by FlagEmbedding
transformers>=4.40.0
```

> **Note on BGE-M3 memory**: At fp16, the model is ~1.1 GB RAM on CPU. On first run it downloads from HuggingFace (~570 MB). Use `use_fp16=True` to halve memory. If the deployment machine has a GPU, it will auto-use it.

---

## 📚 Key Documentation References

Your LLM **must read** these before implementing each module:

### LlamaParse
- **Python SDK quickstart**: https://pypi.org/project/llama-parse/
- **REST API v2 guide**: https://developers.llamaindex.ai/python/cloud/llamaparse/api-v2-guide/
- **Getting started (tiers)**: https://docs.cloud.llamaindex.ai/llamaparse/getting_started
- Use `tier="cost_effective"` for Sanskrit PDFs (preserves structure better than `fast`)
- Pass `language="sa"` for Sanskrit Devanagari OCR

### BGE-M3 (FlagEmbedding)
- **HuggingFace model card**: https://huggingface.co/BAAI/bge-m3
- **FlagEmbedding library**: https://github.com/FlagOpen/FlagEmbedding
- Install: `pip install FlagEmbedding`
- Model ID: `BAAI/bge-m3`
- Always encode with `return_dense=True, return_sparse=True` for hybrid search
- Output vector dimension: **1024** (dense), plus sparse lexical weights dict

### BGE Reranker
- **HuggingFace model card**: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Install: same `FlagEmbedding` library
- Use `FlagReranker` class with `use_fp16=True`

### Qdrant Cloud
- **Python client docs**: https://python-client.qdrant.tech/
- **Quickstart**: https://qdrant.tech/documentation/quickstarts/
- **Hybrid search (dense + sparse)**: https://qdrant.tech/documentation/concepts/hybrid-queries/
- **Named vectors**: https://qdrant.tech/documentation/concepts/collections/#collection-with-multiple-vectors
- Collection needs **two named vectors**: `dense` (1024 dim, Cosine) and `sparse` (for BGE-M3 sparse output)

### Sarvam AI Translation
- **API docs**: https://docs.sarvam.ai/api-reference-docs/translate
- **Dashboard / API key**: https://dashboard.sarvam.ai
- Endpoint: `POST https://api.sarvam.ai/translate`
- Use `source_language_code: "auto"` and `target_language_code: "sa-IN"` for Sanskrit
- Supports Hinglish (mixed Hindi-English) natively

### Groq (Generator LLM)
- **API reference**: https://console.groq.com/docs/api-reference
- **Models list**: https://console.groq.com/docs/models
- **Python SDK**: https://github.com/groq/groq-python
- Primary model: `openai/gpt-oss-120b` — **verify the exact model ID string against https://console.groq.com/docs/models before coding**, as Groq periodically updates model slugs
- Fast fallback within Groq (if needed): `llama-3.1-8b-instant`
- Endpoint is OpenAI-compatible: `https://api.groq.com/openai/v1`

### OpenRouter (Fallback LLM)
- **Quickstart**: https://openrouter.ai/docs/quickstart
- **Free models**: https://openrouter.ai/collections/free-models
- **Models API**: https://openrouter.ai/docs/api-reference/list-models
- Base URL: `https://openrouter.ai/api/v1` (OpenAI-compatible)
- Fallback model for this project: `google/gemma-4-31b-it:free`
- Other free alternatives if needed:
  - `meta-llama/llama-3.3-70b-instruct:free`
  - `deepseek/deepseek-r1:free`

---

## 🏗️ Implementation Details

### 1. `config.py`

```python
COLLECTION_NAME = "sanskrit_texts"
DENSE_DIM = 1024          # BGE-M3 dense output
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATOR_MODEL = "openai/gpt-oss-120b"              # Groq — verify slug at console.groq.com/docs/models
FALLBACK_MODEL  = "google/gemma-4-31b-it:free"  # OpenRouter
TOP_K_RETRIEVE = 10       # retrieve this many candidates
TOP_K_RERANK = 3          # keep this many after reranking
CHUNK_MIN_SHLOKAS = 2
CHUNK_MAX_SHLOKAS = 4
CHUNK_OVERLAP_SHLOKAS = 1
```

---

### 2. `pipeline/parser.py` — LlamaParse Integration

```python
"""
Parse a PDF file using LlamaParse v2 Python SDK.

Docs: https://pypi.org/project/llama-parse/
      https://docs.cloud.llamaindex.ai/llamaparse/getting_started

Usage:
    from pipeline.parser import parse_pdf
    pages_markdown = parse_pdf("path/to/file.pdf")
    # Returns: list of strings, one per page, in Markdown format
"""

from llama_parse import LlamaParse
import os

def parse_pdf(file_path: str) -> list[str]:
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        language="sa",           # Sanskrit Devanagari
        verbose=True,
    )
    documents = parser.load_data(file_path)
    return [doc.text for doc in documents]
```

---

### 3. `pipeline/chunker.py` — Daṇḍa-Aware Sanskrit Chunking

**This is the most critical module. Standard text splitters break Sanskrit.**

Sanskrit uses:
- `।` (U+0964) — single daṇḍa — equivalent to a comma/period within a verse
- `॥` (U+0965) — double daṇḍa — verse/shloka boundary (full stop)

```python
"""
Daṇḍa-aware chunker for Sanskrit texts.

Strategy:
  1. Split on ॥ (double daṇḍa) to isolate individual shlokas
  2. Group CHUNK_MIN to CHUNK_MAX shlokas into a chunk
  3. Slide with CHUNK_OVERLAP shloka overlap between consecutive chunks

Returns list of dicts:
    {
        "text": str,
        "shloka_indices": list[int],
        "source": str,
    }
"""

import re
from config import CHUNK_MIN_SHLOKAS, CHUNK_MAX_SHLOKAS, CHUNK_OVERLAP_SHLOKAS

DOUBLE_DANDA = "\u0965"
SINGLE_DANDA = "\u0964"

def split_into_shlokas(text: str) -> list[str]:
    parts = re.split(f"({re.escape(DOUBLE_DANDA)})", text)
    shlokas = []
    i = 0
    while i < len(parts):
        segment = parts[i].strip()
        if i + 1 < len(parts) and parts[i + 1] == DOUBLE_DANDA:
            segment += DOUBLE_DANDA
            i += 2
        else:
            i += 1
        if segment and segment != DOUBLE_DANDA:
            shlokas.append(segment)
    return shlokas

def chunk_shlokas(
    shlokas: list[str],
    source: str,
    min_size: int = CHUNK_MIN_SHLOKAS,
    max_size: int = CHUNK_MAX_SHLOKAS,
    overlap: int = CHUNK_OVERLAP_SHLOKAS,
) -> list[dict]:
    chunks = []
    i = 0
    while i < len(shlokas):
        group = shlokas[i : i + max_size]
        chunk_text = " ".join(group).strip()
        chunks.append({
            "text": chunk_text,
            "shloka_indices": list(range(i, i + len(group))),
            "source": source,
        })
        i += max(1, max_size - overlap)
    return chunks

def chunk_document(full_text: str, source: str) -> list[dict]:
    shlokas = split_into_shlokas(full_text)
    if not shlokas:
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        shlokas = lines
    return chunk_shlokas(shlokas, source)
```

---

### 4. `pipeline/embedder.py` — BGE-M3 In-Memory Embedding

```python
"""
BGE-M3 embedding using FlagEmbedding library.

Docs: https://huggingface.co/BAAI/bge-m3
      https://github.com/FlagOpen/FlagEmbedding

Model: BAAI/bge-m3
  - 568M params, ~1.1 GB in fp16
  - Returns: dense (1024-dim) + sparse (dict of token→weight)
  - Supports 100+ languages including Sanskrit/Devanagari
"""

from FlagEmbedding import BGEM3FlagModel
import numpy as np

_model = None

def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model

def embed_texts(texts: list[str]) -> list[dict]:
    model = get_model()
    outputs = model.encode(
        texts,
        batch_size=8,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for i in range(len(texts)):
        dense_vec = outputs["dense_vecs"][i]
        sparse_weights = outputs["lexical_weights"][i]
        sparse_named = model.convert_id_to_token(sparse_weights)
        results.append({"dense": dense_vec, "sparse": sparse_named})
    return results

def embed_query(query: str) -> dict:
    return embed_texts([query])[0]
```

---

### 5. `pipeline/vector_store.py` — Qdrant Cloud Integration

```python
"""
Qdrant Cloud vector store.

Docs:
  Python client: https://python-client.qdrant.tech/
  Quickstart:    https://qdrant.tech/documentation/quickstarts/
  Named vectors: https://qdrant.tech/documentation/concepts/collections/#collection-with-multiple-vectors
  Hybrid search: https://qdrant.tech/documentation/concepts/hybrid-queries/

Collection schema:
  - Named vector "dense": size=1024, distance=Cosine
  - Named vector "sparse": SparseVectorParams
  - Payload: { text, source, shloka_indices }
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector,
    FusionQuery, Prefetch, ScoredPoint,
)
import os
import uuid
from config import COLLECTION_NAME, DENSE_DIM

def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

def upsert_chunks(client: QdrantClient, chunks: list[dict], embeddings: list[dict]):
    points = []
    for chunk, emb in zip(chunks, embeddings):
        sparse_indices = []
        sparse_values = []
        for token, weight in emb["sparse"].items():
            sparse_indices.append(abs(hash(token)) % (10**6))
            sparse_values.append(float(weight))

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": emb["dense"].tolist(),
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "shloka_indices": chunk["shloka_indices"],
            },
        )
        points.append(point)

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + batch_size])

def hybrid_search(
    client: QdrantClient,
    query_dense: list[float],
    query_sparse: dict,
    top_k: int = 10,
) -> list[ScoredPoint]:
    sparse_indices = [abs(hash(t)) % (10**6) for t in query_sparse]
    sparse_values = [float(w) for w in query_sparse.values()]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=query_dense, using="dense", limit=top_k * 2),
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
        with_payload=True,
    )
    return results.points
```

---

### 6. `pipeline/translator.py` — Sarvam AI Query Translation

```python
"""
Query translation: English/Hindi/Hinglish → Sanskrit
using Sarvam AI sarvam-translate:v1

Docs: https://docs.sarvam.ai/api-reference-docs/translate
"""

import requests
import os

SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"

def translate_to_sanskrit(query: str) -> str:
    headers = {
        "api-subscription-key": os.getenv("SARVAM_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {
        "input": query,
        "source_language_code": "auto",
        "target_language_code": "sa-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "enable_preprocessing": True,
    }
    try:
        resp = requests.post(SARVAM_ENDPOINT, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("translated_text", query)
    except Exception as e:
        print(f"[Sarvam translation failed]: {e}")
        return query
```

---

### 7. `pipeline/reranker.py` — BGE Reranker

```python
"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3

Docs: https://huggingface.co/BAAI/bge-reranker-v2-m3
"""

from FlagEmbedding import FlagReranker

_reranker = None

def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker

def rerank(query: str, passages: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    reranker = get_reranker()
    pairs = [[query, p] for p in passages]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

---

### 8. `pipeline/generator.py` — LLM Answer Generation

```python
"""
Answer generation using Groq (primary) with OpenRouter fallback.

Groq docs:      https://console.groq.com/docs/api-reference
Groq Python:    https://github.com/groq/groq-python
OpenRouter:     https://openrouter.ai/docs/quickstart

Primary model  (Groq):       openai/gpt-oss-120b
  → Verify the exact model slug at https://console.groq.com/docs/models
    before running; Groq updates slugs periodically.

Fallback model (OpenRouter): google/gemma-4-31b-it:free
  → Listed at https://openrouter.ai/collections/free-models

Both endpoints are OpenAI-compatible.
  Groq:       base_url = "https://api.groq.com/openai/v1"
  OpenRouter: base_url = "https://openrouter.ai/api/v1"
"""

from openai import OpenAI
import os
from config import GENERATOR_MODEL, FALLBACK_MODEL

SYSTEM_PROMPT = """You are a Sanskrit scholar assistant. You answer questions about Sanskrit texts
based only on the provided context chunks. If the context does not contain the answer, say so.
Always cite which chunk your answer is based on.
When quoting Sanskrit, preserve the Devanagari script exactly.
Answer in the same language the question was asked in."""

def build_context_string(chunks: list[str]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}]:\n{chunk}")
    return "\n\n---\n\n".join(parts)

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = build_context_string(context_chunks)
    user_message = f"""Context from the Sanskrit text:

{context}

---

Question: {query}

Please answer based on the context above."""

    # ── Primary: Groq ──
    try:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,   # openai/gpt-oss-120b
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as groq_err:
        print(f"[Groq failed, trying OpenRouter]: {groq_err}")

    # ── Fallback: OpenRouter ──
    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=FALLBACK_MODEL,    # google/gemma-4-31b-it:free
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as or_err:
        return f"[Generation failed. Groq: {groq_err}. OpenRouter: {or_err}]"
```

---

### 9. `app.py` — Streamlit UI

Two-tab interface: **"📤 Ingest Document"** and **"🔍 Query"**.

```python
"""
Sanskrit RAG — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

from pipeline.parser import parse_pdf
from pipeline.chunker import chunk_document
from pipeline.embedder import embed_texts, embed_query
from pipeline.vector_store import get_client, ensure_collection, upsert_chunks, hybrid_search
from pipeline.translator import translate_to_sanskrit
from pipeline.reranker import rerank
from pipeline.generator import generate_answer
from config import TOP_K_RETRIEVE, TOP_K_RERANK

st.set_page_config(page_title="Sanskrit RAG", page_icon="🕉️", layout="wide")
st.title("🕉️ Sanskrit RAG System")
st.caption("Parse Sanskrit PDFs, chunk by shloka, embed with BGE-M3, query in any language.")

# ── Sidebar ──
with st.sidebar:
    st.header("🔑 Configuration")
    keys = {
        "LLAMA_CLOUD_API_KEY": "LlamaParse",
        "QDRANT_URL":          "Qdrant Cloud",
        "QDRANT_API_KEY":      "Qdrant Key",
        "SARVAM_API_KEY":      "Sarvam AI",
        "GROQ_API_KEY":        "Groq",
        "OPENROUTER_API_KEY":  "OpenRouter",
    }
    for env_key, label in keys.items():
        if os.getenv(env_key):
            st.success(f"✓ {label}")
        else:
            st.error(f"✗ {label} missing")

    st.divider()
    st.markdown("**Embedding model:** BAAI/bge-m3")
    st.markdown("**Generator:** openai/gpt-oss-120b (Groq)")
    st.markdown("**Fallback:** google/gemma-4-31b-it (OpenRouter)")
    st.markdown("**Vector DB:** Qdrant Cloud")

tab_ingest, tab_query = st.tabs(["📤 Ingest Document", "🔍 Query"])

# ══ TAB 1 — INGEST ══
with tab_ingest:
    st.subheader("Upload a Sanskrit PDF")
    st.info("Supported: PDF files with Devanagari script (Sanskrit manuscripts, scriptures, commentaries)")

    uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], key="pdf_uploader")

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size // 1024} KB)")
        with col2:
            ingest_btn = st.button("🚀 Start Ingestion", type="primary", use_container_width=True)

        if ingest_btn:
            progress_bar = st.progress(0)
            status_box = st.empty()

            # Step 1 — Save temp file
            status_box.info("💾 Saving uploaded file...")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            progress_bar.progress(5)

            # Step 2 — Parse
            status_box.info("📄 Parsing PDF with LlamaParse (OCR + Devanagari)... this may take 20–60s")
            try:
                pages = parse_pdf(tmp_path)
                full_text = "\n\n".join(pages)
                progress_bar.progress(30)
                status_box.success(f"✅ Parsed {len(pages)} pages ({len(full_text):,} characters)")
            except Exception as e:
                status_box.error(f"❌ Parse failed: {e}")
                st.stop()

            with st.expander("📜 Raw parsed text (preview first 500 chars)"):
                st.text(full_text[:500] + "…" if len(full_text) > 500 else full_text)

            # Step 3 — Chunk
            status_box.info("✂️ Chunking by daṇḍa (॥) boundaries...")
            chunks = chunk_document(full_text, source=uploaded_file.name)
            progress_bar.progress(40)
            status_box.success(f"✅ Created {len(chunks)} chunks")

            with st.expander(f"📦 View all {len(chunks)} chunks"):
                for i, c in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}** (shlokas {c['shloka_indices']}):")
                    st.text(c["text"][:200] + "…" if len(c["text"]) > 200 else c["text"])
                    st.divider()

            # Step 4 — Embed
            status_box.info("🧮 Embedding with BGE-M3 (loading model on first run ~30s)...")
            embed_progress = st.progress(0)
            all_texts = [c["text"] for c in chunks]
            batch_size = 8
            all_embeddings = []

            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i : i + batch_size]
                all_embeddings.extend(embed_texts(batch))
                frac = min(1.0, (i + batch_size) / len(all_texts))
                embed_progress.progress(frac)
                status_box.info(
                    f"🧮 Embedding chunks {i+1}–{min(i+batch_size, len(all_texts))} / {len(all_texts)}"
                )

            embed_progress.progress(1.0)
            progress_bar.progress(75)
            status_box.success(f"✅ Embedded {len(all_embeddings)} chunks")

            # Step 5 — Upsert
            status_box.info("☁️ Upserting to Qdrant Cloud...")
            try:
                client = get_client()
                ensure_collection(client)
                upsert_chunks(client, chunks, all_embeddings)
                progress_bar.progress(100)
                status_box.success(
                    f"✅ Done! {len(chunks)} chunks stored in Qdrant collection `sanskrit_texts`"
                )
            except Exception as e:
                status_box.error(f"❌ Qdrant upsert failed: {e}")

            os.unlink(tmp_path)
            st.balloons()


# ══ TAB 2 — QUERY ══
with tab_query:
    st.subheader("Ask about your Sanskrit texts")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query = st.text_area(
            "Enter your question (English, Hindi, Hinglish, or Sanskrit):",
            placeholder="What does the Bhagavad Gita say about duty?",
            height=80,
        )
    with col_opts:
        translate_query = st.checkbox(
            "🔁 Translate query to Sanskrit", value=True,
            help="Translates your query via Sarvam AI before embedding — improves recall on Sanskrit chunks"
        )
        show_chunks = st.checkbox("Show retrieved chunks", value=True)

    search_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

    if search_btn and query.strip():
        with st.spinner("Searching..."):

            # Step 1 — Translate
            sanskrit_query = query
            if translate_query:
                with st.status("🔁 Translating query to Sanskrit via Sarvam AI..."):
                    sanskrit_query = translate_to_sanskrit(query)
                    st.write(f"**Sanskrit query:** {sanskrit_query}")

            # Step 2 — Embed
            with st.status("🧮 Embedding query with BGE-M3..."):
                q_emb = embed_query(sanskrit_query)
                st.write("Query embedded (dense + sparse)")

            # Step 3 — Search
            with st.status(f"☁️ Searching Qdrant (top {TOP_K_RETRIEVE})..."):
                qdrant_client = get_client()
                results = hybrid_search(
                    qdrant_client,
                    query_dense=q_emb["dense"].tolist(),
                    query_sparse=q_emb["sparse"],
                    top_k=TOP_K_RETRIEVE,
                )
                st.write(f"Retrieved {len(results)} candidates")

            if not results:
                st.warning("No results found. Have you ingested a document yet?")
                st.stop()

            # Step 4 — Rerank
            with st.status(f"📊 Reranking with BGE-reranker-v2-m3 → keep top {TOP_K_RERANK}..."):
                passages = [r.payload["text"] for r in results]
                ranked = rerank(sanskrit_query, passages, top_k=TOP_K_RERANK)
                top_results = [results[idx] for idx, score in ranked]
                top_scores  = [score for idx, score in ranked]
                st.write(f"Reranking complete. Top score: {top_scores[0]:.3f}")

            # Step 5 — Show chunks
            if show_chunks:
                st.markdown("### 📚 Retrieved Chunks")
                for i, (result, score) in enumerate(zip(top_results, top_scores), 1):
                    with st.expander(
                        f"Chunk {i} — score: {score:.3f} | source: {result.payload.get('source', 'unknown')}",
                        expanded=(i == 1),
                    ):
                        st.markdown(result.payload["text"])

            # Step 6 — Generate
            st.markdown("### 💬 Answer")
            context_texts = [r.payload["text"] for r in top_results]

            with st.spinner("Generating answer with openai/gpt-oss-120b (Groq)..."):
                answer = generate_answer(query, context_texts)

            st.markdown(answer)

            if translate_query and sanskrit_query != query:
                with st.expander("🔍 Query details"):
                    st.write(f"**Original query:** {query}")
                    st.write(f"**Sanskrit translation:** {sanskrit_query}")

    elif search_btn:
        st.warning("Please enter a query.")
```

---

## ⚠️ Sanskrit-Specific Gotchas to Handle in Code

### 1. Sandhi (Word Fusion)
Sanskrit words fuse at boundaries. Do **not** split on spaces for tokenization. Let BGE-M3's subword XLM-RoBERTa tokenizer handle this — it was trained on 100+ languages and handles Devanagari subword units correctly.

### 2. Always embed the original Sanskrit, never a translation
Translate **queries** into Sanskrit before embedding. Do NOT translate the corpus chunks. The entire vector space is built on Sanskrit — querying with an English embedding would cross the language barrier incorrectly.

### 3. Chunk overlap must be whole shlokas
Never cut a shloka in half for an overlap. Sanskrit syntax flows continuously via Sandhi — truncating mid-construction destroys semantic meaning. The chunker above respects this with whole-shloka overlap.

### 4. LlamaParse `language="sa"`
Always pass this to trigger Devanagari OCR mode. Without it, LlamaParse may misinterpret the script.

### 5. Qdrant sparse vector hash collision
The hash-based integer index for sparse vectors has theoretical collision risk but is acceptable for this use case. For production, maintain a persistent `token→int` vocabulary mapping stored in a JSON file.

### 6. Verify Groq model slug at runtime
Groq periodically renames or retires model slugs. Before deploying, confirm `openai/gpt-oss-120b` is live at https://console.groq.com/docs/models. If it has been renamed, update `GENERATOR_MODEL` in `config.py` accordingly.

---

## 🚦 Running the App

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and fill in keys
cp .env.example .env

# 4. Launch
streamlit run app.py
```

**First run notes:**
- BGE-M3 download: ~570 MB from HuggingFace (cached after first run)
- BGE-reranker download: ~300 MB (cached after first run)
- LlamaParse: requires internet + valid API key
- Qdrant: requires a live Qdrant Cloud cluster URL + API key

---

## 🔧 .env.example

```
LLAMA_CLOUD_API_KEY=llx-xxxxxxxxxxxxxxxxxxxx
QDRANT_URL=https://xxxx.us-east.aws.cloud.qdrant.io
QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxx
SARVAM_API_KEY=xxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

---

## 📌 Where to Get Each API Key

| Key | URL | Free Tier |
|---|---|---|
| `LLAMA_CLOUD_API_KEY` | https://cloud.llamaindex.ai/api-key | 1,000 pages/day |
| `QDRANT_URL` + `QDRANT_API_KEY` | https://cloud.qdrant.io | 1 GB free cluster |
| `SARVAM_API_KEY` | https://dashboard.sarvam.ai | ₹1,000 free credits |
| `GROQ_API_KEY` | https://console.groq.com/keys | 14,400 req/day free |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | 200 req/day free |

---

*End of prompt. Implement each file in sequence: config → chunker → embedder → vector_store → translator → reranker → generator → app.py*

---

One small heads-up: I wasn't able to confirm that **`openai/gpt-oss-120b`** is the exact model ID string Groq uses for that model. Before running the app, double-check the slug at https://console.groq.com/docs/models and update `GENERATOR_MODEL` in `config.py` if needed — otherwise the API call will return a 404. Everything else (`google/gemma-4-31b-it:free` on OpenRouter, all other models) is unchanged.