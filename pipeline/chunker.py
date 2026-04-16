"""
pipeline/chunker.py
────────────────────
Smart multilingual **semantic** chunking for documents that mix:
  • Sanskrit  — verse/shloka structure, double daṇḍa (॥) boundaries
  • Hindi     — prose with single daṇḍa (।) or Latin full-stop sentence endings
  • English   — prose/commentary with paragraph or sentence boundaries

Strategy (in priority order)
─────────────────────────────
1.  Atomic split      — extract fine-grained units using structural markers:
        Sanskrit : double daṇḍa  (॥ / ।। / ||N||)
        Hindi    : single daṇḍa  (।) or paragraph breaks
        English  : sentence-terminal punctuation (.  !  ?)
2.  Context enrichment— every atomic unit is padded with its neighbours
        (CONTEXT_WINDOW on each side) before embedding, so short shlokas
        receive richer positional context.
3.  Batch embed       — all enriched units are sent to NVIDIA NIM
        (llama-nemotron-embed-1b-v2) in batches of EMBED_BATCH_SIZE.
4.  Cosine distance   — pairwise distances between consecutive embeddings.
5.  Breakpoint detect — adaptive percentile threshold (BREAKPOINT_PERCENTILE,
        default 85th) on the distance array marks semantic topic shifts.
6.  Assemble          — atomic units between breakpoints are merged into
        Chunks; any group exceeding MAX_CHARS is further split with a
        sliding-window so the hard cap is always respected.
7.  Metadata tagging  — each chunk carries:
        source, shloka_indices, script_type, char_count
8.  Graceful fallback — if the NVIDIA NIM API key is absent *or* the API
        call fails, the chunker transparently falls back to the original
        fixed-window strategy (no crash, just a log warning).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

import numpy as np
from openai import OpenAI

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────

# --- Fixed-window fallback (used when NIM is unavailable) ---
SHLOKA_GROUP_SIZE: int = 3      # shlokas merged into one chunk
SHLOKA_OVERLAP: int    = 1      # shlokas repeated at boundaries
MAX_CHARS: int         = 1200   # hard cap before sliding-window kicks in
WINDOW_CHARS: int      = 800    # sliding-window size for prose fallback
OVERLAP_CHARS: int     = 150    # overlap for prose fallback
MIN_CHUNK_CHARS: int   = 30     # discard chunks shorter than this

# --- Semantic chunking knobs ---
BREAKPOINT_PERCENTILE: float = 85.0   # ↑ = fewer, larger chunks  (try 75–95)
CONTEXT_WINDOW: int          = 1      # neighbour units used to enrich embeds
EMBED_BATCH_SIZE: int        = 64     # units per NIM API call

# --- NVIDIA NIM config ---
_NIM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL       = "nvidia/llama-nemotron-embed-1b-v2"
_NIM_API_KEY_ENV = os.getenv("NVIDIA_NIM_EMBED_API_KEY") 


# ─────────────────────────────────────────────────────────────────────────────
#  Regex patterns  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

# Sanskrit double daṇḍa with optional verse number
_DOUBLE_DANDA = re.compile(
    r"(?:"
    r"[॥।]{2}\s*[\d०-९]+\s*[॥।]{2}"   # ॥N॥
    r"|[|]{2}\s*[\d]+\s*[|]{2}"        # ||N||
    r"|[॥।]{2}"                         # bare ॥
    r")",
    re.UNICODE,
)

_SINGLE_DANDA    = re.compile(r"[।]", re.UNICODE)
_PARA_BREAK      = re.compile(r"\n{2,}", re.UNICODE)
_ENG_SENTENCE    = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_DEVANAGARI_CHAR = re.compile(r"[\u0900-\u097F]", re.UNICODE)
_ASCII_WORD      = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    shloka_indices: List[int] = field(default_factory=list)
    script_type: str = "mixed"   # "sanskrit" | "hindi" | "english" | "mixed"
    char_count: int = 0

    def to_dict(self) -> dict:
        return {
            "text":           self.text,
            "source":         self.source,
            "shloka_indices": self.shloka_indices,
            "script_type":    self.script_type,
            "char_count":     self.char_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  (signature unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(full_text: str, source: str = "unknown") -> list[dict]:
    """
    Chunk *full_text* using semantic embedding-based chunking.

    Falls back to the original fixed-window strategy silently when the
    NVIDIA NIM API key is absent or the API call fails for any reason.

    Returns
    -------
    list[dict]  — list of chunk dicts (see ``Chunk.to_dict``).
    """
    if not full_text or not full_text.strip():
        return []

    # ── 1. Extract fine-grained atomic units ─────────────────────────────────
    atomic_units = _extract_atomic_units(full_text)
    if not atomic_units:
        return []

    # ── 2. Try semantic grouping; fall back to fixed-window on any failure ────
    try:
        chunks = _semantic_chunk(atomic_units, source)
    except Exception as exc:
        log.warning(
            "Semantic chunking failed (%s). "
            "Hint: ensure %s is set. Falling back to fixed-window strategy.",
            exc,
            _NIM_API_KEY_ENV,
        )
        chunks = _fixed_chunk_fallback(atomic_units, source)

    # ── 3. Filter micro-chunks, finalise char_count ───────────────────────────
    result = []
    for ch in chunks:
        ch.text       = _clean_text(ch.text)
        ch.char_count = len(ch.text)
        if ch.char_count >= MIN_CHUNK_CHARS:
            result.append(ch.to_dict())

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — atomic unit extraction
# ─────────────────────────────────────────────────────────────────────────────

# Internal type: (text, script_type, original_sequential_index)
_Unit = tuple[str, str, int]


def _extract_atomic_units(text: str) -> list[_Unit]:
    """
    Produce a flat, ordered list of fine-grained units.

    Sanskrit blocks (split by double daṇḍa) are kept as-is.
    Long non-Sanskrit blocks are further split by single daṇḍa / paragraph /
    English sentence boundaries before being added.
    """
    raw = _split_on_double_danda(text)
    units: list[_Unit] = []
    idx = 0

    for raw_unit in raw:
        raw_unit = raw_unit.strip()
        if not raw_unit:
            continue

        script = _detect_script(raw_unit)

        # Sanskrit shlokas: already at the right granularity
        if script == "sanskrit" or len(raw_unit) <= MAX_CHARS:
            units.append((raw_unit, script, idx))
            idx += 1
        else:
            # Long prose block: secondary split for finer granularity
            for sub in _secondary_split(raw_unit, script):
                sub = sub.strip()
                if sub:
                    units.append((sub, _detect_script(sub), idx))
                    idx += 1

    return units


# ─────────────────────────────────────────────────────────────────────────────
#  Steps 2-6 — semantic pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_chunk(units: list[_Unit], source: str) -> list[Chunk]:
    """
    Full semantic pipeline:
      context-enrich → batch-embed → cosine distances →
      percentile breakpoints → assemble chunks.
    """
    texts = [u[0] for u in units]

    # Enrich each unit's text with its neighbours for better short-unit embeddings
    enriched = _build_context_texts(texts, window=CONTEXT_WINDOW)

    # Batch-embed via NVIDIA NIM  (raises on failure → caller engages fallback)
    embeddings = _nim_embed_batch(enriched)  # shape: (N, D)

    # Cosine distance between every consecutive pair
    distances = _consecutive_cosine_distances(embeddings)  # shape: (N-1,)

    # Locate semantic boundaries
    breakpoints = _find_breakpoints(distances, BREAKPOINT_PERCENTILE)

    # Assemble final chunks from groups
    return _assemble_chunks(units, breakpoints, source)


def _build_context_texts(texts: list[str], window: int = 1) -> list[str]:
    """
    Pad each text with up to *window* neighbours on each side.

    This prevents short shlokas/sentences from producing degenerate
    embeddings that trigger spurious breakpoints.
    """
    n = len(texts)
    enriched = []
    for i, t in enumerate(texts):
        start = max(0, i - window)
        end   = min(n, i + window + 1)
        enriched.append(" ".join(texts[start:end]))
    return enriched


def _consecutive_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Return cosine *distance* (1 − similarity) for every consecutive pair.
    Output shape: (N-1,).
    """
    # L2-normalise rows so dot product == cosine similarity
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    # Element-wise dot between row i and row i+1
    similarities = np.einsum("ij,ij->i", normed[:-1], normed[1:])

    # Clamp to [0, 1] so negative cosine values don't create phantom boundaries
    return 1.0 - np.clip(similarities, 0.0, 1.0)


def _find_breakpoints(distances: np.ndarray, percentile: float) -> list[int]:
    """
    Adaptive threshold: only gaps above the Nth percentile of *this document*
    are treated as chunk boundaries — no fixed threshold needed.

    Returns indices into the *units* list where a new chunk should START.
    """
    if len(distances) == 0:
        return []

    threshold = float(np.percentile(distances, percentile))
    # distances[i]  =  gap between units[i] and units[i+1]
    # → new chunk starts at units[i+1]
    return [int(i) + 1 for i in np.where(distances > threshold)[0]]


def _assemble_chunks(
    units: list[_Unit],
    breakpoints: list[int],
    source: str,
) -> list[Chunk]:
    """
    Merge atomic units between breakpoints.
    Groups that still exceed MAX_CHARS are split with a sliding window.
    """
    if not units:
        return []

    # Convert breakpoint list into (start, end) group slices
    group_starts = sorted(set([0] + breakpoints))
    group_slices = [
        (group_starts[k], group_starts[k + 1] if k + 1 < len(group_starts) else len(units))
        for k in range(len(group_starts))
    ]

    chunks: list[Chunk] = []

    for start, end in group_slices:
        group = units[start:end]
        if not group:
            continue

        group_text      = "\n\n".join(u[0] for u in group)
        shloka_indices  = [u[2] for u in group if u[1] == "sanskrit"]
        dominant_script = _dominant_script([u[1] for u in group])

        if len(group_text) <= MAX_CHARS:
            chunks.append(Chunk(
                text           = group_text,
                source         = source,
                shloka_indices = shloka_indices,
                script_type    = dominant_script,
            ))
        else:
            # Group still too large after semantic merge → sliding window
            for window_text in _sliding_window(group_text):
                chunks.append(Chunk(
                    text           = window_text,
                    source         = source,
                    shloka_indices = shloka_indices,
                    script_type    = dominant_script,
                ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  NVIDIA NIM embedding layer
# ─────────────────────────────────────────────────────────────────────────────

def _nim_client() -> OpenAI:
    api_key = _NIM_API_KEY_ENV
    if not api_key:
        raise EnvironmentError(
            f"{_NIM_API_KEY_ENV} is not set. "
            "Semantic chunking requires a valid NVIDIA NIM API key."
        )
    return OpenAI(api_key=api_key, base_url=_NIM_BASE_URL)


def _nim_embed_batch(texts: list[str]) -> np.ndarray:
    """
    Batch-embed *texts* in chunks of EMBED_BATCH_SIZE.
    Returns a float32 ndarray of shape (N, embedding_dim).
    Any API error propagates up so the caller can engage the fixed fallback.
    """
    client   = _nim_client()
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch     = texts[i : i + EMBED_BATCH_SIZE]
        sanitized = [t if (t and t.strip()) else " " for t in batch]

        response = client.embeddings.create(
            input           = sanitized,
            model           = _NIM_MODEL,
            encoding_format = "float",
            extra_body      = {
                "input_type": "passage",   # passage mode for document chunks
                "truncate":   "NONE",      # keep full content; NIM will error
                                           # if a unit exceeds model limit —
                                           # adjust to "END" if you see 400s
            },
        )
        all_vecs.extend(item.embedding for item in response.data)

    return np.array(all_vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful fallback — original fixed-window strategy
# ─────────────────────────────────────────────────────────────────────────────

def _fixed_chunk_fallback(units: list[_Unit], source: str) -> list[Chunk]:
    """
    The original grouping logic: Sanskrit shlokas → fixed overlapping windows,
    prose → direct add with sliding-window cap.
    Activated only when NIM embeddings are unavailable.
    """
    chunks: list[Chunk]         = []
    sanskrit_buffer: list[tuple[int, str]] = []
    prose_buffer:    list[tuple[str, str]] = []

    def flush_sanskrit() -> None:
        if not sanskrit_buffer:
            return
        _group_shlokas(sanskrit_buffer, source, chunks)
        sanskrit_buffer.clear()

    def flush_prose() -> None:
        if not prose_buffer:
            return
        for prose_text, prose_script in prose_buffer:
            _add_prose_chunk(prose_text, prose_script, source, chunks)
        prose_buffer.clear()

    for unit_text, unit_script, orig_idx in units:
        if unit_script == "sanskrit":
            flush_prose()
            sanskrit_buffer.append((orig_idx, unit_text))
        else:
            flush_sanskrit()
            prose_buffer.append((unit_text, unit_script))

    flush_sanskrit()
    flush_prose()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _split_on_double_danda(text: str) -> list[str]:
    """
    Split *text* on every double daṇḍa occurrence, keeping the daṇḍa marker
    attached to its unit (important for embedding context).
    """
    parts = []
    last  = 0
    for m in _DOUBLE_DANDA.finditer(text):
        end   = m.end()
        chunk = text[last:end].strip()
        if chunk:
            parts.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts


def _secondary_split(text: str, script: str) -> list[str]:
    """
    Secondary splitter used when a raw unit exceeds MAX_CHARS.
    Chooses strategy based on dominant script.
    """
    if script in ("sanskrit", "hindi"):
        parts = [p.strip() for p in _SINGLE_DANDA.split(text) if p.strip()]
        if len(parts) > 1:
            return parts

    parts = [p.strip() for p in _PARA_BREAK.split(text) if p.strip()]
    if len(parts) > 1:
        result = []
        for p in parts:
            if len(p) > MAX_CHARS:
                result.extend(_sliding_window(p))
            else:
                result.append(p)
        return result

    if script == "english":
        parts = [p.strip() for p in _ENG_SENTENCE.split(text) if p.strip()]
        if len(parts) > 1:
            return _merge_short_sentences(parts)

    return _sliding_window(text)


def _sliding_window(text: str) -> list[str]:
    """Character-level sliding window for very long undivided prose."""
    if len(text) <= WINDOW_CHARS:
        return [text]
    windows = []
    start   = 0
    while start < len(text):
        end = min(start + WINDOW_CHARS, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start += WINDOW_CHARS - OVERLAP_CHARS
    return windows


def _merge_short_sentences(sentences: list[str], min_len: int = 80) -> list[str]:
    """Merge consecutive very-short English sentences to avoid micro-chunks."""
    merged = []
    buf    = ""
    for s in sentences:
        buf = (buf + " " + s).strip() if buf else s
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def _group_shlokas(
    shloka_buffer: list[tuple[int, str]],
    source: str,
    out: list[Chunk],
) -> None:
    """Group buffered Sanskrit shlokas into overlapping windows (fallback only)."""
    n = len(shloka_buffer)
    if n == 0:
        return
    step = max(1, SHLOKA_GROUP_SIZE - SHLOKA_OVERLAP)
    i    = 0
    while i < n:
        window  = shloka_buffer[i : i + SHLOKA_GROUP_SIZE]
        indices = [idx for idx, _ in window]
        text    = "\n\n".join(t for _, t in window)
        out.append(Chunk(
            text           = text,
            source         = source,
            shloka_indices = indices,
            script_type    = "sanskrit",
        ))
        i += step


def _add_prose_chunk(text: str, script: str, source: str, out: list[Chunk]) -> None:
    """Add a single prose unit as a chunk, with sliding-window cap (fallback only)."""
    if len(text) <= MAX_CHARS:
        out.append(Chunk(text=text, source=source, script_type=script))
        return
    for window in _sliding_window(text):
        out.append(Chunk(text=window, source=source, script_type=script))


def _dominant_script(scripts: list[str]) -> str:
    """Return the most frequent script label from a list."""
    if not scripts:
        return "mixed"
    counts: dict[str, int] = {}
    for s in scripts:
        counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.__getitem__)


def _detect_script(text: str) -> str:
    """
    Heuristic: classify a text block's dominant script.
    Returns one of: "sanskrit" | "hindi" | "english" | "mixed"
    """
    if not text:
        return "mixed"
    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return "mixed"

    deva_count  = len(_DEVANAGARI_CHAR.findall(text))
    eng_count   = len("".join(_ASCII_WORD.findall(text)))
    danda_count = text.count("॥") + text.count("।।")

    deva_ratio = deva_count / total
    eng_ratio  = eng_count  / total

    if deva_ratio > 0.55:
        has_danda   = danda_count > 0 or "।" in text
        has_numbers = bool(re.search(r"[०-९\d]+", text))
        return "sanskrit" if (has_danda or has_numbers) else "hindi"
    if eng_ratio > 0.6:
        return "english"
    if deva_ratio > 0.2 and eng_ratio > 0.2:
        return "mixed"
    if deva_ratio > 0.2:
        return "hindi"
    return "english"


def _clean_text(text: str) -> str:
    """
    Normalise whitespace while preserving Devanagari characters exactly.
    - Normalize to NFC (canonical composition).
    - Collapse 3+ newlines → 2.
    - Collapse horizontal whitespace runs.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()