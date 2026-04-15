"""
pipeline/chunker.py
────────────────────
Smart multilingual chunking for documents that mix:
  • Sanskrit  — verse/shloka structure, double daṇḍa (॥) boundaries
  • Hindi     — prose with single daṇḍa (।) or Latin full-stop sentence endings
  • English   — prose/commentary with paragraph or sentence boundaries

Strategy (in priority order)
─────────────────────────────
1.  Structural split  — split on strong Sanskrit structural markers:
      ॥<number>॥  or  ।।<number>।।  or  ||<number>||
      Double daṇḍa alone: ॥ / ।।
2.  Sentence split    — for non-Sanskrit prose, split on single daṇḍa (।),
      newline-pairs (paragraph), or English full-stop patterns.
3.  Window fallback   — any remaining block > MAX_CHARS is slid with
      WINDOW_CHARS / OVERLAP_CHARS.
4.  Group & overlap   — Sanskrit shlokas are grouped in windows of
      SHLOKA_GROUP_SIZE with SHLOKA_OVERLAP overlap.
5.  Metadata tagging  — each chunk carries:
        source, shloka_indices, script_type, char_count
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────
SHLOKA_GROUP_SIZE: int = 3     # shlokas to merge into one chunk
SHLOKA_OVERLAP: int    = 1     # shlokas to repeat at chunk boundaries
MAX_CHARS: int         = 1200  # hard cap before window-fallback kicks in
WINDOW_CHARS: int      = 800   # window size for prose fallback
OVERLAP_CHARS: int     = 150   # overlap for prose fallback
MIN_CHUNK_CHARS: int   = 30    # discard chunks shorter than this


# ─────────────────────────────────────────────────────────────────────────────
#  Regex patterns
# ─────────────────────────────────────────────────────────────────────────────
# Sanskrit double daṇḍa with optional verse number:
#   ॥१॥  ॥2॥  ||1||  ।।1।।  ॥ (bare)  — including surrounding whitespace
_DOUBLE_DANDA = re.compile(
    r"(?:"
    r"[॥।]{2}\s*[\d०-९]+\s*[॥।]{2}"   # ॥N॥
    r"|[|]{2}\s*[\d]+\s*[|]{2}"        # ||N||
    r"|[॥।]{2}"                         # bare ॥
    r")",
    re.UNICODE,
)

# Single Devanagari daṇḍa (।) — sentence/clause boundary in Hindi/Sanskrit prose
_SINGLE_DANDA = re.compile(r"[।]", re.UNICODE)

# English/mixed paragraph boundary (two or more newlines)
_PARA_BREAK = re.compile(r"\n{2,}", re.UNICODE)

# English sentence boundary (period/!/? followed by space + uppercase)
_ENG_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Detect whether a text block is primarily Devanagari
_DEVANAGARI_CHAR = re.compile(r"[\u0900-\u097F]", re.UNICODE)
_ASCII_WORD      = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
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
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def chunk_document(full_text: str, source: str = "unknown") -> list[dict]:
    """
    Chunk *full_text* using the multilingual Sanskrit/Hindi/English strategy.

    Returns
    -------
    list[dict]  — list of chunk dicts (see ``Chunk.to_dict``).
    """
    if not full_text or not full_text.strip():
        return []

    # ── 1. Coarse split into "sections" by double daṇḍa ─────────────────────
    raw_units = _split_on_double_danda(full_text)

    # ── 2. For each raw unit, decide how to further split ────────────────────
    atomic_units: list[tuple[str, str]] = []   # (text, inferred_type)
    for unit in raw_units:
        unit = unit.strip()
        if not unit:
            continue
        script = _detect_script(unit)
        if len(unit) <= MAX_CHARS:
            atomic_units.append((unit, script))
        else:
            # Long block — apply secondary splitter
            sub_units = _secondary_split(unit, script)
            for sub in sub_units:
                sub = sub.strip()
                if sub:
                    atomic_units.append((sub, _detect_script(sub)))

    # ── 3. Group Sanskrit shlokas; keep prose units as-is ───────────────────
    chunks: list[Chunk] = []
    sanskrit_buffer: list[tuple[int, str]] = []  # (original_index, text)
    prose_buffer:    list[tuple[str, str]]  = []  # (text, script)
    global_idx = 0

    def flush_sanskrit():
        nonlocal global_idx
        if not sanskrit_buffer:
            return
        _group_shlokas(sanskrit_buffer, source, chunks)
        sanskrit_buffer.clear()

    def flush_prose():
        if not prose_buffer:
            return
        for prose_text, prose_script in prose_buffer:
            _add_prose_chunk(prose_text, prose_script, source, chunks)
        prose_buffer.clear()

    for unit_text, unit_script in atomic_units:
        if unit_script == "sanskrit":
            flush_prose()
            sanskrit_buffer.append((global_idx, unit_text))
        else:
            flush_sanskrit()
            prose_buffer.append((unit_text, unit_script))
        global_idx += 1

    flush_sanskrit()
    flush_prose()

    # ── 4. Filter micro-chunks, finalise metadata ────────────────────────────
    result = []
    for ch in chunks:
        ch.text = _clean_text(ch.text)
        ch.char_count = len(ch.text)
        if ch.char_count >= MIN_CHUNK_CHARS:
            result.append(ch.to_dict())

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_on_double_danda(text: str) -> list[str]:
    """
    Split *text* on every double daṇḍa occurrence, keeping the daṇḍa marker
    attached to its unit (important for embedding context).
    """
    parts = []
    last = 0
    for m in _DOUBLE_DANDA.finditer(text):
        end = m.end()
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
        # Try single daṇḍa first
        parts = [p.strip() for p in _SINGLE_DANDA.split(text) if p.strip()]
        if len(parts) > 1:
            return parts

    # Try paragraph breaks
    parts = [p.strip() for p in _PARA_BREAK.split(text) if p.strip()]
    if len(parts) > 1:
        # Check individual parts aren't still too long
        result = []
        for p in parts:
            if len(p) > MAX_CHARS:
                result.extend(_sliding_window(p))
            else:
                result.append(p)
        return result

    # English sentence split
    if script == "english":
        parts = [p.strip() for p in _ENG_SENTENCE.split(text) if p.strip()]
        if len(parts) > 1:
            return _merge_short_sentences(parts)

    # Last resort: sliding window
    return _sliding_window(text)


def _sliding_window(text: str) -> list[str]:
    """Character-level sliding window for very long undivided prose."""
    if len(text) <= WINDOW_CHARS:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(start + WINDOW_CHARS, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start += WINDOW_CHARS - OVERLAP_CHARS
    return windows


def _merge_short_sentences(sentences: list[str], min_len: int = 80) -> list[str]:
    """
    Merge consecutive very-short English sentences to avoid micro-chunks.
    """
    merged = []
    buf = ""
    for s in sentences:
        if buf:
            buf += " " + s
        else:
            buf = s
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
    """
    Group buffered Sanskrit shlokas into overlapping windows of SHLOKA_GROUP_SIZE.
    """
    n = len(shloka_buffer)
    if n == 0:
        return

    step = max(1, SHLOKA_GROUP_SIZE - SHLOKA_OVERLAP)
    i = 0
    while i < n:
        window = shloka_buffer[i : i + SHLOKA_GROUP_SIZE]
        indices = [idx for idx, _ in window]
        text    = "\n\n".join(t for _, t in window)
        out.append(
            Chunk(
                text=text,
                source=source,
                shloka_indices=indices,
                script_type="sanskrit",
            )
        )
        i += step


def _add_prose_chunk(text: str, script: str, source: str, out: list[Chunk]) -> None:
    """Add a single prose unit as a chunk, applying window fallback if needed."""
    if len(text) <= MAX_CHARS:
        out.append(Chunk(text=text, source=source, script_type=script))
        return
    for window in _sliding_window(text):
        out.append(Chunk(text=window, source=source, script_type=script))


def _detect_script(text: str) -> str:
    """
    Heuristic: classify a text block's dominant script.

    Returns one of: "sanskrit", "hindi", "english", "mixed"
    """
    if not text:
        return "mixed"

    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return "mixed"

    deva_count = len(_DEVANAGARI_CHAR.findall(text))
    eng_count  = len("".join(_ASCII_WORD.findall(text)))
    danda_count = text.count("॥") + text.count("।।")

    deva_ratio = deva_count / total
    eng_ratio  = eng_count  / total

    if deva_ratio > 0.55:
        # Distinguish Sanskrit (verse) from Hindi (prose):
        # Sanskrit tends to have daṇḍas and verse numbers
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
    - Collapse multiple blank lines to one.
    - Strip leading/trailing whitespace.
    - Do NOT strip Devanagari or any Unicode character.
    """
    # Normalize unicode to NFC (canonical composition) — safe for Devanagari
    text = unicodedata.normalize("NFC", text)
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs (spaces/tabs) but not newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()
