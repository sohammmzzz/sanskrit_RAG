"""
pipeline/parser.py
──────────────────
LlamaParse v2  ·  tier = agentic_plus
Handles multilingual PDFs: Sanskrit (Devanagari), Hindi, and English mixed content.

Install:  pip install llama-cloud>=1.0
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Optional

import nest_asyncio

# ── Allow asyncio.run() inside environments that already have a running loop
# ── (Jupyter, Streamlit, etc.)
nest_asyncio.apply()


# ─────────────────────────────────────────────────────────────────────────────
#  Custom prompt — guides the agentic model on multilingual Devanagari content
# ─────────────────────────────────────────────────────────────────────────────
_MULTILINGUAL_PROMPT = """
This document contains a mix of Sanskrit (Devanagari script), Hindi, and English text.

Please follow these rules carefully:
1. Preserve ALL Devanagari characters exactly as they appear — do NOT transliterate,
   romanise, or replace Devanagari with Latin equivalents.
2. Keep the double daṇḍa (॥) and single daṇḍa (।) punctuation marks intact;
   do NOT replace them with Western full stops or remove them.
3. Preserve verse/shloka numbering (e.g. १, २, ॥१॥, ||1||) in its original form.
4. For English commentary or translations that appear alongside Sanskrit verses,
   keep both the Sanskrit original and the English side-by-side as they appear.
5. For Hindi prose sections, preserve Devanagari text faithfully.
6. Do NOT merge distinct shlokas/verses into a single paragraph.
7. Output clean, structured Markdown — use blank lines to separate individual shlokas.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run a coroutine regardless of whether we are inside an event loop
# ─────────────────────────────────────────────────────────────────────────────
def _run(coro):
    """Run *coro* synchronously even inside a running event loop (nest_asyncio)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_pdf(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Parse a PDF with LlamaParse v2  (tier = agentic_plus).

    Parameters
    ----------
    pdf_path : str
        Absolute path to a PDF file on disk.
    status_callback : callable, optional
        ``fn(message: str)`` called with progress updates so callers
        (e.g. Streamlit) can display live status.

    Returns
    -------
    list[str]
        One string per page; each string is the cleaned Markdown/text
        produced by LlamaParse.
    """
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Get a key at https://cloud.llamaindex.ai/api-key"
        )

    _status(status_callback, "Initialising LlamaParse v2 (agentic_plus tier)…")

    # ── Try the new llama-cloud >= 1.0 SDK first ──────────────────────────────
    try:
        from llama_cloud import LlamaParse  # pip install llama-cloud>=1.0

        parser = LlamaParse(
            api_key=api_key,
            # ── Tier ──────────────────────────────────────────────────────────
            # agentic_plus: full-document agentic loop — best for complex layouts,
            # mixed scripts, and documents with visual + textual content together.
            tier="agentic_plus",
            # ── Language hints (helps the OCR model weigh character sets) ─────
            # "hi" covers Devanagari (Sanskrit + Hindi); English is always auto-detected.
            language="hi",
            # ── Output format ─────────────────────────────────────────────────
            result_type="markdown",
            # ── Custom prompt to preserve Devanagari faithfully ───────────────
            custom_prompt=_MULTILINGUAL_PROMPT,
            # ── OCR / quality flags ───────────────────────────────────────────
            high_res_ocr=True,          # higher-resolution scan pass
            verbose=False,
        )

        _status(status_callback, "Uploading PDF to LlamaParse cloud (agentic_plus)…")
        t0 = time.time()

        # aparse() returns a ParseResult; aget_text_nodes() returns per-page nodes
        result = _run(parser.aparse(pdf_path))

        _status(status_callback, "Waiting for agentic parsing to complete…")
        text_nodes = _run(result.aget_text_nodes())

        elapsed = time.time() - t0
        _status(
            status_callback,
            f"LlamaParse v2 finished — {len(text_nodes)} pages in {elapsed:.1f}s",
        )

        pages = [node.text.strip() for node in text_nodes if node.text.strip()]
        return pages if pages else _fallback_parse(pdf_path, status_callback)

    except ImportError:
        _status(
            status_callback,
            "llama-cloud not found — falling back to llama-parse v1…  "
            "(run: pip install llama-cloud>=1.0 to use v2)",
        )
        return _v1_parse(pdf_path, status_callback)

    except Exception as exc:
        _status(status_callback, f"LlamaParse v2 error: {exc} — trying v1 fallback…")
        return _v1_parse(pdf_path, status_callback)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback A — llama-parse v1  (pip install llama-parse)
# ─────────────────────────────────────────────────────────────────────────────
def _v1_parse(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Legacy llama-parse v1 SDK fallback.
    Uses parse_mode='parse_document_with_agent' which is the v1 equivalent
    of agentic_plus — processes the whole document in a single pass.
    """
    try:
        from llama_parse import LlamaParse  # pip install llama-parse

        _status(status_callback, "Using llama-parse v1 SDK (parse_document_with_agent)…")

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            language="hi",                          # Devanagari hint
            parse_mode="parse_document_with_agent", # full-document agentic pass
            system_prompt=_MULTILINGUAL_PROMPT,
            verbose=False,
            invalidate_cache=False,
        )

        extra_info = {"file_name": Path(pdf_path).name}
        documents = parser.load_data(pdf_path, extra_info=extra_info)
        _status(status_callback, f"v1 parse done — {len(documents)} document segments")

        pages = [doc.text.strip() for doc in documents if doc.text.strip()]
        return pages if pages else _fallback_parse(pdf_path, status_callback)

    except ImportError:
        _status(status_callback, "llama-parse not installed. Falling back to PyMuPDF…")
        return _fallback_parse(pdf_path, status_callback)
    except Exception as exc:
        _status(status_callback, f"v1 parse error: {exc}. Falling back to PyMuPDF…")
        return _fallback_parse(pdf_path, status_callback)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback B — PyMuPDF (offline, no API required)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_parse(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Pure-Python offline fallback using PyMuPDF (fitz).
    No OCR — works only on text-layer PDFs; Devanagari glyphs are preserved
    as-is from the font layer.

    Install:  pip install pymupdf
    """
    _status(status_callback, "Using PyMuPDF offline text extraction (no OCR)…")
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(text)
            _status(
                status_callback,
                f"PyMuPDF: extracted page {i + 1}/{len(doc)}",
            )
        doc.close()
        _status(status_callback, f"PyMuPDF done — {len(pages)} pages extracted")
        return pages

    except ImportError:
        raise RuntimeError(
            "No PDF parsing backend available.\n"
            "Install at least one of:\n"
            "  pip install llama-cloud>=1.0\n"
            "  pip install llama-parse\n"
            "  pip install pymupdf"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────
def _status(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb is not None:
        cb(msg)
