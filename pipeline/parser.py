"""
pipeline/parser.py
──────────────────
LlamaParse v1  ·  parse_mode = parse_document_with_agent
Handles multilingual PDFs: Sanskrit (Devanagari), Hindi, and English mixed content.

Install:  pip install llama-parse
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
def parse_document(
    file_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Unified entry point.
    .pdf  → parse_pdf()   (LlamaParse v1 → PyMuPDF)
    .doc/.docx → _docx_parse()  (python-docx)
    """
    ext = Path(file_path).suffix.lower()
    if ext in (".doc", ".docx"):
        _status(status_callback, "Word document detected — routing to python-docx parser…")
        return _docx_parse(file_path, status_callback)
    else:
        _status(status_callback, "PDF detected — routing to LlamaParse v1…")
        return parse_pdf(file_path, status_callback)


def parse_pdf(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Parse a PDF with LlamaParse v1 (parse_document_with_agent).

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

    try:
        from llama_parse import LlamaParse  # pip install llama-parse

        _status(status_callback, "Using llama-parse v1 SDK (parse_document_with_agent)…")

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            language="hi",                          # Devanagari hint
            parse_mode="parse_document_with_agent", # full-document agentic pass
            system_prompt=_MULTILINGUAL_PROMPT,
            verbose=False,
            invalidate_cache=False,
        )

        extra_info = {"file_name": Path(pdf_path).name}
        documents = parser.load_data(pdf_path, extra_info=extra_info)
        _status(status_callback, f"LlamaParse v1 done — {len(documents)} document segments")

        pages = [doc.text.strip() for doc in documents if doc.text.strip()]
        return pages if pages else _fallback_parse(pdf_path, status_callback)

    except ImportError:
        _status(status_callback, "llama-parse not installed. Falling back to PyMuPDF…")
        return _fallback_parse(pdf_path, status_callback)
    except Exception as exc:
        _status(status_callback, f"LlamaParse v1 error: {exc}. Falling back to PyMuPDF…")
        return _fallback_parse(pdf_path, status_callback)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback — PyMuPDF (offline, no API required)
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
            "  pip install llama-parse\n"
            "  pip install pymupdf"
        )


def _docx_parse(
    doc_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Extract text from .doc / .docx using python-docx.
    Install:  pip install python-docx
    """
    _status(status_callback, "Extracting text from Word document (python-docx)…")
    try:
        from docx import Document
        doc = Document(doc_path)
        pages, current = [], []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                if current:
                    pages.append("\n".join(current))
                    current = []
            else:
                current.append(text)
        if current:
            pages.append("\n".join(current))
        _status(status_callback, f"python-docx done — {len(pages)} sections extracted")
        return pages if pages else [""]
    except ImportError:
        raise RuntimeError("pip install python-docx  to handle .docx files")


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────
def _status(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb is not None:
        cb(msg)