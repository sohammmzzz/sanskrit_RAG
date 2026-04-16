"""
pipeline/parser.py
──────────────────
Vision-first parser for PDFs and office documents.

Primary strategy (best to worst):
  1. llama-cloud (current SDK) → tier="agentic_plus"  (vision-first, best fidelity)
  2. llama-parse (legacy SDK)   → parse_mode="parse_page_with_agent"
  3. PyMuPDF / python-docx      → offline fallback

Why this version is better:
  - Uses the current LlamaParse SDK (`llama-cloud`) instead of the deprecated
    `llama-cloud-services` package.
  - Uses the v2 tier-based API (`agentic_plus`) instead of v1 parse modes.
  - Sends OCR language hints through `processing_options.ocr_parameters.languages`
    as required by v2.
  - Preserves Devanagari / mixed Hindi-English output with a stricter prompt.
  - Accepts both PDFs and Word documents without changing the public API.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

import nest_asyncio

nest_asyncio.apply()


# ─────────────────────────────────────────────────────────────────────────────
#  Custom prompt for multilingual Devanagari content
#  Works with v2 `agentic_options.custom_prompt` and v1 `system_prompt`.
# ─────────────────────────────────────────────────────────────────────────────
_MULTILINGUAL_PROMPT = """
This document contains a mix of Sanskrit (Devanagari script), Hindi, and English text.

Rules:
1. Preserve ALL Devanagari characters exactly as they appear. Do NOT transliterate,
   romanise, normalise, or replace them with Latin equivalents.
2. Keep daṇḍa (।) and double daṇḍa (॥) punctuation intact.
3. Preserve verse and shloka numbering exactly as shown on the page.
4. Keep adjacent English commentary or translation aligned with the Sanskrit it belongs to.
5. Preserve line breaks between verses. Do NOT merge distinct shlokas into one paragraph.
6. Prefer faithful transcription over cleanup whenever there is ambiguity.
7. Output clean Markdown with sensible paragraph breaks and no invented content.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def parse_pdf(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Parse a document file from disk.

    Supports PDFs and office documents such as DOCX because the cloud parser
    supports both. The name is kept for compatibility with the larger codebase.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to a local document file.
    status_callback : callable, optional
        fn(message: str) — called with live progress for Streamlit / logging.

    Returns
    -------
    list[str]
        One Markdown string per page/segment.
    """
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY is not set.\n"
            "Get your key at: https://cloud.llamaindex.ai/api-key"
        )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {pdf_path}")

    # Try current v2 SDK first
    result = _v2_parse(pdf_path, status_callback)
    if result:
        return result

    # Fall back to legacy v1 SDK
    result = _v1_parse(pdf_path, status_callback)
    if result:
        return result

    # Last resort: offline text extraction
    return _pymupdf_parse(pdf_path, status_callback)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1 — llama-cloud (current SDK)
#
#  Best current practice:
#    - Upload the local file with `client.files.create(..., purpose="parse")`
#    - Parse with tier="agentic_plus"
#    - Send OCR hints via processing_options.ocr_parameters.languages
#    - Use agentic_options.custom_prompt for layout / language guidance
#
#  Why these parameters:
#    - agentic_plus is the highest-fidelity tier for visually complex documents
#      and scanned / broken-font PDFs.
#    - high_res_ocr is always on in v2, so we do NOT set it manually.
#    - Model selection is automatic in v2, so we do NOT pick a vision model.
# ─────────────────────────────────────────────────────────────────────────────
def _v2_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
    try:
        from llama_cloud import AsyncLlamaCloud  # current SDK

        _status(cb, "LlamaParse v2 — initialising agentic_plus tier (vision-first)…")

        async def _run() -> list[str]:
            client = AsyncLlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

            _status(cb, "LlamaParse v2 — uploading document…")
            file_obj = await client.files.create(
                file=pdf_path,
                purpose="parse",
                external_file_id=Path(pdf_path).name,
            )

            _status(cb, "LlamaParse v2 — parsing with vision-first settings…")
            t0 = time.time()

            result = await client.parsing.parse(
                file_id=file_obj.id,
                tier="agentic_plus",
                version="latest",
                expand=["markdown"],
                processing_options={
                    "ignore": {
                        "ignore_diagonal_text": True,
                        "ignore_hidden_text": True,
                    },
                    # OCR language hints only affect text extracted from images.
                    # Hindi is the safest documented Devanagari OCR hint in the
                    # current docs, and English is kept for mixed-script pages.
                    "ocr_parameters": {
                        "languages": ["en", "hi"],
                    },
                },
                agentic_options={
                    "custom_prompt": _MULTILINGUAL_PROMPT,
                },
                output_options={
                    "markdown": {
                        "tables": {
                            "output_tables_as_markdown": True,
                        }
                    }
                },
            )

            elapsed = time.time() - t0

            markdown_pages = getattr(getattr(result, "markdown", None), "pages", []) or []
            pages = []
            for page in markdown_pages:
                text = getattr(page, "markdown", None) or getattr(page, "text", None) or ""
                text = text.strip()
                if text:
                    pages.append(text)

            # Fallbacks in case the SDK response shape changes slightly.
            if not pages:
                text_pages = getattr(getattr(result, "text", None), "pages", []) or []
                for page in text_pages:
                    text = getattr(page, "text", None) or ""
                    text = text.strip()
                    if text:
                        pages.append(text)

            _status(cb, f"LlamaParse v2 done — {len(pages)} pages in {elapsed:.1f}s")
            return pages

        import asyncio
        return asyncio.run(_run())

    except ImportError:
        _status(cb, "llama-cloud not installed — trying legacy v1 SDK…")
        return []

    except Exception as exc:
        _status(cb, f"LlamaParse v2 error: {exc} — trying legacy v1 SDK…")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2 — llama-parse v1 (legacy fallback)
#
#  This path is kept only for compatibility with older environments.
#  The current SDK is `llama-cloud`, and the legacy `llama-cloud-services`
#  package is deprecated.
# ─────────────────────────────────────────────────────────────────────────────
def _v1_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
    try:
        from llama_parse import LlamaParse  # legacy package

        _status(cb, "LlamaParse v1 — initialising parse_page_with_agent…")

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            parse_mode="parse_page_with_agent",
            # Keep the prompt for legacy environments; v1 uses system_prompt.
            system_prompt=_MULTILINGUAL_PROMPT,
            split_by_page=True,
            fast_mode=False,
            invalidate_cache=False,
            verbose=False,
        )

        extra_info = {"file_name": Path(pdf_path).name}

        _status(cb, "LlamaParse v1 — uploading document…")
        t0 = time.time()

        documents = parser.load_data(pdf_path, extra_info=extra_info)
        elapsed = time.time() - t0

        pages = [doc.text.strip() for doc in documents if getattr(doc, "text", "").strip()]
        _status(cb, f"LlamaParse v1 done — {len(pages)} pages in {elapsed:.1f}s")
        return pages

    except ImportError:
        _status(cb, "Legacy llama-parse not installed — falling back to offline parser…")
        return []

    except Exception as exc:
        _status(cb, f"LlamaParse v1 error: {exc} — falling back to offline parser…")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 3 — Offline fallback
#
#  PDF: PyMuPDF text-layer extraction.
#  DOCX: python-docx paragraph extraction.
#  DOC:  not reliably supported offline here; cloud parsing is preferred.
# ─────────────────────────────────────────────────────────────────────────────
def _pymupdf_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
    suffix = Path(pdf_path).suffix.lower()

    if suffix == ".docx":
        _status(cb, "Offline DOCX fallback — using python-docx paragraph extraction.")
        try:
            from docx import Document

            doc = Document(pdf_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                _status(cb, f"python-docx done — {len(paragraphs)} paragraphs extracted")
                return ["\n\n".join(paragraphs)]
            _status(cb, "python-docx done — no paragraph text found")
            return []

        except ImportError:
            raise RuntimeError(
                "DOCX fallback requires python-docx.\n"
                "Install it with:\n"
                "  pip install python-docx"
            )

    _status(
        cb,
        "PyMuPDF offline fallback — WARNING: may produce garbled Devanagari "
        "if the PDF has broken font encoding.",
    )

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(text)
            _status(cb, f"PyMuPDF: page {i + 1}/{len(doc)}")
        doc.close()
        _status(cb, f"PyMuPDF done — {len(pages)} pages extracted")
        return pages

    except ImportError:
        raise RuntimeError(
            "No offline document parsing backend is available.\n"
            "Install at least one of:\n"
            "  pip install llama-cloud\n"
            "  pip install llama-parse\n"
            "  pip install pymupdf\n"
            "  pip install python-docx"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────
def _status(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb is not None:
        cb(msg)
