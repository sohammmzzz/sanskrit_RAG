"""
pipeline/parser.py
──────────────────
LlamaParse v2  (pip install llama-cloud-services)
Handles multilingual PDFs: Sanskrit (Devanagari), Hindi, and English mixed content.

WHY THIS WORKS:
  Many Devanagari PDFs have broken font encoding — the text layer decodes every
  glyph to the same codepoint, producing "अ अ अ अ..." garbage.
  The v2 agentic_plus tier forces high-res screenshot + vision model processing,
  bypassing the corrupt text layer entirely. This matches what the online
  LlamaParse UI does under the hood.

Parse strategy (in order of quality):
  1. llama-cloud-services  →  tier="agentic_plus"  (v2 SDK, vision-first)
  2. llama-parse           →  parse_mode="parse_page_with_agent"  (v1 fallback)
  3. PyMuPDF               →  offline text-layer extraction (last resort)
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
#  (works for both v2 `custom_prompt` and v1 `system_prompt`)
# ─────────────────────────────────────────────────────────────────────────────
_MULTILINGUAL_PROMPT = """
This document contains a mix of Sanskrit (Devanagari script), Hindi, and English text.

Rules:
1. Preserve ALL Devanagari characters exactly as they appear — do NOT transliterate,
   romanise, or replace Devanagari with Latin equivalents.
2. Keep daṇḍa (।) and double daṇḍa (॥) punctuation intact — do NOT replace with
   Western full stops or remove them.
3. Preserve verse/shloka numbering (e.g. १, २, ॥१॥, ||1||) in its original form.
4. When English commentary or translation appears alongside Sanskrit, keep both
   side-by-side exactly as laid out on the page.
5. Preserve Hindi prose in Devanagari faithfully.
6. Do NOT merge distinct shlokas/verses into a single paragraph.
7. Output clean Markdown — separate individual shlokas with a blank line.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def parse_pdf(
    pdf_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Parse a multilingual Sanskrit/Hindi/English PDF.

    Tries three backends in order of quality:
      1. llama-cloud-services v2  (agentic_plus, vision-first — best for broken fonts)
      2. llama-parse v1           (parse_page_with_agent — good fallback)
      3. PyMuPDF                  (offline text layer — last resort)

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to a PDF on disk.
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

    # Try modern v2 SDK first
    result = _v2_parse(pdf_path, status_callback)
    if result:
        return result

    # Fall back to v1 SDK
    result = _v1_parse(pdf_path, status_callback)
    if result:
        return result

    # Last resort: offline PyMuPDF
    return _pymupdf_parse(pdf_path, status_callback)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1 — llama-cloud-services v2  (pip install llama-cloud-services)
#
#  tier="agentic_plus":
#    Highest-fidelity tier. Each page is rendered to a high-res screenshot and
#    fed to a vision model in an agentic loop — this BYPASSES broken PDF font
#    encoding entirely, which is why the online UI works but text-layer extraction
#    does not.
#
#  high_res_ocr=True:
#    Forces the highest DPI render before the vision model pass. Critical for
#    small Devanagari conjunct consonants (e.g. क्ष, ज्ञ) that get blurred at
#    standard resolution.
#
#  take_screenshot=True:
#    Explicitly instructs the parser to render each page as an image, not rely
#    on the text layer. This is the key flag that replicates the online UI.
# ─────────────────────────────────────────────────────────────────────────────
def _v2_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
    try:
        from llama_cloud_services import LlamaParse  # pip install llama-cloud-services

        _status(cb, "LlamaParse v2 — initialising agentic_plus tier (vision-first)…")

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            # ── Tier ─────────────────────────────────────────────────────────
            # agentic_plus = maximum fidelity; uses OCR + LVM + LLM in a loop.
            # Equivalent to the online UI's "Agentic Plus" setting.
            tier="agentic_plus",
            # Always pick up the latest model improvements automatically.
            version="latest",
            # ── Vision flags (KEY for broken-font Devanagari PDFs) ───────────
            # Render each page as a high-res image; skip the corrupt text layer.
            take_screenshot=True,
            high_res_ocr=True,
            # ── Output ───────────────────────────────────────────────────────
            result_type="markdown",
            # ── Language ─────────────────────────────────────────────────────
            language="hi",  # Devanagari — covers Sanskrit + Hindi
            # ── Custom instructions ───────────────────────────────────────────
            # Tells the vision model how to handle the mixed-script content.
            # custom_prompt=_MULTILINGUAL_PROMPT,
        )

        _status(cb, "LlamaParse v2 — uploading & parsing (this may take a minute)…")
        t0 = time.time()

        import asyncio
        result = asyncio.run(parser.aparse(pdf_path))
        markdown_nodes = asyncio.run(result.aget_markdown_nodes())

        elapsed = time.time() - t0
        pages = [node.text.strip() for node in markdown_nodes if node.text.strip()]

        _status(cb, f"LlamaParse v2 done — {len(pages)} pages in {elapsed:.1f}s")
        return pages

    except ImportError:
        _status(
            cb,
            "llama-cloud-services not installed — trying v1 SDK…\n"
            "  pip install llama-cloud-services",
        )
        return []

    except Exception as exc:
        _status(cb, f"LlamaParse v2 error: {exc} — trying v1 SDK…")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2 — llama-parse v1  (pip install llama-parse)
#
#  parse_mode="parse_page_with_agent":
#    Combines OCR, page screenshots, and an LLM+LVM agentic loop per page.
#    Equivalent to the old premium_mode=True. Processes page-by-page (not the
#    whole document at once) which is fine for per-page shloka content.
#
#  use_vendor_multimodal_model + vendor_multimodal_model_name:
#    Plugs in Gemini 2.0 Flash as the vision model. Gemini has strong Devanagari
#    support from its multilingual training data.
# ─────────────────────────────────────────────────────────────────────────────
def _v1_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
    try:
        from llama_parse import LlamaParse  # pip install llama-parse

        _status(cb, "LlamaParse v1 — initialising parse_page_with_agent…")

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            # ── Parsing mode ──────────────────────────────────────────────────
            # parse_page_with_agent: OCR + screenshot + LVM + LLM agentic loop
            # per page. This is what premium_mode=True mapped to.
            parse_mode="parse_page_with_agent",
            # ── Vision model ─────────────────────────────────────────────────
            # Route the vision pass through Gemini 2.0 Flash — strong Devanagari
            # support, fast, and accurate on mixed-script layouts.
            # use_vendor_multimodal_model=True,
            # vendor_multimodal_model_name="gemini-2.5-pro",
            # ── Language & instructions ───────────────────────────────────────
            language="hi",
            user_prompt=_MULTILINGUAL_PROMPT,
            # ── Misc ─────────────────────────────────────────────────────────
            split_by_page=True,
            fast_mode=False,
            invalidate_cache=False,
            verbose=False,
        )

        extra_info = {"file_name": Path(pdf_path).name}

        _status(cb, "LlamaParse v1 — uploading PDF…")
        t0 = time.time()

        documents = parser.load_data(pdf_path, extra_info=extra_info)
        elapsed = time.time() - t0

        pages = [doc.text.strip() for doc in documents if doc.text.strip()]
        _status(cb, f"LlamaParse v1 done — {len(pages)} pages in {elapsed:.1f}s")
        return pages

    except ImportError:
        _status(cb, "llama-parse not installed — falling back to PyMuPDF…")
        return []

    except Exception as exc:
        _status(cb, f"LlamaParse v1 error: {exc} — falling back to PyMuPDF…")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 3 — PyMuPDF offline (pip install pymupdf)
#
#  WARNING: This will reproduce the "अ अ अ अ..." garbage for PDFs with broken
#  font encoding, because it reads the text layer directly. Use only as a
#  last resort when no API key is available or both cloud backends fail.
# ─────────────────────────────────────────────────────────────────────────────
def _pymupdf_parse(
    pdf_path: str,
    cb: Optional[Callable[[str], None]] = None,
) -> list[str]:
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
            "No PDF parsing backend is available.\n"
            "Install at least one of:\n"
            "  pip install llama-cloud-services\n"
            "  pip install llama-parse\n"
            "  pip install pymupdf"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────
def _status(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb is not None:
        cb(msg)