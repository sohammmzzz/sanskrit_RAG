"""
Utility helpers: language detection, basic text cleaning.
"""

import re

DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
HINDI_RANGE      = re.compile(r'[\u0900-\u097F]')


def detect_script(text: str) -> str:
    """Return 'devanagari', 'latin', or 'mixed'."""
    total = len(text.replace(" ", ""))
    if total == 0:
        return "latin"
    deva_chars = len(DEVANAGARI_RANGE.findall(text))
    ratio = deva_chars / total
    if ratio > 0.6:
        return "devanagari"
    if ratio > 0.1:
        return "mixed"
    return "latin"


def clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving Devanagari."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
