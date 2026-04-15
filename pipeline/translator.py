"""
Query translation: English/Hindi/Hinglish → Sanskrit
using Sarvam AI sarvam-translate:v1

Docs: https://docs.sarvam.ai/api-reference-docs/translate

Note: 'auto' source_language_code is NOT supported by sarvam-translate:v1.
      We detect the script/language and pick the correct code.
"""

import requests
import os
import re

SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"

# Devanagari Unicode block
_DEVA = re.compile(r'[\u0900-\u097F]')


def _detect_source_lang(text: str) -> str:
    """
    Heuristic language detection for Sarvam source codes.
    Returns one of: 'en-IN', 'hi-IN', 'sa-IN'
    """
    total = len(text.replace(" ", ""))
    if total == 0:
        return "en-IN"
    deva_ratio = len(_DEVA.findall(text)) / total
    if deva_ratio > 0.6:
        # Mostly Devanagari — could be Hindi or Sanskrit.
        # Sanskrit queries don't need translation, but we still send
        # them through to re-embed in the Sanskrit vector space.
        return "hi-IN"
    # Latin script — English or Hinglish (treat as en-IN)
    return "en-IN"


def translate_to_sanskrit(query: str) -> str:
    src_lang = _detect_source_lang(query)

    # If query is already Sanskrit-script, skip the round-trip
    if src_lang == "sa-IN":
        return query

    headers = {
        "api-subscription-key": os.getenv("SARVAM_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {
        "input": query,
        "source_language_code": src_lang,
        "target_language_code": "sa-IN",
        "model": "sarvam-translate:v1",
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
