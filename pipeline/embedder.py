"""
NVIDIA NIM Embeddings via OpenAI-compatible client (STABLE)

Model:
    nvidia/llama-nemotron-embed-1b-v2

Run:
    python embedder.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

from openai import OpenAI


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEFAULT_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
BASE_URL = "https://integrate.api.nvidia.com/v1"


# -------------------------------------------------------------------
# Client
# -------------------------------------------------------------------
def _get_client() -> OpenAI:
    api_key = "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"

    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY (or NGC_API_KEY) in environment")

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def embed_texts(texts: List[str], model: str = None) -> List[Dict[str, Any]]:
    """
    Batch embeddings
    """
    if not texts:
        return []

    client = _get_client()
    model = model or DEFAULT_MODEL

    sanitized = [t if t and t.strip() else " " for t in texts]

    response = client.embeddings.create(
        input=sanitized,
        model=model,
        encoding_format="float",
        extra_body={
            "input_type": "passage",   # IMPORTANT for documents
            "truncate": "NONE",
        },
    )

    return [
        {"dense": item.embedding, "sparse": {}}
        for item in response.data
    ]


def embed_query(query: str, model: str = None) -> Dict[str, Any]:
    """
    Query embedding
    """
    if not query:
        return {"dense": [], "sparse": {}}

    client = _get_client()
    model = model or DEFAULT_MODEL

    response = client.embeddings.create(
        input=[query],
        model=model,
        encoding_format="float",
        extra_body={
            "input_type": "query",   # IMPORTANT difference
            "truncate": "NONE",
        },
    )

    return {
        "dense": response.data[0].embedding,
        "sparse": {},
    }


# -------------------------------------------------------------------
# Standalone Tests
# -------------------------------------------------------------------
def _norm(vec: List[float]) -> float:
    return sum(v * v for v in vec) ** 0.5


if __name__ == "__main__":
    print("=" * 60)
    print("  NVIDIA NIM Embedding (OpenAI client) — Test")
    print("=" * 60)

    # 1. API key check
    try:
        key = os.getenv("NVIDIA_NIM_EMBED_API_KEY")
        if not key:
            raise ValueError("Missing NVIDIA_API_KEY")

        print(f"\n✓ API key found : {key[:8]}…{key[-4:]}")
    except Exception as e:
        print(f"\n✗ {e}")
        sys.exit(1)

    print(f"  Model          : {DEFAULT_MODEL}")

    # 2. embed_query
    print("\n── Test 1: embed_query ──")
    try:
        result = embed_query("What is the capital of France?")
        dense = result["dense"]

        print(f"  ✓ dim      = {len(dense)}")
        print(f"  ✓ norm     = {_norm(dense):.4f}")
        print(f"  ✓ first 5  = {dense[:5]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 3. embed_texts
    print("\n── Test 2: embed_texts (batch + Sanskrit) ──")
    try:
        texts = [
            "This is sentence one.",
            "This is sentence two.",
            "यह संस्कृत वाक्य है।",
        ]

        results = embed_texts(texts)

        print(f"  ✓ batch size = {len(results)}")

        for i, r in enumerate(results):
            d = r["dense"]
            print(f"  ✓ [{i}] dim={len(d)}, norm={_norm(d):.4f}, preview={d[:3]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 4. Edge cases
    print("\n── Test 3: Edge cases ──")
    try:
        assert embed_texts([]) == []
        print("  ✓ empty list → []")

        r = embed_query("")
        assert r["dense"] == []
        print("  ✓ empty query → dense=[]")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)