"""
Answer generation using Groq (primary) with OpenRouter fallback.

Primary model  (Groq):       compound-beta
Fallback model (OpenRouter): google/gemma-3-27b-it:free
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
    groq_err = None
    try:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        groq_err = e
        print(f"[Groq failed, trying OpenRouter]: {e}")

    # ── Fallback: OpenRouter ──
    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
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
