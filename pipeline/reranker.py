# """
# NVIDIA NIM Reranker (Embedding-based fallback using OpenAI-compatible API)

# Model used:
#   nvidia/llama-nemotron-embed-1b-v2

# This simulates reranking using cosine similarity.
# """

# import os
# import numpy as np
# from typing import List, Tuple
# from openai import OpenAI


# EMBED_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"


# def get_client() -> OpenAI:
#     api_key = os.getenv("NVIDIA_NIM_RERANK_API_KEY")
#     if not api_key:
#         raise ValueError("Set NVIDIA_NIM_RERANK_API_KEY or NVIDIA_API_KEY")

#     return OpenAI(
#         api_key=api_key,
#         base_url="https://integrate.api.nvidia.com/v1"
#     )


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def get_embeddings(client: OpenAI, texts: List[str], input_type: str):
#     response = client.embeddings.create(
#         input=texts,
#         model=EMBED_MODEL,
#         encoding_format="float",
#         extra_body={
#             "input_type": input_type,
#             "truncate": "NONE"
#         }
#     )
#     return [np.array(d.embedding) for d in response.data]


# def rerank(query: str, passages: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
#     if not passages:
#         return []

#     client = get_client()

#     # Get query embedding
#     query_emb = get_embeddings(client, [query], input_type="query")[0]

#     # Get passage embeddings
#     passage_embs = get_embeddings(client, passages, input_type="passage")

#     # Compute similarities
#     scores = []
#     for i, emb in enumerate(passage_embs):
#         sim = cosine_similarity(query_emb, emb)
#         scores.append((i, float(sim)))

#     # Sort descending
#     scores.sort(key=lambda x: x[1], reverse=True)

#     return scores[:top_k]


# if __name__ == "__main__":
#     print("\n🔍 Running NVIDIA NIM embedding-based reranker test...\n")

#     query = "What is the capital of France?"

#     passages = [
#         "Berlin is the capital of Germany.",
#         "Paris is the capital and most populous city of France.",
#         "Madrid is the capital of Spain.",
#         "Rome is the capital of Italy.",
#     ]

#     try:
#         results = rerank(query, passages, top_k=3)

#         print("✅ Top Results:\n")

#         for rank, (idx, score) in enumerate(results, start=1):
#             print(f"{rank}. index={idx} | score={score:.4f}")
#             print(f"   → {passages[idx]}\n")

#         # sanity check
#         best_idx = results[0][0]
#         if "Paris" in passages[best_idx]:
#             print("✔ Sanity check passed (Paris ranked highest)\n")
#         else:
#             print("⚠ Sanity check failed (unexpected ranking)\n")

#     except Exception as e:
#         print("❌ Error occurred:")
#         print(str(e))




"""
NVIDIA NIM Reranker (Direct API implementation)

Model used:
  nvidia/llama-3.2-nv-rerankqa-1b-v2

This utilizes NVIDIA's dedicated reranking endpoint directly.
"""

import os
import requests
from typing import List, Tuple

MODEL_NAME = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
INVOKE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

def rerank(query: str, passages: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
    if not passages:
        return []

    # 1. Retrieve the API key
    api_key = os.getenv("NVIDIA_NIM_RERANK_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_NIM_RERANK_API_KEY or NVIDIA_API_KEY environment variable")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    # 2. Format the payload according to NVIDIA's specific schema
    payload = {
        "model": MODEL_NAME,
        "query": {
            "text": query
        },
        "passages": [{"text": p} for p in passages]
    }

    # 3. Make the API request
    session = requests.Session()
    response = session.post(INVOKE_URL, headers=headers, json=payload)
    
    # Raise an exception if the HTTP request returned an error code
    response.raise_for_status()
    
    # 4. Parse the results
    response_body = response.json()
    rankings = response_body.get("rankings", [])
    
    scores = []
    for rank in rankings:
        idx = rank.get("index")
        # NVIDIA returns raw unnormalized scores under the key 'logit'
        score = rank.get("logit", 0.0) 
        scores.append((idx, float(score)))

    # Sort descending based on the score (just in case)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]

if __name__ == "__main__":
    print("\n🔍 Running NVIDIA NIM reranker test...\n")

    query = "What is the capital of France?"

    passages = [
        "Berlin is the capital of Germany.",
        "Paris is the capital and most populous city of France.",
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy.",
    ]

    try:
        results = rerank(query, passages, top_k=3)

        print("✅ Top Results:\n")

        for rank, (idx, score) in enumerate(results, start=1):
            print(f"{rank}. index={idx} | score={score:.4f}")
            print(f"   → {passages[idx]}\n")

        # sanity check
        best_idx = results[0][0]
        if "Paris" in passages[best_idx]:
            print("✔ Sanity check passed (Paris ranked highest)\n")
        else:
            print("⚠ Sanity check failed (unexpected ranking)\n")

    except Exception as e:
        print("❌ Error occurred:")
        print(str(e))