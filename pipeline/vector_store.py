"""
Qdrant Cloud Hybrid Vector Store.

Collection schema:
  - Named vector "dense": size=DENSE_DIM, distance=Cosine
  - Named vector "sparse": SparseVectorParams
  - Payload: { text, source, shloka_indices }
"""

import os
import sys
import uuid
import hashlib
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector,
    FusionQuery, Fusion, Prefetch, ScoredPoint,
)

# Assuming these are imported from your config
# from config import COLLECTION_NAME, DENSE_DIM
COLLECTION_NAME = "sanskrit_texts"
DENSE_DIM = 2048

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stable_hash(token: str, max_val: int = 10**6) -> int:
    """
    Creates a consistent integer hash for sparse tokens. 
    Unlike Python's built-in hash(), this won't change between script runs.
    """
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % max_val


def get_client() -> QdrantClient:
    """Initialize Qdrant client with gRPC for better performance."""
    print("\n🔗 Connecting to Qdrant Cloud...")
    try:
        client = QdrantClient(
            url="https://f4002ef3-a118-4ef5-a972-76f2080eeb94.us-west-2-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OWMyNWEzMGMtNGNmNy00OWM2LWIwZjYtY2M0YjhjNTQ2ZmFjIn0.QVdq3JeFItVp0t8rYGDwb3MRzQfU0QLjXx85uYxgqR4",
            prefer_grpc=True,
        )
        print("  ✓ Connected successfully")
        return client
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        sys.exit(1)


# ============================================================================
# CORE DATABASE OPERATIONS
# ============================================================================

def ensure_collection(client: QdrantClient):
    """Checks if the collection exists, deletes it if it does, and creates a fresh one."""
    
    # Check if it already exists
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"\n⚠️  Collection '{COLLECTION_NAME}' already exists. Overwriting...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"  ✓ Existing collection deleted.")

    # Create the fresh collection
    print(f"\n🏗️  Creating fresh hybrid collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print("  ✓ Collection created successfully")


def upsert_chunks(client: QdrantClient, chunks: list[dict], embeddings: list[dict]):
    """Uploads documents in batches with a progress bar."""
    if not chunks:
        print("✅ No chunks to upload.")
        return

    points = []
    print("\n📝 Preparing payload and stable hashes...")
    
    for chunk, emb in zip(chunks, embeddings):
        sparse_indices = []
        sparse_values = []
        
        # Safely handle sparse tokens using the stable hash
        if "sparse" in emb and emb["sparse"]:
            for token, weight in emb["sparse"].items():
                sparse_indices.append(get_stable_hash(token))
                sparse_values.append(float(weight))

        # Safely handle dense vectors (prevents .tolist() errors if already a list)
        dense_vector = emb["dense"]
        if hasattr(dense_vector, "tolist"):
            dense_vector = dense_vector.tolist()

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vector,
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "Unknown"),
                "shloka_indices": chunk.get("shloka_indices", []),
            },
        )
        points.append(point)

    batch_size = 100
    print(f"\n⬆️  Uploading {len(points)} documents to Qdrant...")
    
    try:
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            
        print("✅ Successfully uploaded all hybrid chunks!")
    except Exception as e:
        print(f"\n❌ Error during upload: {str(e)}")
        raise


def hybrid_search(
    client: QdrantClient,
    query_dense: list[float],
    query_sparse: dict,
    top_k: int = 10,
) -> list[ScoredPoint]:
    """Performs Reciprocal Rank Fusion (RRF) search using dense and sparse vectors."""
    
    # Safely handle dense vector
    if hasattr(query_dense, "tolist"):
        query_dense = query_dense.tolist()

    if not query_sparse:
        print("ℹ️  No sparse query provided, falling back to dense search.")
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_dense,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        return results.points

    # MUST use the exact same stable hash logic as the upsert function
    sparse_indices = [get_stable_hash(t) for t in query_sparse.keys()]
    sparse_values = [float(w) for w in query_sparse.values()]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=query_dense, using="dense", limit=top_k * 2),
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


# ============================================================================
# STANDALONE CONNECTION TEST
# ============================================================================

def run_connection_test():
    """
    Tests the Qdrant connection by creating a temporary collection,
    inserting dummy hybrid data, verifying it, and deleting the collection.
    """
    print("=" * 70)
    print("🚀 RUNNING QDRANT CONNECTION & HYBRID VECTOR TEST")
    print("=" * 70)

    # Generate a unique test collection name to avoid overwriting production data
    test_collection = f"test_connection_{str(uuid.uuid4())[:8]}"
    client = get_client()

    try:
        # 1. Create Test Collection
        print(f"\n🏗️  Creating temporary test collection: '{test_collection}'...")
        client.create_collection(
            collection_name=test_collection,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )
        print("  ✓ Test collection created")

        # 2. Insert Dummy Data
        print("\n📝 Inserting dummy hybrid document...")
        dummy_dense = [0.1] * DENSE_DIM  # Mock dense vector
        dummy_sparse_indices = [get_stable_hash("test"), get_stable_hash("connection")]
        dummy_sparse_values = [0.8, 0.5]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dummy_dense,
                "sparse": SparseVector(indices=dummy_sparse_indices, values=dummy_sparse_values),
            },
            payload={"text": "This is a connection test document.", "source": "test_script"}
        )
        
        client.upsert(collection_name=test_collection, points=[point])
        print("  ✓ Dummy data inserted successfully")

        # 3. Verify Insertion
        print("\n🔍 Verifying insertion...")
        info = client.get_collection(collection_name=test_collection)
        print(f"  ✓ Collection now has {info.points_count} point(s)")
        if info.points_count != 1:
            raise ValueError("Point count mismatch! Data was not inserted correctly.")

        # 4. Cleanup / Delete Collection
        print(f"\n🗑️  Cleaning up: Deleting test collection '{test_collection}'...")
        client.delete_collection(collection_name=test_collection)
        print("  ✓ Test collection completely removed")

        print("\n" + "=" * 70)
        print("✅ ALL CONNECTION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 70)
        
        # Attempt safe cleanup if test failed midway
        if client.collection_exists(collection_name=test_collection):
            print(f"⚠️  Attempting emergency cleanup of '{test_collection}'...")
            client.delete_collection(collection_name=test_collection)
            print("  ✓ Emergency cleanup successful")
        sys.exit(1)


if __name__ == "__main__":
    run_connection_test()