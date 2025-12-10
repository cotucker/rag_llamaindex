import time
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from tqdm import tqdm

VECTOR_DIM = 768
DATASET_SIZES = [1_000, 10_000, 50_000, 100_000, 1_000_000]
TOP_K = 5

def generate_vectors(count, dim):
    vectors = np.random.rand(count, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def brute_force_search(query_vec, all_vectors, k):
    scores = np.dot(all_vectors, query_vec)
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    return top_k_indices

def run_benchmark():
    print(f"🚀 STARTING BENCHMARK (Dim={VECTOR_DIM}, Top_K={TOP_K})")
    print("-" * 65)
    print(f"{'Items':<10} | {'Brute Force (s)':<18} | {'Chroma HNSW (s)':<18} | {'Speedup':<10}")
    print("-" * 65)

    chroma_client = chromadb.EphemeralClient()

    for size in DATASET_SIZES:
        vectors = generate_vectors(size, VECTOR_DIM)
        ids = [str(uuid.uuid4()) for _ in range(size)]
        query_vec = generate_vectors(1, VECTOR_DIM)[0]
        collection_name = f"bench_{size}"
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Явно указываем метрику
        )

        batch_size = 5000
        for i in range(0, size, batch_size):
            end = min(i + batch_size, size)
            collection.add(
                embeddings=vectors[i:end].tolist(),
                ids=ids[i:end]
            )

        start_time = time.perf_counter()
        _ = brute_force_search(query_vec, vectors, TOP_K)
        bf_duration = time.perf_counter() - start_time

        collection.query(query_embeddings=[query_vec.tolist()], n_results=TOP_K)

        start_time = time.perf_counter()
        _ = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=TOP_K
        )
        hnsw_duration = time.perf_counter() - start_time

        speedup = bf_duration / hnsw_duration
        print(f"{size:<10} | {bf_duration:.6f}           | {hnsw_duration:.6f}           | {speedup:.1f}x")

    print("-" * 65)

if __name__ == "__main__":
    run_benchmark()
