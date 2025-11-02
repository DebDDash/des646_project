import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
import os


def normalize(vecs: np.ndarray) -> np.ndarray:
    """Efficient in-place L2 normalization."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-12)
    return vecs.astype(np.float32)


def build_faiss_index_auto(embeddings: np.ndarray):
    """
    Automatically choose and build an optimal FAISS index:
    - <1K → FlatL2 (exact)
    - 1K–100K → IVFFlat
    - >100K → IVFPQ
    """
    n, d = embeddings.shape
    embeddings = normalize(embeddings)
    print(f"[Auto] Selecting FAISS index for {n:,} vectors...")

    if n < 1_000:
        index = faiss.IndexFlatIP(d)
    elif n < 100_000:
        nlist = max(1, n // 10)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"[Info] Training IVFFlat index (nlist={nlist})...")
        index.train(embeddings)
    else:
        nlist = max(1000, n // 10)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8, faiss.METRIC_INNER_PRODUCT)
        print(f"[Info] Training IVFPQ index (nlist={nlist})...")
        index.train(embeddings)

    index.add(embeddings)
    print(f"[Info] Built index with {index.ntotal:,} vectors.")
    return index


def search_faiss(index, queries: np.ndarray, k: int = 5):
    """Fast FAISS search (cosine similarity)."""
    queries = normalize(queries)
    distances, indices = index.search(queries, k)
    return distances, indices


def find_duplicates_faiss_fast(embeddings: np.ndarray, threshold: float = 0.95, k: int = 5):
    """High-speed duplicate finder using FAISS inner-product search."""
    embeddings = normalize(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    sims, idxs = index.search(embeddings, k + 1)

    # Vectorized filtering
    mask = sims > threshold
    mask[:, 0] = False  # remove self-matches
    dup_pairs = np.argwhere(mask)

    # Group by source index
    duplicates = {}
    for i, j in dup_pairs:
        duplicates.setdefault(i, []).append(int(j))
    return list(duplicates.items())


def build_knn_graph_fast(embeddings: np.ndarray, k: int = 10, out_path: str = "knn_graph.npy"):
    """Fast cosine kNN graph using FAISS."""
    embeddings = normalize(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    sims, idxs = index.search(embeddings, k)
    np.save(out_path, {"similarities": sims, "indices": idxs})
    return sims, idxs


def build_sklearn_fallback(embeddings: np.ndarray, n_neighbors: int = 5):
    """CPU-only fallback (for very small datasets or missing FAISS)."""
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    return nn
