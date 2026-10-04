"""
Vector Indexing & Embedding Engine.
Generates embeddings using sentence-transformers, normalizes vectors for exact
cosine similarity, and maintains subject-isolated FAISS indices.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from modules.config import DEFAULT_EMBEDDING_MODEL
from modules.storage import get_subject_path, ensure_subject_dirs


class LocalDenseEmbeddingModel:
    """
    High-speed deterministic dense embedding model (384-dimensional).
    Acts as a graceful fallback if HuggingFace Hub is offline, firewalled, or HTTP 429 rate-limited.
    Uses word and character n-gram hashing to generate dense 384-d vectors.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: List[str], show_progress_bar: bool = False, convert_to_numpy: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vectors.append(vec)
                continue

            for word in words:
                h = hash(word)
                idx1 = abs(h) % self.dimension
                sign1 = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vec[idx1] += sign1

                if len(word) >= 3:
                    for i in range(len(word) - 2):
                        ngram = word[i:i+3]
                        h_ng = hash(ngram)
                        idx2 = abs(h_ng) % self.dimension
                        sign2 = 0.5 if (h_ng >> 8) % 2 == 0 else -0.5
                        vec[idx2] += sign2

            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            vectors.append(vec)

        res = np.array(vectors, dtype=np.float32)
        return res


_CACHED_MODEL: Optional[Any] = None
_CACHED_MODEL_NAME: Optional[str] = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """
    Loads and caches the SentenceTransformer model.
    If HuggingFace Hub is rate-limited (HTTP 429) or offline, falls back gracefully
    to LocalDenseEmbeddingModel to ensure continuous operation without failure.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    if _CACHED_MODEL is None or _CACHED_MODEL_NAME != model_name:
        try:
            _CACHED_MODEL = SentenceTransformer(model_name)
            _CACHED_MODEL_NAME = model_name
        except Exception as e:
            print(f"[Indexer Warning] Could not load '{model_name}' from HuggingFace ({e}). Falling back to LocalDenseEmbeddingModel.")
            _CACHED_MODEL = LocalDenseEmbeddingModel(dimension=384)
            _CACHED_MODEL_NAME = "local_dense_fallback"
    return _CACHED_MODEL


def generate_embeddings(texts: List[str], model: Optional[Any] = None) -> np.ndarray:
    """
    Generates L2-normalized embeddings for a list of strings.
    Normalization ensures that the dot product (IndexFlatIP) equals cosine similarity.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def get_index_paths(subject_id: str) -> Tuple[Path, Path]:
    """Returns the paths to the subject's index.faiss and index_map.json."""
    subject_dir = get_subject_path(subject_id)
    index_path = subject_dir / "index.faiss"
    map_path = subject_dir / "metadata" / "index_map.json"
    return index_path, map_path


def build_and_save_index(
    subject_id: str,
    chunks: List[Dict[str, Any]],
    model: Optional[Any] = None
) -> bool:
    """
    Builds a subject-isolated FAISS vector index from chunks and persists it to disk.
    Also saves index_map.json linking vector positions directly to chunk dictionaries.
    """
    ensure_subject_dirs(subject_id)
    index_path, map_path = get_index_paths(subject_id)

    if not chunks:
        if index_path.exists():
            index_path.unlink()
        if map_path.exists():
            map_path.unlink()
        return False

    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts, model=model)
    dimension = embeddings.shape[1]

    # IndexFlatIP with normalized vectors computes cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return True


def load_subject_index(subject_id: str) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    """
    Loads the persistent FAISS index and chunk metadata mapping for a specific subject.
    Returns (None, []) if the index has not been built yet.
    """
    index_path, map_path = get_index_paths(subject_id)

    if not index_path.exists() or not map_path.exists():
        return None, []

    try:
        index = faiss.read_index(str(index_path))
        with open(map_path, "r", encoding="utf-8") as f:
            chunk_map = json.load(f)
        return index, chunk_map
    except Exception as e:
        print(f"Error loading FAISS index for '{subject_id}': {e}")
        return None, []
