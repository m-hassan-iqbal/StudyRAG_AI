"""
Hybrid Retrieval Engine.
Implements subject-isolated semantic search (FAISS) + keyword matching,
score normalization, hybrid ranking, confidence evaluation, and context formatting.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from modules.config import (
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from modules.indexer import load_subject_index, generate_embeddings, get_embedding_model
from modules.storage import load_all_subject_chunks

# Lightweight set of common English stopwords
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def semantic_search(query: str, subject_id: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Performs vector similarity search against the subject's isolated FAISS index.
    Returns chunk dictionaries with added 'semantic_score'.
    """
    index, chunk_map = load_subject_index(subject_id)
    if index is None or not chunk_map:
        return []

    query = query.strip()
    if not query:
        return []

    query_emb = generate_embeddings([query])
    actual_k = min(top_k, len(chunk_map))
    
    scores, indices = index.search(query_emb, actual_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunk_map):
            chunk = dict(chunk_map[idx])
            chunk["semantic_score"] = max(0.0, min(1.0, float(score)))
            results.append(chunk)

    return results


def keyword_search(query: str, subject_id: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Performs pure Python keyword matching over all chunks in the subject.
    Computes query term match coverage and token frequency.
    """
    chunks = load_all_subject_chunks(subject_id)
    if not chunks:
        return []

    query_words = [
        w.lower() for w in re.findall(r"\w+", query)
        if len(w) > 1 and w.lower() not in STOP_WORDS
    ]

    if not query_words:
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1]

    if not query_words:
        return []

    scored_chunks = []
    unique_query_words = set(query_words)
    total_query_terms = len(unique_query_words)

    for chunk in chunks:
        text = chunk.get("text", "").lower()
        chunk_words = re.findall(r"\w+", text)
        chunk_word_set = set(chunk_words)

        matched_terms = sum(1 for term in unique_query_words if term in chunk_word_set)
        term_coverage = matched_terms / total_query_terms if total_query_terms > 0 else 0.0

        term_freq = sum(chunk_words.count(term) for term in unique_query_words)
        freq_factor = min(1.0, term_freq / (len(unique_query_words) * 3))

        kw_score = 0.7 * term_coverage + 0.3 * freq_factor

        if kw_score > 0.05:
            item = dict(chunk)
            item["keyword_score"] = round(kw_score, 4)
            scored_chunks.append(item)

    scored_chunks.sort(key=lambda x: x["keyword_score"], reverse=True)
    return scored_chunks[:top_k]


def hybrid_search(
    query: str,
    subject_id: str,
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Merges semantic search and keyword search within the selected subject.
    Normalizes weights and calculates hybrid_score:
        hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
    """
    total_w = semantic_weight + keyword_weight
    w_sem = semantic_weight / total_w if total_w > 0 else 0.7
    w_kw = keyword_weight / total_w if total_w > 0 else 0.3

    semantic_results = semantic_search(query, subject_id, top_k=top_k * 2)
    keyword_results = keyword_search(query, subject_id, top_k=top_k * 2)

    merged: Dict[str, Dict[str, Any]] = {}

    for item in semantic_results:
        cid = item["chunk_id"]
        merged[cid] = dict(item)
        merged[cid]["keyword_score"] = 0.0

    for item in keyword_results:
        cid = item["chunk_id"]
        if cid in merged:
            merged[cid]["keyword_score"] = item["keyword_score"]
        else:
            merged[cid] = dict(item)
            merged[cid]["semantic_score"] = 0.0

    scored_list = []
    for item in merged.values():
        s_score = item.get("semantic_score", 0.0)
        k_score = item.get("keyword_score", 0.0)
        h_score = (w_sem * s_score) + (w_kw * k_score)
        
        item["hybrid_score"] = round(h_score, 4)
        scored_list.append(item)

    scored_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_chunks = scored_list[:top_k]

    is_confident = False
    if top_chunks:
        best_score = top_chunks[0]["hybrid_score"]
        if best_score >= confidence_threshold or top_chunks[0].get("keyword_score", 0) > 0.5:
            is_confident = True

    return top_chunks, is_confident


def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Builds structured, transparent context blocks for the LLM prompt.
    Includes document name, real page number (or N/A), chunk ID, and text.
    """
    if not retrieved_chunks:
        return "No relevant course material found."

    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        filename = chunk.get("filename", "Unknown")
        page = chunk.get("page")
        page_str = str(page) if page is not None else "N/A"
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "").strip()

        block = (
            f"SOURCE {idx}\n"
            f"Document: {filename}\n"
            f"Page: {page_str}\n"
            f"Chunk ID: {chunk_id}\n\n"
            f"{text}"
        )
        context_parts.append(block)

    return "\n\n──────────────────────────────────────\n\n".join(context_parts)
