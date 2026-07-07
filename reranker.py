"""
Cross-Encoder Reranker.

Implements a second-stage ranking step to improve retrieval accuracy.
Vector search (first-stage) is fast but misses fine-grained semantic matches.
A Cross-Encoder (second-stage) processes the query and document chunk
together, yielding a much higher-fidelity relevance score.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, fast and accurate)
"""

from sentence_transformers import CrossEncoder
import math
import time

# Module-level singleton
_rerank_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the Cross-Encoder model."""
    global _rerank_model
    if _rerank_model is None:
        # Fast, low-footprint cross encoder model
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _rerank_model


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using a Cross-Encoder.

    Args:
        query: Search query in English.
        chunks: Chunks retrieved from first-stage vector search.
        top_k: Number of final chunks to return.

    Returns:
        Sorted list of chunks containing updated relevance scores (as distances).
    """
    if not chunks:
        return []

    # If we have very few chunks, rerank all; otherwise, rerank up to 10
    candidates = chunks[:10]

    model = _get_reranker()
    
    # Pairs for cross-encoder scoring: (query, document_text)
    pairs = [[query, c["text"]] for c in candidates]
    
    start_time = time.time()
    scores = model.predict(pairs)
    duration = time.time() - start_time
    
    # Attach scores to candidates
    # Higher cross-encoder score = more relevant.
    # Convert to a distance-like metric where lower is better or store score directly
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)
        # Scale score for display: Sigmoid of raw score is a good approximation of relevance probability
        # Raw MS-Marco scores can range from -10 to +10.
        try:
            chunk["relevance_probability"] = 1 / (1 + math.exp(-score))
        except OverflowError:
            chunk["relevance_probability"] = 0.0 if score < 0 else 1.0

    # Sort candidates by score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"⚡ Reranked {len(candidates)} candidates in {duration:.3f}s. Top score: {reranked[0]['rerank_score']:.3f}")
    
    return reranked[:top_k]
