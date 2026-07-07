"""
Embedding model wrapper using sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast inference (~14k sentences/sec on CPU)
  - Good quality for semantic similarity tasks
  - Runs locally — no API costs, no rate limits
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Module-level singleton to avoid reloading the model on every call.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloads on first use, ~90MB)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts into dense vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The search query.

    Returns:
        A single embedding vector.
    """
    model = _get_model()
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()
