"""
ChromaDB vector store operations.

Handles collection management, document insertion, and similarity search.
Data persists to disk at CHROMA_DB_DIR so re-ingestion is not needed
between restarts.
"""

import chromadb
from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, TOP_K_RESULTS
from embeddings import embed_texts, embed_query


def _get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the document collection."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for normalized embeddings
    )


def add_documents(chunks: list[dict]) -> int:
    """
    Add document chunks to the vector store.

    Args:
        chunks: List of dicts with 'text' and 'metadata' keys.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    client = _get_client()
    collection = _get_collection(client)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate unique IDs: source_file + chunk_index
    ids = [
        f"{m['source_file']}_chunk_{m['chunk_index']}"
        for m in metadatas
    ]

    # Embed all texts in batch
    embeddings = embed_texts(texts)

    # Upsert to handle re-ingestion without duplicates
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def query(query_text: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Search for the most relevant chunks given a query.

    Args:
        query_text: The search query (will be embedded internally).
        top_k: Number of results to return.

    Returns:
        List of dicts with 'text', 'metadata', and 'distance' keys,
        ordered by relevance (most relevant first).
    """
    client = _get_client()
    collection = _get_collection(client)

    query_embedding = embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten the nested lists from Chroma's response format
    output = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    return output


def get_chunks_by_document(source_file: str) -> list[dict]:
    """
    Retrieve all chunks belonging to a specific document.
    Used by the /contradict endpoint to compare two documents.

    Args:
        source_file: The source filename to filter by.

    Returns:
        List of chunk dicts from that document.
    """
    client = _get_client()
    collection = _get_collection(client)

    results = collection.get(
        where={"source_file": source_file},
        include=["documents", "metadatas"],
    )

    output = []
    if results["documents"]:
        for i in range(len(results["documents"])):
            output.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

    return output


def list_documents() -> list[str]:
    """
    List all unique source documents in the vector store.

    Returns:
        Sorted list of source filenames.
    """
    client = _get_client()
    collection = _get_collection(client)

    # Get all metadatas to extract unique source files
    all_data = collection.get(include=["metadatas"])

    source_files = set()
    if all_data["metadatas"]:
        for meta in all_data["metadatas"]:
            source_files.add(meta.get("source_file", "unknown"))

    return sorted(source_files)


def get_collection_count() -> int:
    """Return total number of chunks in the store."""
    client = _get_client()
    collection = _get_collection(client)
    return collection.count()


def clear_collection():
    """Delete all data from the collection. Used for re-ingestion."""
    client = _get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist, or has already been deleted, that's fine
