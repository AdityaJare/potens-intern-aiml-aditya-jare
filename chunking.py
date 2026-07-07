"""
Chunking strategy for document processing.

Strategy: RecursiveCharacterTextSplitter
─────────────────────────────────────────
Why recursive over alternatives:
  - Fixed-size: Splits mid-sentence, losing semantic coherence.
  - Sentence-based: Uneven chunk sizes, some too small for context.
  - Semantic: Requires an LLM call per split — too slow for ingestion.
  - Recursive: Tries paragraph breaks first, then sentences, then words.
    This preserves meaning at natural boundaries while keeping chunks uniform.

Parameters:
  - chunk_size=500 chars: ~100 words. Large enough for a complete thought,
    small enough to be precise in retrieval. Empirically, 300-600 works well
    for policy/legal text.
  - chunk_overlap=100 chars: ~20% overlap. Ensures context at chunk boundaries
    isn't lost. Important for questions that span two paragraphs.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter with configured parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(text: str, metadata: dict) -> list[dict]:
    """
    Split a document into chunks with metadata attached to each.

    Args:
        text: Full document text.
        metadata: Base metadata dict (source_file, etc.) to attach to every chunk.

    Returns:
        List of dicts, each with 'text', 'metadata' keys.
        Metadata includes: source_file, page_number (if available),
        chunk_index, total_chunks.
    """
    splitter = create_splitter()
    chunks = splitter.split_text(text)

    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_meta = {
            "page_number": 1,  # Default page 1 for non-pdf documents
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        result.append({
            "text": chunk_text,
            "metadata": chunk_meta,
        })

    return result


def chunk_document_by_pages(pages: list[dict], source_file: str) -> list[dict]:
    """
    Split a multi-page document (e.g., PDF) into chunks, preserving page numbers.

    Args:
        pages: List of dicts with 'text' and 'page_number' keys.
        source_file: Filename of the source document.

    Returns:
        List of chunk dicts with page-aware metadata.
    """
    splitter = create_splitter()
    all_chunks = []
    global_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": source_file,
                    "page_number": page["page_number"],
                    "chunk_index": global_index,
                },
            })
            global_index += 1

    # Backfill total_chunks now that we know the count
    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = global_index

    return all_chunks
