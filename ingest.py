"""
Document ingestion pipeline.

Reads all documents from the docs/ folder, parses them (PDF or text),
chunks them, embeds them, and stores them in ChromaDB.

Usage:
    python ingest.py              # Ingest all documents
    python ingest.py --clear      # Clear existing data and re-ingest
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF

from config import DOCS_DIR
from chunking import chunk_document, chunk_document_by_pages
from vector_store import add_documents, get_collection_count, clear_collection


def parse_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Returns:
        List of dicts with 'text' and 'page_number' keys.
    """
    pages = []
    doc = fitz.open(str(file_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:  # Skip blank pages
            pages.append({
                "text": text,
                "page_number": page_num + 1,  # 1-indexed
            })
    doc.close()
    return pages


def parse_text_file(file_path: Path) -> str:
    """Read a plain text or markdown file."""
    return file_path.read_text(encoding="utf-8")


def ingest_document(file_path: Path) -> int:
    """
    Ingest a single document: parse → chunk → embed → store.

    Returns:
        Number of chunks created.
    """
    suffix = file_path.suffix.lower()
    source_file = file_path.name

    print(f"  📄 Processing: {source_file}")

    if suffix == ".pdf":
        pages = parse_pdf(file_path)
        if not pages:
            print(f"    ⚠️  No text extracted from {source_file}")
            return 0
        chunks = chunk_document_by_pages(pages, source_file)
    elif suffix in (".txt", ".md"):
        text = parse_text_file(file_path)
        if not text.strip():
            print(f"    ⚠️  Empty file: {source_file}")
            return 0
        chunks = chunk_document(text, {"source_file": source_file})
    else:
        print(f"    ⚠️  Unsupported format: {suffix}")
        return 0

    print(f"    → {len(chunks)} chunks created")

    num_added = add_documents(chunks)
    print(f"    ✅ {num_added} chunks stored in vector DB")

    return num_added


def ingest_all(clear_first: bool = False):
    """
    Ingest all documents from the docs/ directory.

    Args:
        clear_first: If True, clears existing vector store before ingestion.
    """
    if not DOCS_DIR.exists():
        print(f"❌ Documents directory not found: {DOCS_DIR}")
        print("   Please add your documents to the 'docs/' folder.")
        return

    supported_extensions = {".pdf", ".txt", ".md"}
    doc_files = [
        f for f in sorted(DOCS_DIR.iterdir())
        if f.suffix.lower() in supported_extensions
    ]

    if not doc_files:
        print(f"❌ No supported documents found in {DOCS_DIR}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return

    print(f"\n{'='*60}")
    print(f"📚 Document Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"   Source directory: {DOCS_DIR}")
    print(f"   Documents found: {len(doc_files)}")

    if clear_first:
        print("\n🗑️  Clearing existing vector store...")
        clear_collection()

    print(f"\n{'─'*60}")
    total_chunks = 0
    for doc_file in doc_files:
        chunks = ingest_document(doc_file)
        total_chunks += chunks
        print()

    print(f"{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Total chunks stored: {total_chunks}")
    print(f"   Vector store size: {get_collection_count()} chunks")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    clear_flag = "--clear" in sys.argv
    ingest_all(clear_first=clear_flag)
