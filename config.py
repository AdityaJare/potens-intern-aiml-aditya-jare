"""
Configuration constants for the RAG Document Q&A system.
All tunable parameters are centralized here for easy experimentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
EXAMPLES_DIR = BASE_DIR / "examples"

# ── Groq LLM ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"     # High quality, fast Llama 3.3 model
GROQ_TEMPERATURE = 0.2
GROQ_MAX_OUTPUT_TOKENS = 2048

# ── Embeddings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # 384-dim, fast, good quality
EMBEDDING_DIMENSION = 384

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 500                            # Characters per chunk
CHUNK_OVERLAP = 100                         # Overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Recursive split order

# ── Vector Store ─────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"
TOP_K_RESULTS = 5                           # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3                  # Minimum similarity score (0-1)

# ── Multilingual ─────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
}
DEFAULT_LANGUAGE = "en"

# ── Confidence ───────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5                  # Below this → "low confidence" warning
