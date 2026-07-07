"""
Core RAG engine — the brain of the system.

Orchestrates the full retrieval-augmented generation pipeline:
  Query → Language Detection → (Translation) → Embedding → Retrieval →
  LLM Generation → (Translation back) → Structured Response

Two main endpoints:
  - ask(): Answer questions with citations
  - contradict(): Detect contradictions between two documents
"""

from vector_store import query as vector_query, get_chunks_by_document, list_documents
from llm_client import ask_with_citations, detect_contradictions, translate_text
from language_utils import detect_language, needs_translation, get_language_name
from config import CONFIDENCE_THRESHOLD
from reranker import rerank_chunks


def ask(user_query: str) -> dict:
    """
    Answer a question using the RAG pipeline.

    Flow:
    1. Detect language of the query.
    2. If non-English → translate to English for retrieval.
    3. Embed and search the vector store for relevant chunks (retrieve top 10).
    4. Rerank top 10 candidate chunks down to top 5 using Cross-Encoder.
    5. Send chunks + query to LLM for answer generation.
    6. If original language was non-English → translate answer back.
    7. Return structured response with citations and confidence.

    Args:
        user_query: The user's question in any supported language.

    Returns:
        Dict with keys: answer, citations, confidence, language,
        language_name, no_answer, original_query, retrieval_query,
        num_chunks_retrieved.
    """
    # Step 1: Detect language
    detected_lang = detect_language(user_query)
    language_name = get_language_name(detected_lang)

    # Step 2: Translate to English if needed
    retrieval_query = user_query
    if needs_translation(detected_lang):
        retrieval_query = translate_text(user_query, detected_lang, "en")

    # Step 3: Retrieve relevant chunks (retrieve top 10 for reranking)
    raw_chunks = vector_query(retrieval_query, top_k=10)

    if not raw_chunks:
        return {
            "answer": "No documents have been ingested yet. Please run the ingestion pipeline first.",
            "citations": [],
            "confidence": 0.0,
            "language": detected_lang,
            "language_name": language_name,
            "no_answer": True,
            "original_query": user_query,
            "retrieval_query": retrieval_query,
            "num_chunks_retrieved": 0,
        }

    # Step 4: Rerank chunks using Cross-Encoder down to top 5
    retrieved_chunks = rerank_chunks(retrieval_query, raw_chunks, top_k=5)

    # Step 5: Generate answer with citations
    llm_result = ask_with_citations(retrieval_query, retrieved_chunks)

    answer = llm_result.get("answer", "Failed to generate an answer.")
    citations = llm_result.get("citations", [])
    try:
        confidence = float(llm_result.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    no_answer = llm_result.get("no_answer", False)

    # Step 6: Translate answer back if needed
    if needs_translation(detected_lang) and not no_answer:
        answer = translate_text(answer, "en", detected_lang)

    # Step 7: Build response
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "confidence_level": _confidence_label(confidence),
        "language": detected_lang,
        "language_name": language_name,
        "no_answer": no_answer,
        "original_query": user_query,
        "retrieval_query": retrieval_query,
        "num_chunks_retrieved": len(retrieved_chunks),
    }


def contradict(doc1_name: str, doc2_name: str, topic: str = "") -> dict:
    """
    Detect contradictions between two documents.

    Args:
        doc1_name: Filename of the first document.
        doc2_name: Filename of the second document.
        topic: Optional topic to focus the comparison on.

    Returns:
        Dict with contradiction analysis results.
    """
    # Retrieve all chunks for each document
    doc1_chunks = get_chunks_by_document(doc1_name)
    doc2_chunks = get_chunks_by_document(doc2_name)

    if not doc1_chunks:
        return {
            "error": f"No chunks found for document: {doc1_name}",
            "has_contradiction": False,
            "contradictions": [],
            "summary": "",
        }

    if not doc2_chunks:
        return {
            "error": f"No chunks found for document: {doc2_name}",
            "has_contradiction": False,
            "contradictions": [],
            "summary": "",
        }

    # Send to LLM for contradiction analysis
    result = detect_contradictions(
        doc1_chunks, doc2_chunks,
        doc1_name, doc2_name,
        topic
    )

    result["doc1"] = doc1_name
    result["doc2"] = doc2_name
    result["doc1_chunks_analyzed"] = min(len(doc1_chunks), 8)
    result["doc2_chunks_analyzed"] = min(len(doc2_chunks), 8)
    result["topic"] = topic if topic else "All topics"

    return result


def get_available_documents() -> list[str]:
    """Return list of ingested document names."""
    return list_documents()


def _confidence_label(score: float) -> str:
    """Map confidence score to a human-readable label."""
    if score >= 0.8:
        return "high"
    elif score >= CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def flag_for_review(query: str, answer: str, confidence: float, language: str) -> str:
    """
    Log a low-confidence or query of interest to a file for human review.
    Implements the Human-in-the-Loop stretch goal.
    """
    import json
    import time
    from config import BASE_DIR
    
    log_file = BASE_DIR / "human_review_log.json"
    
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "language": language,
        "status": "pending_review"
    }
    
    # Read existing
    entries = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []
            
    entries.append(entry)
    
    # Write back
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        return f"Successfully logged query for human review."
    except Exception as e:
        return f"Failed to save log: {e}"
