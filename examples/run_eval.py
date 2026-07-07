"""
Evaluation runner for the Potens RAG system.

Runs queries from eval_set.json, performs retrieval on the local vector store,
and evaluates if the target source document is present in the top-k results.
If GEMINI_API_KEY is present, it also queries the LLM and displays the answer.

Metrics calculated:
- Retrieval@top-3 accuracy
- Retrieval@top-5 accuracy
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import EXAMPLES_DIR, GROQ_API_KEY
import vector_store
import rag_engine
import language_utils
import llm_client


def load_eval_set() -> list[dict]:
    eval_file = EXAMPLES_DIR / "eval_set.json"
    if not eval_file.exists():
        print(f"❌ Evaluation set not found at {eval_file}")
        return []
    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    eval_items = load_eval_set()
    if not eval_items:
        return

    print("\n" + "=" * 70)
    print("🧪 Running RAG Evaluation Runner")
    print("=" * 70)
    print(f"Total test cases: {len(eval_items)}")
    
    # Check if vector DB has documents
    num_chunks = vector_store.get_collection_count()
    if num_chunks == 0:
        print("❌ Error: Vector DB is empty. Please run ingestion first using: python ingest.py")
        return
    print(f"Database active with {num_chunks} chunks.")
    print("-" * 70)

    has_groq = bool(GROQ_API_KEY)
    if not has_groq:
        print("⚠️  GROQ_API_KEY is not set. Skipping LLM generation evaluation.")
        print("   Only local vector retrieval accuracy will be computed.")
        print("-" * 70)

    hits_top_3 = 0
    hits_top_5 = 0
    results = []

    for item in eval_items:
        qid = item["id"]
        q_text = item["question"]
        expected_doc = item["expected_document"]
        lang = item["language"]

        print(f"\n[Test Case {qid}] Lang: {lang.upper()}")
        print(f"❓ Q: {q_text}")
        print(f"🎯 Target Doc: {expected_doc}")

        # Handle translation if query is Hindi
        retrieval_query = q_text
        if lang != "en":
            if has_groq:
                try:
                    retrieval_query = llm_client.translate_text(q_text, lang, "en")
                    print(f"   🌐 Translated search query: {retrieval_query}")
                except Exception as e:
                    print(f"   ⚠️  Translation failed, searching raw query. Error: {e}")
            else:
                print("   ⚠️  Cannot translate Hindi query without Groq key. Searching in Hindi (retrieval rate may drop).")

        # Vector Store Search + Reranker (same as actual RAG pipeline)
        raw_chunks = vector_store.query(retrieval_query, top_k=10)
        from reranker import rerank_chunks
        top_chunks = rerank_chunks(retrieval_query, raw_chunks, top_k=5)

        # Check hits
        retrieved_files = [c["metadata"].get("source_file") for c in top_chunks]
        
        hit_3 = expected_doc in retrieved_files[:3]
        hit_5 = expected_doc in retrieved_files[:5]

        if hit_3:
            hits_top_3 += 1
            print("   ✅ Hit in top-3")
        elif hit_5:
            hits_top_5 += 1
            print("   ⚠️  Hit in top-5 (missed top-3)")
        else:
            print("   ❌ Missed completely")

        print("   🔍 Chunks retrieved:")
        for idx, chunk in enumerate(top_chunks):
            dist = chunk.get("distance", 0.0)
            fname = chunk["metadata"].get("source_file", "unknown")
            page = chunk["metadata"].get("page_number", "N/A")
            c_idx = chunk["metadata"].get("chunk_index", 0)
            print(f"      {idx+1}. [{fname} | Page {page} | Chunk {c_idx}] (distance: {dist:.3f})")

        # Run full pipeline if API key exists
        llm_answer = "Skipped (no API key)"
        citations = []
        if has_groq:
            try:
                ans_res = rag_engine.ask(q_text)
                llm_answer = ans_res["answer"]
                citations = [c.get("source_file") for c in ans_res["citations"]]
                print(f"   💡 LLM Answer Preview: {llm_answer[:120]}...")
                print(f"   🔖 LLM Citations: {citations}")
            except Exception as e:
                print(f"   ⚠️  LLM call failed: {e}")

        results.append({
            "id": qid,
            "question": q_text,
            "expected_doc": expected_doc,
            "hit_top_3": hit_3,
            "hit_top_5": hit_5 or hit_3,
            "retrieved_docs": retrieved_files,
            "llm_answer": llm_answer,
            "llm_citations": citations
        })

    # Summary
    total = len(eval_items)
    acc_top_3 = (hits_top_3 / total) * 100
    acc_top_5 = ((hits_top_3 + hits_top_5) / total) * 100

    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Retrieval@top-3 accuracy: {acc_top_3:.1f}% ({hits_top_3}/{total})")
    print(f"Retrieval@top-5 accuracy: {acc_top_5:.1f}% ({hits_top_3 + hits_top_5}/{total})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
