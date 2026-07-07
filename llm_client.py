"""
LLM client for Google Gemini API.

Handles all interactions with the Gemini model, including:
- Q&A with citation extraction
- Contradiction detection between documents
- Language translation for multilingual support
- Confidence scoring

All prompts are engineered to prevent hallucination and enforce
structured output.
"""

import json
import re
import time
import requests
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_OUTPUT_TOKENS


def _strip_code_fences(text: str) -> str:
    """Robustly remove markdown code fences (like ```json ... ```) from a text response."""
    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\n?(.*?)\n?```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _generate_content_with_retry(prompt: str, max_retries: int = 3, initial_delay: float = 2.0) -> str:
    """
    Call the Groq API Chat Completions endpoint with exponential backoff retry on HTTP status 429 (Rate Limit).
    
    Returns:
        The content string from the response.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please set it in your .env file or input it in the sidebar. "
            "Get a Groq key at: https://console.groq.com/keys"
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": GROQ_TEMPERATURE,
        "max_tokens": GROQ_MAX_OUTPUT_TOKENS
    }

    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # If rate limited (status 429), raise an exception to trigger the retry logic
            if response.status_code == 429:
                raise requests.exceptions.RequestException("Rate Limit 429 Exceeded")
            
            response.raise_for_status()
            
            # Parse OpenAI-compatible response body
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                raise ValueError(f"Unexpected response format from Groq API: {data}")
                
        except (requests.exceptions.RequestException, Exception) as e:
            # Check if this error is due to a rate limit
            is_rate_limit = False
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    is_rate_limit = True
            if "429" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower() or "resource_exhausted" in str(e).lower():
                is_rate_limit = True
                
            if is_rate_limit:
                if attempt == max_retries:
                    raise RuntimeError("Groq API Rate Limit Exceeded after retries.") from e
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e

    raise RuntimeError("Unreachable")


def ask_with_citations(query: str, context_chunks: list[dict]) -> dict:
    """
    Answer a question using retrieved document chunks, with citations.

    The prompt is carefully designed to:
    1. Force the model to cite specific chunks.
    2. Refuse to answer if the context doesn't contain the information.
    3. Return a confidence score.

    Args:
        query: The user's question.
        context_chunks: List of dicts with 'text', 'metadata' keys.

    Returns:
        Dict with 'answer', 'citations', 'confidence', 'no_answer' keys.
    """
    # Build context string with numbered references
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "N/A")
        chunk_idx = meta.get("chunk_index", i)
        context_parts.append(
            f"[Source {i+1}] File: {source} | Page: {page} | Chunk: {chunk_idx}\n"
            f"{chunk['text']}\n"
        )

    context_str = "\n---\n".join(context_parts)

    prompt = f"""You are a precise document Q&A assistant. Answer the user's question ONLY using the provided document excerpts below. Follow these rules strictly:

RULES:
1. ONLY use information from the provided excerpts. Do NOT use any external knowledge.
2. If the provided excerpts do NOT contain enough information to answer the question, you MUST respond with:
   {{"answer": "The provided documents do not contain sufficient information to answer this question.", "citations": [], "confidence": 0.0, "no_answer": true}}
3. For every claim in your answer, cite the specific source using [Source N] format.
4. Provide a confidence score from 0.0 to 1.0 based on how well the excerpts support your answer.
5. Return your response as a valid JSON object with these exact keys: "answer", "citations", "confidence", "no_answer".

DOCUMENT EXCERPTS:
{context_str}

USER QUESTION: {query}

Respond with a JSON object in this exact format:
{{
    "answer": "Your detailed answer here with [Source N] citations inline.",
    "citations": [
        {{
            "source_number": 1,
            "source_file": "filename.txt",
            "page_or_chunk": "Page X, Chunk Y",
            "snippet": "The exact text snippet used from this source"
        }}
    ],
    "confidence": 0.85,
    "no_answer": false
}}"""

    response_text = _generate_content_with_retry(prompt)

    # Parse the JSON response
    try:
        text = _strip_code_fences(response_text)
        result = json.loads(text)
        # Ensure confidence is a float
        if "confidence" in result:
            try:
                result["confidence"] = float(result["confidence"])
            except (ValueError, TypeError):
                result["confidence"] = 0.0
    except (json.JSONDecodeError, AttributeError):
        # Fallback: return raw text as answer
        result = {
            "answer": response_text if response_text else "Failed to generate response.",
            "citations": [],
            "confidence": 0.0,
            "no_answer": False,
        }

    return result


def detect_contradictions(doc1_chunks: list[dict], doc2_chunks: list[dict],
                          doc1_name: str, doc2_name: str, topic: str = "") -> dict:
    """
    Analyze two documents for contradictions on a given topic.

    Args:
        doc1_chunks: Chunks from the first document.
        doc2_chunks: Chunks from the second document.
        doc1_name: Name of the first document.
        doc2_name: Name of the second document.
        topic: Optional topic to focus the comparison on.

    Returns:
        Dict with contradiction analysis results.
    """
    # Limit chunks to avoid token limits (take first 8 from each)
    doc1_text = "\n\n".join([c["text"] for c in doc1_chunks[:8]])
    doc2_text = "\n\n".join([c["text"] for c in doc2_chunks[:8]])

    topic_instruction = f'Focus specifically on the topic: "{topic}".' if topic else "Analyze for any contradictions across all topics covered."

    prompt = f"""You are a legal and policy document analyst. Compare the following excerpts from two documents and identify any contradictions, conflicts, or inconsistencies between them.

{topic_instruction}

DOCUMENT 1: {doc1_name}
{doc1_text}

---

DOCUMENT 2: {doc2_name}
{doc2_text}

Analyze carefully and respond with a JSON object in this exact format:
{{
    "has_contradiction": true/false,
    "contradictions": [
        {{
            "topic": "The specific topic where contradiction exists",
            "doc1_position": "What Document 1 states on this topic",
            "doc2_position": "What Document 2 states on this topic",
            "reasoning": "Detailed explanation of why these positions conflict"
        }}
    ],
    "summary": "A brief overall summary of the comparison"
}}

If there are no contradictions, return an empty contradictions array and explain in the summary why the documents are consistent."""

    response_text = _generate_content_with_retry(prompt)

    try:
        text = _strip_code_fences(response_text)
        result = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        result = {
            "has_contradiction": False,
            "contradictions": [],
            "summary": response_text if response_text else "Failed to analyze contradictions.",
        }

    return result


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text between languages using Gemini.

    Args:
        text: Text to translate.
        source_lang: Source language code (e.g., 'hi', 'en').
        target_lang: Target language code.

    Returns:
        Translated text.
    """
    lang_names = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    prompt = f"""Translate the following text from {source_name} to {target_name}. 
Provide ONLY the translated text, nothing else. Do not add explanations or notes.

Text to translate:
{text}"""

    response_text = _generate_content_with_retry(prompt)

    return response_text.strip() if response_text else text
