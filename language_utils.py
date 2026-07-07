"""
Language detection and translation utilities.

Handles the multilingual flow:
1. Detect the language of the incoming query.
2. If non-English, translate query to English for retrieval.
3. After getting the answer, translate it back to the original language.

This "translate at the boundary" approach is practical for a 24-hour build:
- Embeddings work best with English text (all-MiniLM-L6-v2 is English-trained).
- Gemini handles translation well for Indian languages.
- The retrieval quality stays high because we search in the same language
  as the documents (English).
"""

from langdetect import detect, LangDetectException
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: Input text to analyze.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'hi').
        Falls back to DEFAULT_LANGUAGE on detection failure.
    """
    try:
        lang = detect(text)
        # Map to supported languages, default to 'en' if unsupported
        if lang in SUPPORTED_LANGUAGES:
            return lang
        return DEFAULT_LANGUAGE
    except LangDetectException:
        return DEFAULT_LANGUAGE


def needs_translation(lang: str) -> bool:
    """Check if the detected language requires translation for retrieval."""
    return lang != "en"


def get_language_name(lang_code: str) -> str:
    """Get the human-readable name for a language code."""
    return SUPPORTED_LANGUAGES.get(lang_code, lang_code)
