"""
Streamlit Web UI for the Potens Document Q&A and Contradiction Detection RAG system.

Features:
- Main dashboard styled with modern, dark-mode-optimized components.
- Ask Q&A Tab: Text/Voice query input, auto-language detection, expandable citations,
  and colored confidence badges.
- Contradict Tab: Select any two documents and run cross-document conflict analysis
  on a specific topic.
- Documents Tab: Visualizes ingested documents, status, chunk counts, and allows
  initiating a manual ingestion pipeline.
"""

import streamlit as st
import os
from pathlib import Path

# Set page config first
st.set_page_config(
    page_title="Potens Operations & Policy Cockpit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    /* Premium visual styling */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .citation-card {
        background-color: #1e293b;
        border-left: 4px solid #8b5cf6;
        padding: 1rem;
        border-radius: 0.375rem;
        margin-bottom: 0.75rem;
    }
    .citation-header {
        font-weight: 600;
        color: #c084fc;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .citation-snippet {
        font-style: italic;
        color: #cbd5e1;
        font-size: 0.85rem;
    }
    .confidence-high {
        background-color: #064e3b;
        color: #34d399;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .confidence-medium {
        background-color: #78350f;
        color: #fbbf24;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .confidence-low {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

import config
import rag_engine
import vector_store
from ingest import ingest_all

# Title Section
st.markdown("<h1 class='main-title'>Potens Operations & Policy Cockpit</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Production-grade RAG engine for Indian policy, legal, and compliance questions.</p>", unsafe_allow_html=True)

# API Key validation helper
def check_api_key():
    if not config.GROQ_API_KEY:
        st.sidebar.error("⚠️ GROQ_API_KEY is not set.")
        key_input = st.sidebar.text_input("Enter Groq API Key:", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            config.GROQ_API_KEY = key_input
            st.sidebar.success("API Key updated for session.")
            st.rerun()
        return False
    return True

has_api_key = check_api_key()

# Sidebar: Corpus Info & Re-ingestion
st.sidebar.title("📚 Policy Database")

# Get list of ingested documents
try:
    ingested_docs = rag_engine.get_available_documents()
    db_count = vector_store.get_collection_count()
except Exception as e:
    ingested_docs = []
    db_count = 0

st.sidebar.metric(label="Total Ingested Chunks", value=db_count)

if ingested_docs:
    st.sidebar.write("**Ingested Files:**")
    for doc in ingested_docs:
        st.sidebar.caption(f"📄 {doc}")
else:
    st.sidebar.warning("No documents ingested yet.")

st.sidebar.markdown("---")
st.sidebar.write("**System Ingestion Action:**")
if st.sidebar.button("Re-run Ingestion (Clear & Index)"):
    with st.spinner("Ingesting documents... This will take a moment."):
        try:
            ingest_all(clear_first=True)
            st.sidebar.success("Ingestion complete!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("**Tech Stack Details:**")
st.sidebar.caption("LLM: `llama-3.3-70b-versatile` (Groq)")
st.sidebar.caption("Embeddings: `all-MiniLM-L6-v2` (local)")
st.sidebar.caption("Database: ChromaDB (persistent)")

# Main interface tabs
tab_ask, tab_contradict, tab_docs = st.tabs([
    "🔍 Document Q&A", 
    "⚖️ Cross-Doc Contradiction", 
    "📁 Source Documents"
])

# ── TAB 1: ASK A QUESTION ──────────────────────────────────────────
with tab_ask:
    st.subheader("Ask a Question across Ingested Policies")
    st.caption("Auto-detects queries in English, Hindi, and other Indian languages.")

    # Input Method Selection
    input_method = st.radio("Choose Input Method:", ["Text Input", "Voice Input (Requires Microphone)"], horizontal=True)

    query_input = ""
    
    if input_method == "Text Input":
        query_input = st.text_input(
            "Enter your query/question:",
            placeholder="e.g., What is the penalty for failure to protect data under the DPDPA?"
        )
    else:
        # Web Speech API Voice input integration via Custom HTML/JS in an iframe
        # Since streamlit runs on server side, we can use a custom JS component to capture speech
        # and pass it back. Let's provide a convenient text area but also a small browser speech recognition helper.
        st.info("💡 Web Speech API is supported through browser integration. Speak when prompt starts.")
        
        # Audio capturing helper using HTML5 & Streamlit experimental components or simple speech trigger instructions
        # Since Python speech-recognition requires PyAudio (which fails on windows headless/containers),
        # we will use Web Speech API in browser.
        # Let's render a HTML5 voice input button that uses Web Speech API.
        import streamlit.components.v1 as components
        
        voice_js = """
        <div style="font-family: sans-serif; background-color: #1e293b; padding: 15px; border-radius: 8px;">
            <button id="start-btn" style="background-color: #3b82f6; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; font-weight: bold;">
                🎤 Click to Speak
            </button>
            <p id="status" style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">Click to start speaking...</p>
            <script>
                const startBtn = document.getElementById('start-btn');
                const status = document.getElementById('status');
                
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    const recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    
                    // Default to Indian English / Hindi detection
                    recognition.lang = 'en-IN';
                    
                    startBtn.onclick = () => {
                        recognition.start();
                        status.textContent = 'Listening... Speak now.';
                        startBtn.style.backgroundColor = '#ef4444';
                    };
                    
                    recognition.onresult = (event) => {
                        const transcript = event.results[0][0].transcript;
                        status.textContent = 'Transcript: ' + transcript;
                        startBtn.style.backgroundColor = '#3b82f6';
                        
                        // Send transcript back to Streamlit
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: transcript
                        }, '*');
                    };
                    
                    recognition.onerror = (event) => {
                        status.textContent = 'Error: ' + event.error;
                        startBtn.style.backgroundColor = '#3b82f6';
                    };
                    
                    recognition.onend = () => {
                        if (status.textContent === 'Listening... Speak now.') {
                            status.textContent = 'Stopped listening.';
                            startBtn.style.backgroundColor = '#3b82f6';
                        }
                    };
                } else {
                    status.textContent = 'Speech recognition not supported in this browser.';
                    startBtn.disabled = true;
                }
            </script>
        </div>
        """
        st.markdown("**Web Speech API (Browser Recognition):**", unsafe_allow_html=True)
        # Handle the component return value
        voice_val = components.html(voice_js, height=120)
        
        # Standard input backup in case the iframe value isn't read by this version of Streamlit
        query_input = st.text_input(
            "Speech Transcript / Custom Query Text:",
            placeholder="Type speech transcript here if voice recognition did not fill it automatically."
        )

    # Initialize session state for Q&A history
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "flagged" not in st.session_state:
        st.session_state.flagged = False

    if st.button("Submit Query", key="ask_btn") and query_input:
        if not has_api_key:
            st.error("Please add a Groq API Key first.")
        else:
            with st.spinner("Analyzing documents & generating answer..."):
                try:
                    res = rag_engine.ask(query_input)
                    st.session_state.last_query = query_input
                    st.session_state.last_result = res
                    st.session_state.flagged = False
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg or "resourceexhausted" in err_msg or "rate limit" in err_msg:
                        st.error("🛑 **Groq API Rate Limit / Quota Exceeded**")
                        st.warning(
                            "You are currently using the Groq API. "
                            "You have hit a rate limit (requests per minute or daily limit). "
                            "Please wait about 30 seconds and try again, or configure a different API Key in the sidebar."
                        )
                    else:
                        st.error(f"Error executing query: {e}")
                        st.exception(e)

    # Render persistent last result from session state if available
    if st.session_state.last_result:
        res = st.session_state.last_result
        orig_q = st.session_state.last_query

        # Language notification
        if res["language"] != "en":
            st.info(f"🌐 Query detected in **{res['language_name']}**. Output translated to match.")
        
        # Confidence scoring display
        score = res.get("confidence", 0.0)
        level = res.get("confidence_level", "low")
        if level == "high":
            badge_html = f"<span class='confidence-high'>High Confidence ({score:.2f})</span>"
        elif level == "medium":
            badge_html = f"<span class='confidence-medium'>Medium Confidence ({score:.2f})</span>"
        else:
            badge_html = f"<span class='confidence-low'>Low Confidence ({score:.2f})</span>"
            
        st.markdown(f"**Confidence Level:** {badge_html}", unsafe_allow_html=True)
        
        # Output Answer
        st.markdown("### Answer")
        st.write(res.get("answer", "No answer generated."))
        
        # Human in the Loop workflow (Low Confidence action)
        if level == "low" or res.get("no_answer"):
            st.warning("⚠️ **Low Confidence Warning:** The retrieved sources may not completely address this question or the evidence is weak. You should flag this for administrative review.")
            
            if not st.session_state.flagged:
                if st.button("📥 Flag for Human Review", key="flag_btn"):
                    status_msg = rag_engine.flag_for_review(
                        orig_q,
                        res["answer"],
                        res["confidence"],
                        res["language"]
                    )
                    st.session_state.flagged = True
                    st.success(status_msg)
            else:
                st.info("✅ This response has been flagged and recorded to the Human Review Log.")

        # Citations
        if res["citations"]:
            st.markdown("### Sources Cited")
            for cit in res["citations"]:
                c_num = cit.get("source_number", "?")
                c_file = cit.get("source_file", "unknown")
                c_loc = cit.get("page_or_chunk", "unknown")
                c_snip = cit.get("snippet", "")
                
                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-header">[Source {c_num}] File: {c_file} | Reference: {c_loc}</div>
                    <div class="citation-snippet">"{c_snip}"</div>
                </div>
                """, unsafe_allow_html=True)
        elif res["no_answer"]:
            st.warning("No source documents contained sufficient information to verify this query. System refused to hallucinate.")
        else:
            st.caption("No specific citations generated.")

# ── TAB 2: DETECT CONTRADICTIONS ───────────────────────────────────
with tab_contradict:
    st.subheader("Detect Cross-Document Inconsistencies")
    st.write("Compare two different policy documents for contradictions, misalignments, or inconsistencies.")
    
    if len(ingested_docs) < 2:
        st.warning("Please ingest at least 2 documents to compare them.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            doc1 = st.selectbox("Select First Document:", ingested_docs, key="doc1_select")
        with col2:
            doc2_options = [d for d in ingested_docs if d != doc1]
            if doc2_options:
                doc2 = st.selectbox("Select Second Document:", doc2_options, key="doc2_select")
            else:
                st.warning("No other documents available to compare.")
                doc2 = None
            
        topic = st.text_input(
            "Comparison Topic (Optional):",
            placeholder="e.g., Penalties, Localisation rules, Data collection limitations..."
        )
        
        if doc1 and doc2:
            if st.button("Analyze Inconsistencies", key="contradict_btn"):
                if not has_api_key:
                    st.error("Please add a Groq API Key first.")
                else:
                    with st.spinner("Analyzing document differences..."):
                        try:
                            res = rag_engine.contradict(doc1, doc2, topic)
                            
                            if "error" in res:
                                st.error(res["error"])
                            else:
                                st.markdown("### Comparison Summary")
                                st.write(res.get("summary", ""))
                                
                                if res.get("has_contradiction"):
                                    st.markdown("### ⚠️ Contradictions Found")
                                    for con in res.get("contradictions", []):
                                        with st.expander(f"Topic: {con.get('topic', 'Unknown')}"):
                                            st.markdown(f"**Document 1 ({doc1}):**")
                                            st.info(con.get("doc1_position", ""))
                                            st.markdown(f"**Document 2 ({doc2}):**")
                                            st.info(con.get("doc2_position", ""))
                                            st.markdown("**Why they conflict:**")
                                            st.warning(con.get("reasoning", ""))
                                else:
                                    st.success("No contradictions detected. The documents appear consistent on the topics analyzed.")
                            
                        except Exception as e:
                            err_msg = str(e).lower()
                            if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg or "resourceexhausted" in err_msg or "rate limit" in err_msg:
                                st.error("🛑 **Groq API Rate Limit / Quota Exceeded**")
                                st.warning(
                                    "You are currently using the Groq API. "
                                    "You have hit a rate limit (requests per minute or daily limit). "
                                    "Please wait about 30 seconds and try again, or configure a different API Key in the sidebar."
                                )
                            else:
                                st.error(f"Comparison failed: {e}")
                                st.exception(e)

# ── TAB 3: DOCUMENTS ───────────────────────────────────────────────
with tab_docs:
    st.subheader("Source Documents Database")
    st.write("Current source files located in the `docs/` directory:")
    
    docs_path = Path("docs")
    if docs_path.exists():
        files = list(docs_path.iterdir())
        if files:
            for f in files:
                suffix = f.suffix.lower()
                size = f.stat().st_size
                st.write(f"- 📄 **{f.name}** ({suffix[1:].upper()} file, {size/1024:.1f} KB)")
        else:
            st.warning("No files found in the `docs/` folder.")
    else:
        st.warning("`docs/` directory does not exist.")
