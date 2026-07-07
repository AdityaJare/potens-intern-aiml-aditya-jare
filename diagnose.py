import sys
import os

print("=" * 60)
print("🐍 PYTHON ENVIRONMENT DIAGNOSTICS")
print("=" * 60)
print(f"Python Executable: {sys.executable}")
print(f"Current Working Directory: {os.getcwd()}")
print("\nPython Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")

print("\nAttempting to import packages:")
packages = [
    "streamlit",
    "chromadb",
    "sentence_transformers",
    "google.generativeai",
    "fitz",
    "langdetect",
    "langchain_text_splitters"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}: SUCCESS")
    except ImportError as e:
        print(f"  ❌ {pkg}: FAILED ({e})")
print("=" * 60)
