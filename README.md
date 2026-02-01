# RAG Q&A Chatbot (Streamlit + ChromaDB)

This project ingests documents (PDF/TXT/DOCX), chunks them, stores embeddings in ChromaDB, and provides a Streamlit chat UI for retrieval-augmented generation (RAG) with source attribution.

Quick steps

1. Create a Python venv and install dependencies:

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare documents

Put your documents into `data/documents/` (PDF, DOCX, TXT) or upload them via the Streamlit UI.

3. Run the app

```bash
streamlit run src/app.py
```

4. Optional: set `OPENAI_API_KEY` environment variable to enable LLM answer generation (gpt-3.5-turbo). Without it the app will return extractive context snippets.

Notes

- The ingestion uses `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings and stores vectors in `chroma_db/` for persistence.
- Chunking is done with ~700 characters and 10% overlap.
- Evaluation expects `tests/questions.txt` and `tests/answers.txt` — one per line; the app eval button will run up to 20 tests and return accuracy/time metrics.

Deployment

- You can deploy this app on Streamlit Community Cloud, Render, or similar providers. Make sure to set any required environment variables (`OPENAI_API_KEY`) in your deployment settings.
# RAG Chatbot (Local LLM using Ollama)

This project implements a Retrieval-Augmented Generation (RAG) chatbot
using a locally running LLM via Ollama.

## Features
- No API keys required
- Fully offline after model download
- Context-aware answers only
- Uses Mistral / LLaMA3 models locally

## Tech Stack
- Python
- Streamlit
- Ollama
- Vector Store (FAISS / Chroma)

## How to Run
1. Install Ollama
2. Pull model: `ollama pull mistral`
3. Install dependencies: `pip install -r requirements.txt`
4. Run app: `streamlit run app.py`
