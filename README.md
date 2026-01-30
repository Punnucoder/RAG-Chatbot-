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
