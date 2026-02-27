# RAG-based Research Assistant (Transformers)

This project implements a Retrieval-Augmented Generation (RAG) system using:

- FAISS for vector search
- HuggingFace embeddings
- Ollama (Gemma) for local LLM inference
- LangChain for orchestration

## Features
- Supports multiple PDFs
- Uses local LLM
- Prevents hallucinations using strict context
- Modular ingestion and querying

## Setup

```bash
pip install -r requirements.txt
ollama pull gemma:2b
python ingest.py
python rag_chain.py
