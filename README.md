---
title: Insurance RAG Pipeline
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Document Q&A — Production-Grade RAG Pipeline

**Live API:** https://aditivj-insurance-rag-pipeline.hf.space/docs

A retrieval-augmented generation (RAG) system for querying insurance policy and claims documents. Built to demonstrate production-grade ML engineering practices: layout-aware document parsing, two-stage retrieval, semantic caching, cost/latency instrumentation, and a ground-truth eval pipeline.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /ingest | POST | Upload PDF, chunk and embed into ChromaDB |
| /query | POST | Ask a question, returns answer + latency + cost |
| /metrics | GET | Returns p50/p95 latency, cache hit rate, avg cost |
| /docs | GET | Swagger UI for interactive testing |

## Stack
- LLM: Groq LLaMA 3.1 8B
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Vector DB: ChromaDB
- API: FastAPI + Docker
