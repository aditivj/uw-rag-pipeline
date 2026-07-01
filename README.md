# Insurance Document Q&A — Production-Grade RAG Pipeline

A retrieval-augmented generation (RAG) system for querying insurance policy and claims documents. Built to demonstrate production-grade ML engineering practices: layout-aware document parsing, two-stage retrieval, semantic caching, cost/latency instrumentation, and a ground-truth eval pipeline.

---

## Architecture

    PDF Documents
         |
         v
    pdfplumber (table-aware extraction)
         |  |-- Table rows: column-labeled, section-tagged chunks
         |  +-- Prose: RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
         v
    sentence-transformers (all-MiniLM-L6-v2) -- local embeddings
         |
         v
    ChromaDB -- persistent vector store
         |
         v
    Two-stage retrieval:
      1. Embedding similarity -> top-10 candidates
      2. CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) -> top-3
         |
         |-- Semantic cache (cosine similarity > 0.92) -> skip LLM if hit
         v
    Groq LLaMA 3.1 8B -- answer generation
         |
         v
    FastAPI -- REST API with latency + cost logging per request

---

## Stack

| Component | Technology |
|---|---|
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| Vector DB | ChromaDB (persistent) |
| PDF parsing | pdfplumber + LangChain PyPDFLoader |
| API | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |

---

## Key Engineering Decisions

### 1. Two-stage retrieval
Embedding similarity alone is fast but imprecise. A cross-encoder reranker scores query-chunk pairs directly, improving precision at the cost of latency. We use embeddings to retrieve top-10 candidates cheaply, then rerank to top-3 for LLM context. This reduced irrelevant chunks reaching the LLM without significantly increasing end-to-end latency.

### 2. Layout-aware table extraction
Standard PDF text extraction (PyPDFLoader) flattens tables into undifferentiated prose blocks. Multi-column insurance tables (coverage limits, exclusions, premium breakdowns) lost their structure entirely, causing retrieval failures on specific value queries.

Fix: Used pdfplumber to detect table boundaries, extract rows independently, and reconstruct each row as a labeled string:

    [SECTION III - EXCLUSIONS] Cyber Liability - Description: Loss arising from electronic data...
    [COVERAGE SUMMARY] Coverage A - Dwelling - Limit (INR): 85,00,000, Deductible (INR): 10,000

This preserved column semantics and section context, directly improving retrieval quality on exclusion and coverage queries.

### 3. Section-heading context tagging
Table rows extracted in isolation lose their parent section context. A row reading "Cyber Liability: Loss arising from electronic data..." is ambiguous -- is it a coverage or an exclusion? We built a heading-detection algorithm using pdfplumber word bounding boxes to walk upward from each table and attach the nearest section heading to every row chunk.

### 4. Semantic caching
Repeated or semantically similar queries bypass embedding lookup, reranking, and LLM generation entirely. Cache uses cosine similarity with a 0.92 threshold. Measured latency reduction: 8,716ms to 673ms (87%) on repeated queries.

Production consideration: In-memory cache has no TTL or invalidation. After re-ingestion with updated documents, stale cache entries caused incorrect answers until server restart. In production this requires a Redis-backed cache with TTL eviction tied to ingestion events.

### 5. Per-request cost and latency instrumentation
Every request logs: timestamp, latency_ms, input_tokens, output_tokens, estimated_cost_usd, cache_hit. Accessible via GET /metrics which returns p50/p95 latency, cache hit rate, and average cost per query.

---

## Eval Results

Ground truth: 20 hand-labeled Q&A pairs across 3 insurance documents.

| Stage | Keyword Pass Rate | True Pass Rate |
|---|---|---|
| Baseline (PyPDFLoader only) | 50% | ~55% |
| + Table-aware extraction | 60% | ~70% |
| + Section heading tagging | 60% | ~75% |
| + Column label tagging | 60% | 85% |

Note: strict keyword matching undercounts correct answers phrased differently. True pass rate assessed via primary keyword match + manual review. In production this would be replaced with an LLM-as-judge scorer or RAGAS.

### Latency benchmarks (20-query eval, cold run)
- P50: 681ms
- P95: 7,557ms
- Avg cost per query: $0.000291
- Cache hit rate: 0% (no repeated queries in eval set)

---

## Known Limitations and Production Gaps

| Limitation | Root Cause | Production Fix |
|---|---|---|
| Semantic cache has no TTL | In-memory dict, no expiry | Redis with TTL + invalidation on re-ingest |
| Re-ingestion not idempotent | ChromaDB upsert by filename+index | Content-hash based chunk IDs |
| Keyword eval scorer undercounts | Strict string match | LLM-as-judge or RAGAS |
| High P95 latency (7.5s) | Larger document retrieval overhead | Async retrieval, batched reranking |
| No hallucination detection | No faithfulness verification | Add RAGAS faithfulness scorer |
| No cross-document reasoning | Single-document retrieval only | Metadata filtering + multi-hop retrieval |

---

## Project Structure

    insurance-rag/
    |-- app/
    |   |-- main.py          # FastAPI endpoints (/ingest, /query, /metrics)
    |   |-- ingest.py        # PDF parsing: table-aware + prose chunking
    |   |-- retriever.py     # Two-stage retrieval + semantic cache
    |   |-- llm.py           # Groq LLaMA prompt + token tracking
    |   +-- logger.py        # Per-request latency + cost logging
    |-- data/docs/           # Insurance PDF documents
    |-- evals/
    |   |-- ground_truth.json  # 20 hand-labeled Q&A pairs
    |   |-- run_eval.py        # Eval runner
    |   +-- eval_results.json  # Latest eval output
    |-- logs/
    |   +-- requests.jsonl   # Request-level latency + cost log
    |-- Dockerfile
    |-- docker-compose.yml
    +-- README.md

---

## Running Locally

    # 1. Clone and set up
    git clone <repo-url>
    cd insurance-rag
    pip install -r requirements.txt

    # 2. Set Groq API key (free at console.groq.com)
    export GROQ_API_KEY="your_key_here"

    # 3. Start the API
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

    # 4. Ingest documents via Swagger UI
    # Go to http://localhost:8001/docs -> POST /ingest

    # 5. Run eval
    python evals/run_eval.py

---

## Running with Docker

    docker-compose up --build

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /ingest | POST | Upload PDF, chunk and embed into ChromaDB |
| /query | POST | Ask a question, returns answer + latency + cost |
| /metrics | GET | Returns p50/p95 latency, cache hit rate, avg cost |
| /docs | GET | Swagger UI for interactive testing |

---

## What I Would Do Next in Production

1. Replace keyword scorer with RAGAS -- faithfulness, answer relevancy, context precision
2. Add Redis semantic cache with TTL and ingestion-triggered invalidation
3. Content-hash based chunk IDs for idempotent re-ingestion
4. Async retrieval to reduce P95 latency on large document sets
5. Streaming responses via FastAPI StreamingResponse
6. Push latency/cost metrics to Prometheus + Grafana dashboard
