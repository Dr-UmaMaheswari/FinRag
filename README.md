# FinRAG

**FinRAG** is a production-style Retrieval-Augmented Generation (RAG) system for **multi-bank cheque collection / CTS policy compliance**.

It ingests **RBI CTS FAQs** and **multiple Indian banks’ cheque collection policies** into Milvus, runs **hybrid retrieval** (dense embeddings + BM25 lexical search + cross-encoder reranker), and exposes a **FastAPI** backend that can:

- Route queries to the right bank policy collections (RBI + one or more banks)
- Retrieve and fuse policy snippets across banks
- Generate LLM answers that **compare RBI guidance vs each bank’s policy** for specific compliance questions (e.g., payee name validation, endorsement irregularities, Positive Pay, outstation cheques).

This repo is intentionally structured as a **portfolio-ready project** for engineering- and architect-level roles working on GenAI, RAG, or applied NLP in regulated domains.

---

## Why this project is relevant

From a hiring manager’s perspective, FinRAG demonstrates:

- **End-to-end ownership of a GenAI system**  
  From ingestion and vector DB integration to retrieval logic, prompt design, and API exposure.

- **Hybrid retrieval beyond toy examples**  
  Combines Milvus (dense), BM25 (lexical), RRF fusion, cross-encoder reranking, adjacency expansion, and bank-balanced context compression.

- **Domain modeling in a regulated space**  
  CTS cheque collection policies, RBI FAQs, and bank-wise routing for Axis, TMB, IDBI, Union Bank, HDFC, ICICI, and others.

- **Production-oriented code structure**  
  Clean separation of concerns:
  - `milvus_backend.py` – storage + vector search
  - `retriever.py` / `router_retriever.py` – hybrid retrieval + router logic
  - `prompt.py` – bank-aware, comparison-oriented prompts
  - `routes.py` / `debug_routes.py` – FastAPI endpoints (normal + debug)

- **Explainability and observability**  
  `/v1/debug/retrieve_router` returns per-collection hybrid stats (dense_count, bm25_count, fused_count, rerank_in, final_out, top IDs), helping debug retrieval and explain system behaviour to non-ML stakeholders.

If you are a recruiter or hiring manager, you can think of FinRAG as a **mini production RAG system** for a realistic banking compliance use case.

---

## Core Use Case

**Question type:**  

> “How does Bank X handle payee name validation / endorsement irregularities under CTS, and how does that compare to RBI CTS guidance?”

**FinRAG workflow:**

1. Detect requested bank(s) (e.g., `axis`, `tmb`) and always include `rbi` as the regulator reference.
2. Retrieve relevant passages from:
   - `rbi_faq_chunks` (RBI CTS FAQ)
   - `<bank>_policy_chunks` (bank-specific cheque collection policy)
3. Fuse dense + BM25 results, rerank with a cross-encoder, and expand adjacent chunks for narrative continuity.
4. Apply bank-balanced context compression so RBI doesn’t dominate context when comparing multiple banks.
5. Build a **bank-aware prompt** that explicitly asks the LLM to:
   - Summarise RBI view
   - Summarise each bank’s policy
   - Highlight similarities and differences
6. Return a structured answer with citations.

---

## Key Features

- **Hybrid Retrieval**
  - Dense embeddings via SentenceTransformers (Milvus as vector store)
  - BM25 lexical index (local, per-collection)
  - Reciprocal Rank Fusion (RRF) to combine dense + lexical hits
  - Cross-encoder reranker to re-score top candidates

- **Bank Router**
  - `bank_router.py` infers `(bank, collection)` from PDF filenames during ingestion  
    (e.g., `tmb.pdf` → `tmb`, `tmb_policy_chunks`)
  - `_normalize_bank_to_collection()` maps user inputs (`axis`, `tmb`, `idbi`, `union`) to internal Milvus collections.

- **Bank-Balanced Context Compression**
  - Allocates context budget between **RBI** and **other banks**
  - Removes near-duplicate chunks via Jaccard similarity
  - Preserves narrative continuity via `(source_id, order_key)` sorting
  - For small candidate sets, compression becomes a no-op to avoid losing key chunks

- **Bank-Aware Prompting**
  - `build_rag_prompt()` groups contexts by bank and generates two modes:
    - **Single-bank mode:** “Explain Bank X policy on …”
    - **Multi-bank comparison mode:** “Compare RBI vs Bank A vs Bank B on …”
  - Uses `_pretty_bank_name()` to turn internal keys (`axis_bank`, `tmb`, `idbi`, etc.) into human-readable labels.

- **FastAPI Backend + Debug Endpoints**
  - `/v1/query` – main RAG endpoint (LLM answer + citations)
  - `/v1/debug/retrieve_router` – debug endpoint exposing hybrid retrieval traces per collection and globally.

---

## Tech Stack

- **Language / Runtime**
  - Python 3.10+

- **Backend**
  - FastAPI
  - Uvicorn

- **Retrieval & Storage**
  - Milvus (vector DB)
  - BM25 lexical search
  - SentenceTransformers encoder
  - Cross-Encoder reranker

- **LLM**
  - Pluggable LLM backend (local or hosted) – used for final answer generation based on the bank-aware prompt.

---

## Example API Calls

### 1. Debug hybrid retrieval (Axis + TMB)

```bash
curl "http://127.0.0.1:8000/v1/debug/retrieve_router?q=payee%20endorsement&banks=axis,tmb&top_k=8"


## Architecture (High Level)
**Ingest → Chunk → Embed → Index → Retrieve → Rerank (optional) → Generate (grounded) → Cite**

Key guarantees:
- Responses must be grounded in retrieved context
- Citations are returned for auditability
- Evaluation harness enables regression testing

---

## Quickstart

### 1) Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .
Options:
pip install -e .[dev]
pip install -e .[milvus]
pip install -e .[hybrid]
pip install -e .[dev,milvus,hybrid]
uvicorn rag_starterkit.main:app --reload --port 8000
2) Run with Docker
For CPU Only Machine:
docker compose up --build

FOR GPU Enabled Machine:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build


API

GET /health → health check

POST /v1/ingest → ingest sample docs (local folder)

POST /v1/query → query RAG and receive answer + citations

Example:

curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the sample policy say about refunds?"}'

Evaluation

Run offline evaluation:

python -m rag_starterkit.eval.run_eval --queries samples/queries.jsonl

What to customize

Chunking: src/rag_starterkit/rag/chunking.py

Vector store: src/rag_starterkit/rag/vectorstore.py

Generation rules/guardrails: src/rag_starterkit/rag/generator.py

Config: src/rag_starterkit/core/config.py

Roadmap (enterprise upgrades)

Add reranker (CrossEncoder)
Add hybrid search (BM25 + vectors)
Add auth (API key/JWT)
Add observability (OpenTelemetry)
Add evaluation metrics (faithfulness/groundedness)
