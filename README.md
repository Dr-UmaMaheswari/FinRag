# RAG Enterprise Starterkit

Enterprise-grade reference implementation for Retrieval-Augmented Generation (RAG) systems with:
- Controlled retrieval + citations
- Config-driven components
- Minimal API surface (FastAPI)
- Offline evaluation harness
- Docker-first execution

This starterkit is designed to be reused for domain variants (Finance RAG, HR RAG, Logistics RAG) via configuration and dataset swaps.

---

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