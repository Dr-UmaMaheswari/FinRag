from typing import List, Optional

from fastapi import APIRouter, Query
from rag_starterkit.api.schemas import IngestRequest, QueryRequest, QueryResponse
from rag_starterkit.data.ingest import ingest_path, chunk_only_pdf
from rag_starterkit.rag.generator import generate_answer

# NEW: router retriever
from rag_starterkit.rag.router_retriever import RouterRetriever


router = APIRouter()


def _normalize_bank_to_collection(bank: str) -> str:
    b = (bank or "").strip().lower()
    b = b.replace(" ", "_").replace("-", "_")

    # RBI
    if b in {"rbi"}:
        return "rbi_faq_chunks"

    # Axis
    if b in {"axis", "axis_bank"}:
        return "axis_bank_policy_chunks"

    # BOI
    if b in {"boi", "bank_of_india", "boi_bank"}:
        return "boi_bank_policy_chunks"

    # Bank of Baroda (your ingest uses "barado")
    if b in {"bob", "bank_of_baroda", "baroda", "bank_of_barado_bank", "barado"}:
        return "bank_of_barado_bank_policy_chunks"

    # Canara
    if b in {"canara", "canara_bank"}:
        return "canara_bank_policy_chunks"

    # CSB
    if b in {"csb", "csb_bank"}:
        return "csb_bank_policy_chunks"

    # CUB
    if b in {"cub", "cub_bank"}:
        return "cub_bank_policy_chunks"

    # Central Bank
    if b in {"central", "central_bank"}:
        return "central_bank_policy_chunks"

    # CSCM
    if b in {"cscm", "cscm_bank"}:
        return "cscm_bank_policy_chunks"

    # DCB
    if b in {"dcb", "dcb_bank"}:
        return "dcb_bank_policy_chunks"

    # HDFC
    if b in {"hdfc", "hdfc_bank"}:
        return "hdfc_bank_policy_chunks"

    # HPSCB
    if b in {"hpscb", "hpscb_bank"}:
        return "hpscb_bank_policy_chunks"

    # ICICI
    if b in {"icici", "icici_bank"}:
        return "icici_bank_policy_chunks"

    # IDBI
    if b in {"idbi"}:
        return "idbi_policy_chunks"

    # IDFC
    if b in {"idfc"}:
        return "idfc_policy_chunks"

    # Indian Bank
    if b in {"indian", "indian_bank"}:
        return "indian_bank_policy_chunks"

    # IndusInd (your ingest: indusind_bank_policy_chunks)
    if b in {"indusind", "indusind_bank"}:
        return "indusind_bank_policy_chunks"

    # JK Bank
    if b in {"jk", "jk_bank", "jammu_kashmir", "jammu_and_kashmir"}:
        return "jk_bank_policy_chunks"

    # Kotak (your ingest uses "kodak")
    if b in {"kotak", "kotak_bank", "kodak", "kodak_bank"}:
        return "kodak_bank_policy_chunks"

    # KVB
    if b in {"kvb", "kvb_bank"}:
        return "kvb_bank_policy_chunks"

    # Maharashtra (your ingest uses "maharastra")
    if b in {"maharashtra", "maharastra", "maharastra_bank", "maharashtra_bank"}:
        return "maharastra_bank_policy_chunks"

    # MCB
    if b in {"mcb", "mcb_bank"}:
        return "mcb_bank_policy_chunks"

    # Punjab (likely PNB, but your collection name is punjab_bank_policy_chunks)
    if b in {"punjab", "punjab_bank", "pnb", "punjab_national_bank"}:
        return "punjab_bank_policy_chunks"

    # TMB
    if b in {"tmb", "tamilnad_mercantile_bank", "tamil_nadu_mercantile_bank"}:
        return "tmb_policy_chunks"

    # Union
    if b in {"union", "union_bank"}:
        return "union_policy_chunks"

    # UCO
    if b in {"uco", "uco_bank"}:
        return "uco_policy_chunks"

    # YES
    if b in {"yes", "yes_bank"}:
        return "yes_bank_policy_chunks"

    # NPCI
    if b in {"npci", "npci_bank"}:
        return "npci_bank_policy_chunks"

    # Generic fallback
    return "generic_policy_chunks"
def banks_to_collections(banks: Optional[str]) -> List[str]:
    """
    banks: comma-separated list of bank tokens or explicit collection names.
      Examples:
        "axis,tmb"
        "axis_bank,tmb_policy_chunks"
        "icici"
    Returns:
      - list of collection names (excluding RBI; RBI handled via include_rbi=True)
      - falls back to ["generic_policy_chunks"] if nothing matches
    """
    if not banks:
        return []

    cols: List[str] = []
    for token in banks.split(","):
        token = (token or "").strip()
        if not token:
            continue

        # allow either "tmb" OR "tmb_policy_chunks"
        if token.endswith("_chunks"):
            col = token
        else:
            col = _normalize_bank_to_collection(token)

        # RBI is handled separately by include_rbi flag
        if col == "rbi_faq_chunks":
            continue

        if col and col not in cols:
            cols.append(col)

    return cols or ["generic_policy_chunks"]


def get_embedder():
    """
    Wire this to your current embedding implementation.

    Expected interface:
      embedder.embed(text: str) -> List[float]

    If you already have a module like rag_starterkit.rag.embedder,
    replace this function with:
      from rag_starterkit.rag.embedder import get_embedder
    """
    from rag_starterkit.rag.embeddings import get_default_embedder  # adjust if needed
    return get_default_embedder()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/v1/debug/chunk_pdf")
def debug_chunk_pdf(path: str = Query(...), max_chars: int = 2200):
    out_json = "data/chunk_debug.json"
    return chunk_only_pdf(path, max_chars=max_chars, out_json=out_json)


@router.post("/v1/ingest")
def ingest(req: IngestRequest):
    result = ingest_path(req.path)
    return {"ingested": result}


@router.get("/v1/debug/retrieve_router")
def debug_retrieve_router(
    q: str = Query(..., description="Query string"),
    banks: Optional[str] = Query(
        None,
        description="Comma separated list of banks/collections e.g. tmb,union or tmb_policy_chunks",
    ),
    top_k: int = 6,
):
    embedder = get_embedder()
    rr = RouterRetriever(embedder=embedder)

    bank_cols = banks_to_collections(banks)

    # default to TMB if none provided
    if not bank_cols:
        bank_cols = ["tmb_policy_chunks"]

    hits, trace = rr.retrieve(
        query=q,
        bank_collections=bank_cols,
        top_k_dense_each=8,
        top_k_final=top_k,
        include_rbi=True,
        return_trace=True,
    )

    return {
        "query": q,
        "collections": ["rbi_faq_chunks"] + bank_cols,
        "top_k": top_k,
        "debug": trace,
        "results": [
            {
                "chunk_id": h.chunk_id,
                "bank": h.bank,
                "source_id": h.source_id,
                "page_start": h.page_start,
                "page_end": h.page_end,
                "order_key": h.order_key,
                "section_type": h.section_type,
                "policy_topic": h.policy_topic,
                "score": h.score,
                "preview": (h.text or "")[:350],
            }
            for h in hits
        ],
    }

@router.post("/ingest_file")
def ingest_file( req:IngestRequest):
    return ingest_path(req.path)
   

@router.post("/v1/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    bank: Optional[str] = Query(
        None,
        description="Optional bank hint (e.g., tmb, union, idbi). If omitted, defaults to tmb.",
    ),
    banks: Optional[str] = Query(
        None,
        description="Optional comma-separated banks/collections (e.g., axis,tmb or tmb_policy_chunks). Overrides `bank` if provided.",
    ),
     use_reranker: bool = Query(
        True,
        description="Enable cross-encoder reranking (disable for fast mode).",
    ),
     use_judge: bool = Query(
         True, 
         description="Enable LLM-as-judge evaluation (disable for fast mode)."
    ),
):
    """
    Router RAG:
      Milvus (RBI + bank collections) -> rerank -> compress -> generate
    """

    embedder = get_embedder()
    rr = RouterRetriever(embedder=embedder)

    # Support: if you later add `banks: List[str]` into QueryRequest schema,
    # you can check it here; for now use query param `bank`.
    if banks:
        bank_cols = banks_to_collections(banks)
    elif bank:
        bank_cols = [_normalize_bank_to_collection(bank)]
    else:
        bank_cols = ["tmb_policy_chunks"]

    retrieved = rr.retrieve(
        query=req.query,
        bank_collections=bank_cols,
        top_k_dense_each=8,
        top_k_final=req.top_k,
        include_rbi=True,
        use_reranker=use_reranker,
    )

    # Convert to your existing "contexts" format expected by prompt+citations
    contexts = []
    for h in retrieved:
        contexts.append(
            {
                "id": h.source_id or h.chunk_id,
                "text": h.text,
                "bank": h.bank,
                "source_id": h.source_id,
                "page_start": h.page_start,
                "page_end": h.page_end,
                "order_key": h.order_key,
                "section_type": h.section_type,
                "policy_topic": h.policy_topic,
                "score": h.score,
            }
        )

    answer, citations, quality = generate_answer(req.query, contexts,use_judge=use_judge)

    return QueryResponse(
        answer=answer,
        citations=citations,
        quality=quality,
    )
