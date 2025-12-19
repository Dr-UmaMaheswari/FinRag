from fastapi import APIRouter
from rag_starterkit.rag.retriever import retrieve_context
from rag_starterkit.rag.vectorstore import peek_documents

debug_router = APIRouter()
@debug_router.get("/v1/debug/peek")
def debug_peek(n: int = 3):
    docs = peek_documents(n=n)
    return {
        "n": n,
        "results": [{"id": d["id"], "snippet": (d["text"] or "")[:400]} for d in docs],
    }
@debug_router.get("/v1/debug/retrieve")
def debug_retrieve(q: str, top_k: int = 3):
    contexts = retrieve_context(q, top_k=top_k)
    return {
        "query": q,
        "top_k": top_k,
        "hits": len(contexts),
        "results": [
            {"id": c["id"], "snippet": (c["text"] or "")[:300]}
            for c in contexts
        ],
    }
