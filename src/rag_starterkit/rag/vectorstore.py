import os
import chromadb
from rag_starterkit.rag.embeddings import embed_texts

# Use an absolute path to avoid "working directory" surprises
BASE_DIR = os.path.abspath(os.getcwd())
CHROMA_DIR = os.path.join(BASE_DIR, ".chroma")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(name="rag_docs")


def add_documents(docs: list[dict]) -> int:
    """
    Upsert documents into Chroma (safe on repeated ingests).
    Returns current collection count.
    """
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    embeddings = embed_texts(texts)

    # upsert is safer than add (prevents duplicate-id errors)
    _collection.upsert(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=[{"source": d["id"]} for d in docs],
    )
    return _collection.count()


def query_documents(query: str, top_k: int = 4) -> list[dict]:
    query_embedding = embed_texts([query])[0]

    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]

    for doc_id, doc in zip(ids, docs):
        contexts.append({"id": doc_id, "text": doc})

    return contexts


def count_documents() -> int:
    return _collection.count()


def peek_documents(n: int = 3) -> list[dict]:
    """
    True peek: fetch stored docs directly (no similarity query).
    """
    res = _collection.get(limit=n, include=["documents", "metadatas"])
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    out = []
    for doc_id, doc in zip(ids, docs):
        out.append({"id": doc_id, "text": doc})
    return out
