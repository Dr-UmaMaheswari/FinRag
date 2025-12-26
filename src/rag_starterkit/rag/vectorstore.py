from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import chromadb
from rag_starterkit.rag.embeddings import embed_texts

# Use an absolute path to avoid "working directory" surprises
BASE_DIR = os.path.abspath(os.getcwd())
CHROMA_DIR = os.path.join(BASE_DIR, ".chroma")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(name="rag_docs")


# -------------------------
# Embedding normalizers
# -------------------------
def _as_list_of_lists(x: Any) -> List[List[float]]:
    """
    Normalize embeddings to List[List[float]].

    Supports:
      - numpy arrays / tensors with .tolist()
      - list[list[float]]
      - list[np.ndarray] / list[tensor]
      - single vector list[float] (wrapped)
      - single vector np.ndarray (wrapped)
    """
    if x is None:
        return []

    if hasattr(x, "tolist") and callable(getattr(x, "tolist")):
        x_list = x.tolist()
        # vector -> wrap
        if x_list and isinstance(x_list[0], (int, float)):
            return [x_list]
        return x_list

    if isinstance(x, list):
        if not x:
            return []
        # list[float] -> wrap
        if isinstance(x[0], (int, float)):
            return [x]
        # list[np.ndarray]/tensor -> per-row tolist
        if hasattr(x[0], "tolist") and callable(getattr(x[0], "tolist")):
            return [v.tolist() for v in x]
        # already list[list[float]]
        return x

    raise TypeError(f"Unsupported embedding type: {type(x)}")


def _as_single_vector_list(x: Any) -> List[float]:
    """
    Normalize a single embedding vector to List[float].
    """
    if x is None:
        return []

    if hasattr(x, "tolist") and callable(getattr(x, "tolist")):
        x_list = x.tolist()
        # accidentally matrix -> take first row
        if x_list and isinstance(x_list[0], list):
            return x_list[0]
        return x_list

    if isinstance(x, list):
        # accidentally matrix -> take first row
        if x and isinstance(x[0], list):
            return x[0]
        return x

    raise TypeError(f"Unsupported embedding type for vector: {type(x)}")


# -------------------------
# Document validation
# -------------------------
def _normalize_docs(docs: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Filters and normalizes docs for vector DB upsert.

    Keeps only docs with:
      - id: non-empty str
      - text: non-empty str after strip()

    Returns (texts, ids, metadatas)
    """
    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict] = []

    for d in docs or []:
        doc_id = d.get("id")
        text = d.get("text")

        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue

        texts.append(text)
        ids.append(doc_id)
        metadatas.append({"source": d.get("source", doc_id)})

    return texts, ids, metadatas


# -------------------------
# Chroma-backed operations
# -------------------------
def add_documents(docs: List[Dict]) -> int:
    """
    Upsert documents into Chroma (safe on repeated ingests).
    Returns current collection count.

    Important:
    - If docs are empty or all invalid (empty text/missing id), we do a NO-OP and
      return current count. This prevents Chroma's "empty embeddings" error
      during folder ingestion when some files yield no chunks.
    """
    if not docs:
        return _collection.count()

    texts, ids, metadatas = _normalize_docs(docs)

    # Nothing valid to embed/store -> NO-OP (safe for ingestion loops)
    if not texts:
        return _collection.count()

    embeddings_raw = embed_texts(texts)
    embeddings = _as_list_of_lists(embeddings_raw)

    # Fail fast if embedder misbehaves
    if not embeddings:
        raise ValueError(
            "embed_texts() returned empty embeddings for non-empty texts. "
            f"texts={len(texts)} first_text_len={len(texts[0]) if texts else None}"
        )

    if len(embeddings) != len(texts):
        raise ValueError(
            "Embedding count mismatch: "
            f"got {len(embeddings)} vectors for {len(texts)} texts"
        )

    _collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return _collection.count()


def query_documents(query: str, top_k: int = 4) -> List[Dict]:
    if not isinstance(query, str) or not query.strip():
        return []

    embeddings_raw = embed_texts([query])
    # embed_texts returns List[List[float]]; take first vector
    if isinstance(embeddings_raw, list) and embeddings_raw:
        query_vec = _as_single_vector_list(embeddings_raw[0])
    else:
        query_vec = _as_single_vector_list(embeddings_raw)

    if not query_vec:
        raise ValueError("Query embedding is empty. Check embed_texts() output for query input.")

    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts: List[Dict] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0] if "distances" in results else []
    metas = results.get("metadatas", [[]])[0] if "metadatas" in results else []

    for i, (doc_id, doc) in enumerate(zip(ids, docs)):
        item: Dict[str, Any] = {"id": doc_id, "text": doc}
        if i < len(dists):
            item["distance"] = dists[i]
        if i < len(metas):
            item["metadata"] = metas[i]
        contexts.append(item)

    return contexts


def count_documents() -> int:
    return _collection.count()


def peek_documents(n: int = 3) -> List[Dict]:
    """
    True peek: fetch stored docs directly (no similarity query).
    """
    if n <= 0:
        return []
    res = _collection.get(limit=n, include=["documents", "metadatas"])
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    out: List[Dict] = []
    for doc_id, doc, meta in zip(ids, docs, metas):
        out.append({"id": doc_id, "text": doc, "metadata": meta})
    return out
