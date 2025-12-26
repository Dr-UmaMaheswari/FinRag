from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from rag_starterkit.rag.vector_backend import VectorBackend
from rag_starterkit.rag.embeddings import embed_texts


def _l2_normalize(vec: List[float], eps: float = 1e-12) -> List[float]:
    s = 0.0
    for v in vec:
        s += float(v) * float(v)
    n = math.sqrt(s)
    if n < eps:
        return [0.0 for _ in vec]
    return [float(v) / n for v in vec]


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


class MilvusBackend(VectorBackend):
    """
    Milvus backend that supports per-bank collections (Plan B).

    Primary API used by ingest.py:
      - upsert_chunks(collection_name, rows)

    Also supports compatibility API:
      - upsert(docs) / query(query, top_k) against default collection.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        dim: Optional[int] = None,
        default_collection: Optional[str] = None,
    ):
        self.host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.dim = int(dim or os.getenv("MILVUS_DIM", "384"))
        self.default_collection = default_collection or os.getenv("MILVUS_COLLECTION", "rag_docs")

        connections.connect(alias="default", host=self.host, port=self.port)

    # -----------------------------
    # Collection management
    # -----------------------------
    def ensure_collection(self, name: str) -> Collection:
        if utility.has_collection(name):
            col = Collection(name)
            return col

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="bank", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="order_key", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="section_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="policy_topic", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description=f"RAG chunks: {name}")
        col = Collection(name=name, schema=schema)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200},
        }
        col.create_index(field_name="embedding", index_params=index_params)
        col.load()
        return col

    # -----------------------------
    # Plan-B API (per bank)
    # -----------------------------
    def upsert_chunks(self, collection_name: str, rows: List[Dict]) -> int:
        """
        Upsert rows into a specific Milvus collection.

        Each row should include:
          chunk_id, doc_id, bank, source_id, page_start, page_end,
          order_key, section_type, policy_topic, text
        Embedding is computed internally (normalized).
        Returns number of rows processed (for ingest reporting).
        """
        if not rows:
            return 0

        col = self.ensure_collection(collection_name)

        # Filter invalid rows (must have chunk_id and text)
        filtered: List[Dict] = []
        for r in rows:
            if not _nonempty_str(r.get("chunk_id")):
                continue
            if not _nonempty_str(r.get("text")):
                continue
            filtered.append(r)

        if not filtered:
            return 0

        texts = [r["text"] for r in filtered]
        vecs = embed_texts(texts)  # List[List[float]]
        if not vecs or len(vecs) != len(filtered):
            raise ValueError(
                f"Embedding mismatch: vectors={len(vecs) if vecs else 0} rows={len(filtered)}"
            )

        vecs = [_l2_normalize([float(x) for x in v]) for v in vecs]

        # Prepare field-wise insert arrays in the schema order
        chunk_ids = [str(r.get("chunk_id")) for r in filtered]
        doc_ids = [str(r.get("doc_id", "")) for r in filtered]
        banks = [str(r.get("bank", "")) for r in filtered]
        source_ids = [str(r.get("source_id", "")) for r in filtered]
        page_starts = [int(r.get("page_start", 0) or 0) for r in filtered]
        page_ends = [int(r.get("page_end", 0) or 0) for r in filtered]
        order_keys = [str(r.get("order_key", "")) for r in filtered]
        section_types = [str(r.get("section_type", "")) for r in filtered]
        policy_topics = [str(r.get("policy_topic", "")) for r in filtered]
        texts_out = [str(r.get("text", "")) for r in filtered]

        # "Upsert" across Milvus versions: delete then insert
        try:
            # Build: chunk_id in ["id1","id2",...]
            safe_ids = [str(x).replace('"', '\\"') for x in chunk_ids]
            expr = 'chunk_id in ["' + '","'.join(safe_ids) + '"]'
            col.delete(expr)
        except Exception:
            pass

        col.insert([
            chunk_ids,
            doc_ids,
            banks,
            source_ids,
            page_starts,
            page_ends,
            order_keys,
            section_types,
            policy_topics,
            texts_out,
            vecs,
        ])
        col.flush()
        return len(filtered)

    def search(
        self,
        collection_name: str,
        query_vec: List[float],
        top_k: int = 4,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search a specific collection with a precomputed query vector.
        """
        col = self.ensure_collection(collection_name)
        qvec = _l2_normalize([float(x) for x in query_vec])

        res = col.search(
            data=[qvec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=output_fields or [
                                                "chunk_id", "text", "source_id", "doc_id", "bank",
                                                 "page_start", "page_end",
                                                 "order_key", "section_type", "policy_topic",
                                            ]

        )

        out: List[Dict] = []
        for hits in res:
            for h in hits:
                ent = h.entity
                out.append(
                    {
                        "chunk_id": ent.get("chunk_id"),
                        "text": ent.get("text"),
                        "score": float(h.score),
                        "source_id": ent.get("source_id"),
                        "doc_id": ent.get("doc_id"),
                        "bank": ent.get("bank"),
                        "page_start": ent.get("page_start"),
                        "page_end": ent.get("page_end"),
                        "order_key": ent.get("order_key"),
                        "section_type": ent.get("section_type"),
                        "policy_topic": ent.get("policy_topic"),
                    }
                )
        return out

    # -----------------------------
    # Compatibility API (legacy)
    # -----------------------------
    def upsert(self, docs: List[Dict]) -> int:
        """
        Compatibility: accept docs like {"id":..., "text":..., ...}
        and store them into default collection.
        """
        if not docs:
            return 0
        rows = []
        for d in docs:
            if not _nonempty_str(d.get("id")) or not _nonempty_str(d.get("text")):
                continue
            rows.append(
                {
                    "chunk_id": d["id"],
                    "doc_id": str(d.get("doc_id", "")),
                    "bank": str(d.get("bank", "")),
                    "source_id": str(d.get("source_id", d.get("source", ""))),
                    "page_start": int(d.get("page_start", 0) or 0),
                    "page_end": int(d.get("page_end", 0) or 0),
                    "order_key": str(d.get("order_key", "")),
                    "section_type": str(d.get("section_type", "")),
                    "policy_topic": str(d.get("policy_topic", "")),
                    "text": d["text"],
                }
            )
        return self.upsert_chunks(self.default_collection, rows)

    def query(self, query: str, top_k: int) -> List[Dict]:
        """
        Compatibility: query against default collection.
        """
        q = embed_texts([query])
        if not q or not q[0]:
            return []
        hits = self.search(self.default_collection, q[0], top_k=top_k)
        return [
            {
                "id": h.get("chunk_id"),
                "text": h.get("text"),
                "score": h.get("score"),
                "metadata": {
                    "source_id": h.get("source_id"),
                    "doc_id": h.get("doc_id"),
                    "bank": h.get("bank"),
                    "page_start": h.get("page_start"),
                    "page_end": h.get("page_end"),
                },
            }
            for h in hits
        ]

    def count(self) -> int:
        """
        Compatibility: count default collection.
        """
        col = self.ensure_collection(self.default_collection)
        try:
            col.load()
        except Exception:
            pass
        return int(col.num_entities)

    def count_collection(self, collection_name: str) -> int:
        col = self.ensure_collection(collection_name)
        try:
            col.load()
        except Exception:
            pass
        return int(col.num_entities)
    # -----------------------------
    # Fetch All texts API For BM25
    # -----------------------------

    def fetch_all_texts(
        self,
        collection_name: str,
        limit: int = 200000,
        batch_size: int = 8000,   # keep comfortably under 16384
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch documents for building lexical index (BM25) using paging.
        Milvus enforces (offset+limit) <= 16384 per query call, so we page.
        """
        col = self.ensure_collection(collection_name)
        try:
            col.load()
        except Exception:
            pass

        fields = output_fields or [
            "chunk_id", "text", "source_id", "doc_id", "bank",
            "page_start", "page_end", "order_key", "section_type", "policy_topic",
        ]

        expr = 'chunk_id != ""'  # tautology for VARCHAR pk in most setups

        out: List[Dict] = []
        offset = 0

        # Ensure we never exceed server-side window; keep offset < 16384 always.
        # We page by repeatedly querying with offset reset to 0 is NOT supported,
        # so we must use offset but keep (offset+batch_size) <= 16384.
        #
        # Therefore: we cannot fetch >16384 using offset paging alone unless Milvus
        # version supports "limit without offset restriction" (yours does not).
        #
        # Workaround: fetch in deterministic slices via ORDER BY primary key is not
        # available. Instead, we must use chunk_id prefix partitioning (recommended below).
        #
        # For now, we fetch up to the max window.
        max_window = 16384
        effective_limit = min(limit, max_window)

        while offset < effective_limit:
            bs = min(batch_size, effective_limit - offset)
            res = col.query(
                expr=expr,
                output_fields=fields,
                limit=bs,
                offset=offset,
            )
            if not res:
                break

            for ent in res:
                out.append({
                    "chunk_id": ent.get("chunk_id"),
                    "text": ent.get("text", ""),
                    "source_id": ent.get("source_id", ""),
                    "doc_id": ent.get("doc_id", ""),
                    "bank": ent.get("bank", ""),
                    "page_start": ent.get("page_start", 0),
                    "page_end": ent.get("page_end", 0),
                    "order_key": ent.get("order_key", ""),
                    "section_type": ent.get("section_type", ""),
                    "policy_topic": ent.get("policy_topic", ""),
                })

            offset += bs

        return out
