from __future__ import annotations

import os
import math
import time
from typing import List, Optional

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from rag_starterkit.rag.embeddings import embed_texts, get_default_embedder


def l2_normalize(vec: List[float], eps: float = 1e-12) -> List[float]:
    s = 0.0
    for v in vec:
        s += float(v) * float(v)
    n = math.sqrt(s)
    if n < eps:
        return [0.0 for _ in vec]
    return [float(v) / n for v in vec]


def ensure_collection(name: str, dim: int) -> Collection:
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
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"Smoke test collection: {name}")
    col = Collection(name=name, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="embedding", index_params=index_params)
    return col


def upsert_one(col: Collection, chunk_id: str, text: str, bank: str, doc_id: str):
    # Embed and normalize
    vec = embed_texts([text])[0]
    vec = l2_normalize([float(x) for x in vec])

    # Delete existing primary key then insert (portable "upsert")
    try:
        col.delete(f'chunk_id in ["{chunk_id}"]')
    except Exception:
        pass

    data = [
        [chunk_id],            # chunk_id
        [doc_id],              # doc_id
        [bank],                # bank
        ["smoke_test"],        # source_id
        [0],                   # page_start
        [0],                   # page_end
        ["0"],                 # order_key
        ["smoke_test"],        # section_type
        ["smoke_test"],        # policy_topic
        [text],                # text
        [vec],                 # embedding
    ]

    col.insert(data)
    col.flush()


def search(col: Collection, query: str, top_k: int = 5):
    qvec = embed_texts([query])[0]
    qvec = l2_normalize([float(x) for x in qvec])

    # Load ensures searchable in many deployments
    col.load()

    res = col.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["chunk_id", "text", "bank", "doc_id", "source_id"],
    )
    return res


def main():
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")

    collection_name = os.getenv("MILVUS_COLLECTION_TEST", "tmb_policy_chunks")
    bank = os.getenv("MILVUS_TEST_BANK", "tmb")
    doc_id = os.getenv("MILVUS_TEST_DOC_ID", "smoke_doc")

    embedder = get_default_embedder()
    dim = int(os.getenv("MILVUS_DIM", str(embedder.dim)))

    print("=== Milvus Smoke Test ===")
    print(f"Host/Port: {host}:{port}")
    print(f"Collection: {collection_name}")
    print(f"Embedding dim: {dim}")
    print()

    connections.connect(alias="default", host=host, port=port)
    print("Connected to Milvus: OK")

    col = ensure_collection(collection_name, dim)
    print("Collection ready:", collection_name)

    # Make sure index exists (for existing collections you created earlier)
    if not col.indexes:
        print("No index found; creating HNSW index...")
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
        )

    col.load()

    # Insert a deterministic record
    chunk_id = f"smoke::{collection_name}::001"
    text = "Cheque Truncation System (CTS) inward clearing policy and return memo rules."
    print("Upserting test record:", chunk_id)
    upsert_one(col, chunk_id=chunk_id, text=text, bank=bank, doc_id=doc_id)

    # Small wait for consistency in some setups
    time.sleep(0.2)

    # Search
    query_text = "What are CTS inward clearing return memo rules?"
    print("Searching:", query_text)
    res = search(col, query_text, top_k=5)

    print("\n=== Top Hits ===")
    for hits in res:
        for h in hits:
            ent = h.entity
            snippet = (ent.get("text") or "")[:120].replace("\n", " ")
            print(
                f"- score={float(h.score):.4f} "
                f"chunk_id={ent.get('chunk_id')} bank={ent.get('bank')} doc_id={ent.get('doc_id')} "
                f"text='{snippet}...'"
            )

    # Count
    print("\nEntities in collection:", int(col.num_entities))
    print("Smoke test: DONE")


if __name__ == "__main__":
    main()
