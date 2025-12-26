# test_bm25.py
import traceback

from rag_starterkit.rag.milvus_backend import MilvusBackend
from rag_starterkit.rag.bm25_index import bm25_manager


def main():
    print("Starting BM25 test...")

    backend = MilvusBackend()
    col = "rbi_faq_chunks"

    # 1) Count
    n_entities = backend.count_collection(col)
    print(f"Milvus entities in {col} = {n_entities}")

    # 2) Fetch docs (IMPORTANT: keep <= 16384 until you implement a scalable dump method)
    docs = backend.fetch_all_texts(col, limit=10000)  # safe under 16384
    print("Fetched docs:", len(docs))

    if docs:
        d0 = docs[0]
        print("Sample chunk_id:", d0.get("chunk_id"))
        print("Sample bank/source:", d0.get("bank"), "/", d0.get("source_id"))
        print("Sample text head:", (d0.get("text") or "")[:160].replace("\n", " "))

    # 3) Build BM25
    indexed = bm25_manager.build_index(col, docs)
    print("Indexed docs:", indexed)

    # 4) Search
    q = "payee name validation"
    hits = bm25_manager.search(col, q, top_k=6)
    print("BM25 hits:", len(hits))

    for i, h in enumerate(hits, 1):
        print(f"\n#{i} lexical_score={h['lexical_score']:.4f} chunk_id={h['chunk_id']}")
        print((h["text"] or "")[:240].replace("\n", " "))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR:")
        traceback.print_exc()
