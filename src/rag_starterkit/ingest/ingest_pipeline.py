import os
from typing import Dict, List, Tuple

from .pdf_loader import load_pdf_pages
from .heading_detector import detect_headings
from .toc_parser import parse_toc_from_text
from .hierarchy_builder import build_tree
from .leaf_chunker import leaf_chunks_from_tree
from .concept_tagger import tag_concepts
from .relations_builder import build_relations
from .bank_router import infer_bank_and_collection
from rag_starterkit.rag.backend_factory import get_vector_backend

RELATIONS_PATH = os.path.join("data", "relations.json")


def _infer_doc_id(file_path: str) -> str:
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0].lower().replace(" ", "_")
    return name[:64]



def ingest_folder(folder_path: str) -> Dict:
    """
    Ingest all PDFs in folder_path.

    Supports:
    - Milvus Plan B (per-bank collections) if backend exposes upsert_chunks(collection_name, rows)
    - Backward-compatible: if backend only exposes upsert(docs_for_store), uses that.

    Returns summary counts + relations.json path.
    """
    backend = get_vector_backend()

    all_chunks_payload = []
    stored_total = 0
    parsed_total = 0
    per_collection_counts: Dict[str, int] = {}

    for root, _, files in os.walk(folder_path):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue

            path = os.path.join(root, fn)
            doc_id = _infer_doc_id(path)
            bank, collection_name = infer_bank_and_collection(path)

            pages = load_pdf_pages(path)
            doc_last_page = pages[-1].page_num if pages else 1

            # TOC-first: scan first few pages for TOC text
            toc_text = ""
            for p in pages[:3]:
                if "table of contents" in (p.text or "").lower():
                    toc_text += "\n" + (p.text or "")

            toc_items = parse_toc_from_text(toc_text) if toc_text.strip() else []
            if len(toc_items) < 3:
                toc_items = detect_headings(pages)

            tree = build_tree(toc_items)

            chunks = leaf_chunks_from_tree(
                pages=pages,
                doc_id=doc_id,
                source_path=path,
                root=tree,
                doc_last_page=doc_last_page,
                max_chars=2200,
            )

            docs_for_store = []
            rows_for_milvus = []

            source_id = os.path.basename(path)

            for ch in chunks:
                tags = tag_concepts(ch.text)

                # Always attach metadata at chunk level
                ch.metadata.update({
                    "bank": bank,
                    "collection": collection_name,
                    "concept_tags": tags,
                    "doc_id": doc_id,
                    "source_id": source_id,
                })

                # Generic doc payload (Chroma-like backends)
                docs_for_store.append({
                    "id": ch.chunk_id,
                    "text": ch.text,
                    **ch.metadata,
                })

                # Relations payload (bank/corpus linking)
                all_chunks_payload.append({
                    "chunk_id": ch.chunk_id,
                    "doc_id": doc_id,
                    "bank": bank,
                    "collection": collection_name,
                    "source_id": source_id,
                    "page_start": ch.metadata.get("page_start"),
                    "page_end": ch.metadata.get("page_end"),
                    "order_key": ch.metadata.get("order_key"),
                    "section_type": ch.metadata.get("section_type"),
                    "policy_topic": ch.metadata.get("policy_topic"),
                    "text": ch.text,
                    "concept_tags": tags,
                })

                # Milvus row payload (only used if backend supports it)
                # NOTE: Milvus backend typically expects embedding too.
                # If your MilvusBackend computes embeddings internally, keep this as-is.
                # If it does NOT, you must add an embedder call here and include "embedding".
                rows_for_milvus.append({
                    "chunk_id": ch.chunk_id,
                    "doc_id": doc_id,
                    "bank": bank,
                    "source_id": source_id,
                    "page_start": int(ch.metadata.get("page_start", 0) or 0),
                    "page_end": int(ch.metadata.get("page_end", 0) or 0),
                    "order_key": str(ch.metadata.get("order_key", "")),
                    "section_type": str(ch.metadata.get("section_type", "")),
                    "policy_topic": str(ch.metadata.get("policy_topic", "") or ""),
                    "text": ch.text,
                    # "embedding": <ADD HERE if your Milvus backend does not embed internally>
                })

            parsed_total += len(docs_for_store)

            # -----------------------------
            # Store to vector backend
            # -----------------------------
            stored_now = 0

            # Preferred: Milvus Plan B API
            if hasattr(backend, "upsert_chunks"):
                # If your MilvusBackend embeds internally, it can accept rows without "embedding".
                stored_now = backend.upsert_chunks(collection_name, rows_for_milvus)
            else:
                # Backward compatible: single default collection (Chroma-style)
                stored_now = backend.upsert(docs_for_store)

            stored_total += int(stored_now)
            per_collection_counts[collection_name] = per_collection_counts.get(collection_name, 0) + int(stored_now)

    # Build relations after all docs in folder processed
    os.makedirs("data", exist_ok=True)
    build_relations(
        all_chunks_payload,
        RELATIONS_PATH,
        build_similarity=True,
        sim_threshold=0.78
    )

    return {
        "chunks_parsed": parsed_total,
        "stored_total": stored_total,
        "per_collection_stored": per_collection_counts,
        "relations_path": RELATIONS_PATH,
    }
