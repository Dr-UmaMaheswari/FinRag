from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import json
import re
import hashlib

from rag_starterkit.ingest.pdf_loader import load_pdf_pages
from rag_starterkit.ingest.heading_detector import detect_headings
from rag_starterkit.ingest.toc_parser import parse_toc_from_text
from rag_starterkit.ingest.hierarchy_builder import build_tree
from rag_starterkit.ingest.leaf_chunker import leaf_chunks_from_tree
from rag_starterkit.ingest.concept_tagger import tag_concepts
from rag_starterkit.ingest.relations_builder import build_relations
from rag_starterkit.ingest.bank_router import infer_bank_and_collection
from rag_starterkit.ingest.faq_chunker import chunk_faq_text,_is_faq_doc
from rag_starterkit.rag.backend_factory import get_vector_backend
from rag_starterkit.rag.embeddings import get_default_embedder

RELATIONS_PATH = os.path.join("data", "relations.json")


# ---------------------------
# Helpers
# ---------------------------

def _infer_doc_id(file_path: str) -> str:
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0].lower().replace(" ", "_")
    return name[:64]



def _extract_toc_text(pages, max_scan_pages: int = 3) -> str:
    toc_text = ""
    for p in pages[:max_scan_pages]:
        if "table of contents" in (p.text or "").lower():
            toc_text += "\n" + (p.text or "")
    return toc_text


def _pdf_full_text(pages) -> str:
    # Join page texts. FAQ chunker typically works better on the full linear text.
    return "\n".join([(p.text or "") for p in (pages or [])])



# ---------------------------
# Chunk-only debug
# ---------------------------

def chunk_only_pdf(
    pdf_path: str,
    doc_id: Optional[str] = None,
    max_chars: int = 2200,
    out_json: Optional[str] = None
) -> Dict:
    """
    Chunking-only test: parses ONE PDF, builds hierarchy (TOC-first + fallback),
    and returns chunks + summary. No vector DB writes.

    If out_json is provided, saves chunks to that JSON file.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if doc_id is None:
        doc_id = _infer_doc_id(pdf_path)

    pages = load_pdf_pages(pdf_path)
    doc_last_page = pages[-1].page_num if pages else 1

    full_text = _pdf_full_text(pages)

    # FAQ override
    if _is_faq_doc(pdf_path, full_text):
        faq_chunks = chunk_faq_text(full_text, doc_id=doc_id, source_path=pdf_path)
        chunk_dicts: List[Dict] = []
        for ch in faq_chunks:
            tags = tag_concepts(ch["text"])
            chunk_dicts.append({
                "chunk_id": ch["chunk_id"],
                "doc_id": doc_id,
                "page_start": ch["metadata"].get("page_start", 1),
                "page_end": ch["metadata"].get("page_end", 1),
                "order_key": ch["metadata"].get("order_key", ""),
                "title_path": ["FAQ"],
                "number_path": [ch["metadata"].get("order_key", "")],
                "concept_tags": tags,
                "text_preview": ch["text"][:500]
            })

        result = {
            "pdf": pdf_path,
            "doc_id": doc_id,
            "strategy_used": "faq",
            "pages": doc_last_page,
            "chunks": len(chunk_dicts),
            "chunk_samples": chunk_dicts[:5],
            "all_chunks": chunk_dicts
        }

        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # Non-FAQ: existing hierarchy flow
    toc_text = _extract_toc_text(pages, max_scan_pages=3)
    toc_items = parse_toc_from_text(toc_text) if toc_text.strip() else []
    used = "toc" if len(toc_items) >= 3 else "headings_fallback"

    if used == "headings_fallback":
        toc_items = detect_headings(pages)

    tree = build_tree(toc_items)

    chunks = leaf_chunks_from_tree(
        pages=pages,
        doc_id=doc_id,
        source_path=pdf_path,
        root=tree,
        doc_last_page=doc_last_page,
        max_chars=max_chars
    )

    chunk_dicts: List[Dict] = []
    for ch in chunks:
        tags = tag_concepts(ch.text)
        chunk_dicts.append({
            "chunk_id": ch.chunk_id,
            "doc_id": ch.doc_id,
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "order_key": ch.order_key,
            "title_path": ch.title_path,
            "number_path": ch.number_path,
            "concept_tags": tags,
            "text_preview": ch.text[:500]
        })

    result = {
        "pdf": pdf_path,
        "doc_id": doc_id,
        "strategy_used": used,
        "pages": doc_last_page,
        "chunks": len(chunk_dicts),
        "chunk_samples": chunk_dicts[:5],
        "all_chunks": chunk_dicts
    }

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------
# Ingest folder (Plan B)
# ---------------------------

def ingest_folder(folder_path: str) -> Dict:
    """
    Ingest all PDFs under folder_path (recursively).

    - Milvus Plan B: routes each PDF to its bank collection (e.g., tmb_policy_chunks)
    - Computes embeddings in this layer (required for Milvus insert if backend expects embedding)
    - Still works with older Chroma-style backends (upsert only)
    - Builds relations.json after processing all PDFs

    Returns summary dict.
    """
    if not folder_path or not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    backend = get_vector_backend()
    embedder = get_default_embedder()

    all_chunks_payload: List[Dict] = []
    parsed_total = 0
    stored_total = 0
    per_collection_stored: Dict[str, int] = {}

    for root, _, files in os.walk(folder_path):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue

            path = os.path.join(root, fn)
            doc_id = _infer_doc_id(path)
            bank, collection_name = infer_bank_and_collection(path)
            source_id = os.path.basename(path)

            pages = load_pdf_pages(path)
            doc_last_page = pages[-1].page_num if pages else 1
            full_text = _pdf_full_text(pages)

            # Choose chunking strategy
            chunk_dicts: List[Dict] = []

            if _is_faq_doc(path, full_text):
                # FAQ chunking (TOC/headings not required)
                chunk_dicts = chunk_faq_text(full_text, doc_id=doc_id, source_path=path)

                # If STILL empty, most likely scanned PDF / no extracted text
                if not chunk_dicts and len((full_text or "").strip()) < 2000:
                    print(f"[ingest] WARNING: very low extracted text for FAQ PDF: {path} (likely scanned; OCR needed)")
            else:
                # Existing hierarchy leaf chunking
                toc_text = _extract_toc_text(pages, max_scan_pages=3)
                toc_items = parse_toc_from_text(toc_text) if toc_text.strip() else []
                if len(toc_items) < 3:
                    toc_items = detect_headings(pages)

                tree = build_tree(toc_items)

                leaf_chunks = leaf_chunks_from_tree(
                    pages=pages,
                    doc_id=doc_id,
                    source_path=path,
                    root=tree,
                    doc_last_page=doc_last_page,
                    max_chars=2200
                )

                # Convert to dict chunks (common representation)
                for ch in leaf_chunks:
                    chunk_dicts.append({
                        "chunk_id": ch.chunk_id,
                        "text": ch.text,
                        "metadata": dict(ch.metadata or {}),
                    })

            # Prepare docs for non-milvus backends
            docs_for_store: List[Dict] = []

            # Prepare rows for milvus backend
            rows_for_milvus: List[Dict] = []

            for ch in chunk_dicts:
                text = ch["text"]
                meta = ch.get("metadata", {}) or {}
                tags = tag_concepts(text)

                # Attach common metadata
                meta.update({
                    "bank": bank,
                    "collection": collection_name,
                    "concept_tags": tags,
                    "doc_id": doc_id,
                    "source_id": source_id,
                })

                docs_for_store.append({
                    "id": ch["chunk_id"],
                    "text": text,
                    **meta,
                })

                # Relations payload (used for cross-doc graph)
                all_chunks_payload.append({
                    "chunk_id": ch["chunk_id"],
                    "doc_id": doc_id,
                    "bank": bank,
                    "collection": collection_name,
                    "source_id": source_id,
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "order_key": meta.get("order_key"),
                    "section_type": meta.get("section_type"),
                    "policy_topic": meta.get("policy_topic"),
                    "text": text,
                    "concept_tags": tags,
                })

                rows_for_milvus.append({
                    "chunk_id": ch["chunk_id"],
                    "doc_id": doc_id,
                    "bank": bank,
                    "source_id": source_id,
                    "page_start": int(meta.get("page_start", 0) or 0),
                    "page_end": int(meta.get("page_end", 0) or 0),
                    "order_key": str(meta.get("order_key", "")),
                    "section_type": str(meta.get("section_type", "")),
                    "policy_topic": str(meta.get("policy_topic") or ""),
                    "text": text,
                    # embedding added below (batch)
                })

            parsed_total += len(chunk_dicts)

            # -----------------------------
            # Store to vector backend
            # -----------------------------
            stored_now = 0

            if not rows_for_milvus:
                # Nothing to store for this PDF
                continue

            # Preferred Milvus Plan-B API
            if hasattr(backend, "upsert_chunks"):
                texts = [r["text"] for r in rows_for_milvus]
                embeddings = embedder.embed_batch(texts)
                for r, e in zip(rows_for_milvus, embeddings):
                    r["embedding"] = e

                stored_now = int(backend.upsert_chunks(collection_name, rows_for_milvus))
                stored_total += stored_now
                per_collection_stored[collection_name] = per_collection_stored.get(collection_name, 0) + stored_now
            else:
                # Backward compatible Chroma-style
                stored_now = int(backend.upsert(docs_for_store))
                stored_total += stored_now

    # Build relations after all docs processed
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
        "per_collection_stored": per_collection_stored,
        "relations_path": RELATIONS_PATH
    }


# ---------------------------
# Entry used by /v1/ingest
# ---------------------------

def ingest_path(path: str) -> Dict:
    """
    Ingest all PDFs under `path` (recursively), chunk (FAQ-aware),
    upsert to configured vector backend, and build cross-document relations.
    """
    if not path or not isinstance(path, str):
        raise ValueError("ingest_path requires a non-empty string path")

    return ingest_folder(path)
