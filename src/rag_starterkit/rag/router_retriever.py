# src/rag_starterkit/rag/router_retriever.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List,Union,Tuple,Any

from sentence_transformers import CrossEncoder

from rag_starterkit.rag.backend_factory import get_vector_backend
from rag_starterkit.rag.bm25_index import bm25_manager

from rag_starterkit.llm.device import get_torch_device
device = get_torch_device()
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_id: str
    bank: str
    page_start: int
    page_end: int
    order_key: str
    section_type: str
    policy_topic: str


def _expand_adjacent(
    backend,
    chunks: List[RetrievedChunk],
    collections: List[str],
    per_seed: int = 1,
) -> List[RetrievedChunk]:
    """
    Cheap adjacency expansion (within already-retrieved set):
    For each (bank, source_id), sort by order_key and include +/- neighbors.
    Does not query Milvus again; it only improves continuity from the existing set.
    """

    def parse_okey(ok: str):
        # handles "0008-0000-0001" style and "FAQ-0008" style
        nums = [int(x) for x in re.findall(r"\d+", ok or "")]
        return nums if nums else None

    seen = {c.chunk_id for c in chunks}
    expanded = list(chunks)

    by_doc: Dict[tuple, List[RetrievedChunk]] = {}
    for c in chunks:
        by_doc.setdefault((c.bank, c.source_id), []).append(c)

    for _, items in by_doc.items():
        items_sorted = sorted(
            items,
            key=lambda x: parse_okey(x.order_key) or [10**9],
        )
        for idx, _seed in enumerate(items_sorted):
            for j in range(1, per_seed + 1):
                for nb_idx in (idx - j, idx + j):
                    if 0 <= nb_idx < len(items_sorted):
                        nb = items_sorted[nb_idx]
                        if nb.chunk_id not in seen:
                            expanded.append(nb)
                            seen.add(nb.chunk_id)

    return expanded


def _select_with_bank_quota(
    reranked: List[RetrievedChunk],
    top_k_final: int,
    min_per_bank: int = 2,
) -> List[RetrievedChunk]:
    """
    Keep at least `min_per_bank` chunks per bank (where available),
    then fill remaining slots by rerank score.
    """
    if not reranked:
        return []

    by_bank: Dict[str, List[RetrievedChunk]] = {}
    for c in reranked:
        by_bank.setdefault(c.bank or "unknown", []).append(c)

    selected: List[RetrievedChunk] = []

    # Pass 1: take min_per_bank from each bank (in rerank order)
    for _, items in by_bank.items():
        take = min(min_per_bank, len(items))
        selected.extend(items[:take])

    # De-duplicate by chunk_id
    seen = set()
    uniq_selected: List[RetrievedChunk] = []
    for c in selected:
        if c.chunk_id in seen:
            continue
        uniq_selected.append(c)
        seen.add(c.chunk_id)
    selected = uniq_selected

    # Pass 2: fill remaining by global rerank score
    if len(selected) < top_k_final:
        for c in reranked:
            if c.chunk_id in seen:
                continue
            selected.append(c)
            seen.add(c.chunk_id)
            if len(selected) >= top_k_final:
                break

    selected.sort(key=lambda x: x.score, reverse=True)
    return selected[:top_k_final]


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = get_torch_device()
        self.model = CrossEncoder(
            model_name,
            device=self.device
        )

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, (c.text or "")[:1500]) for c in chunks]

        scores = self.model.predict(pairs)
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

        for c, s in zip(chunks, scores):
            c.score = float(s)

        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks[:top_k]



def _rrf_fuse(
    dense_hits: List[Dict],
    lex_hits: List[Dict],
    top_k: int,
    k: int = 60,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion over dense + lexical ranked lists.
    Returns merged hits with 'fused_score'.
    """
    fused: Dict[str, Dict] = {}

    def add_hits(hits: List[Dict]):
        for rank, h in enumerate(hits, start=1):
            cid = h.get("chunk_id")
            if not cid:
                continue
            if cid not in fused:
                fused[cid] = dict(h)
                fused[cid]["fused_score"] = 0.0
            fused[cid]["fused_score"] += 1.0 / (k + rank)

    add_hits(dense_hits)
    add_hits(lex_hits)

    out = list(fused.values())
    out.sort(key=lambda x: float(x.get("fused_score", 0.0)), reverse=True)
    return out[:top_k]


def _ensure_bm25_built(backend, collection_name: str, limit: int = 16384) -> None:
    """
    Lazy-build BM25 index for a collection on first use.
    NOTE: Your Milvus query window constraints may apply in fetch_all_texts().
    """
    if bm25_manager.has_index(collection_name):
        return
    docs = backend.fetch_all_texts(collection_name, limit=limit)
    bm25_manager.build_index(collection_name, docs)


def compress_context(
    chunks: List[RetrievedChunk],
    max_chars: int = 9000,
    dedupe_jaccard: float = 0.85,
) -> List[RetrievedChunk]:
    """
    Lightweight compression:
      - Dedupe near-duplicate chunks
      - Enforce max_chars budget
      - Preserve narrative order by (bank, source_id, order_key)
    """
    def jaccard(a: str, b: str) -> float:
        A = set((a or "").lower().split())
        B = set((b or "").lower().split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    kept: List[RetrievedChunk] = []
    total = 0

    for c in chunks:
        if total >= max_chars:
            break

        dup = False
        for k in kept:
            if jaccard((c.text or "")[:1200], (k.text or "")[:1200]) >= dedupe_jaccard:
                dup = True
                break
        if dup:
            continue

        add_len = len(c.text or "")
        if total + add_len > max_chars:
            remain = max_chars - total
            if remain > 400:
                c.text = (c.text or "")[:remain]
                kept.append(c)
                total += len(c.text or "")
            break

        kept.append(c)
        total += add_len

    kept.sort(key=lambda x: (x.bank, x.source_id, x.order_key))
    return kept

def compress_context_balanced(
    chunks: List[RetrievedChunk],
    max_chars: int = 9000,
    rbi_ratio: float = 0.35,
    dedupe_jaccard: float = 0.85,
) -> List[RetrievedChunk]:
    """
    Bank-balanced compression:
      - Allocate a fixed ratio of max_chars to RBI (default 35%)
      - Allocate remaining budget across other banks present (default 65% split evenly)
      - Within each bank, keep high-scoring chunks while deduping near-duplicates
      - Preserve narrative order inside each bank by (source_id, order_key)
      - Return merged chunks sorted by score desc (final stage already applies quota)
    """
    if len(chunks) <= 12:
        return sorted(chunks, key=lambda x: x.score, reverse=True)
    def jaccard(a: str, b: str) -> float:
        A = set((a or "").lower().split())
        B = set((b or "").lower().split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    # Group by bank
    by_bank: Dict[str, List[RetrievedChunk]] = {}
    for c in chunks:
        by_bank.setdefault(c.bank or "unknown", []).append(c)

    # Identify RBI bank label(s)
    # Your data uses "rbi" for RBI chunks.
    rbi_keys = [k for k in by_bank.keys() if k.lower() == "rbi"]

    # Determine budgets
    rbi_budget = int(max_chars * rbi_ratio) if rbi_keys else 0
    other_budget = max_chars - rbi_budget

    other_banks = [k for k in by_bank.keys() if k.lower() != "rbi"]
    per_other_budget = int(other_budget / max(1, len(other_banks))) if other_banks else 0

    def pack_bank(bank: str, items: List[RetrievedChunk], budget: int) -> List[RetrievedChunk]:
        if budget <= 0 or not items:
            return []

        # Start from score order for relevance
        items_sorted = sorted(items, key=lambda x: x.score, reverse=True)

        kept: List[RetrievedChunk] = []
        total = 0

        for c in items_sorted:
            if total >= budget:
                break

            # dedupe against already kept in this bank
            dup = False
            for k in kept:
                if jaccard((c.text or "")[:1200], (k.text or "")[:1200]) >= dedupe_jaccard:
                    dup = True
                    break
            if dup:
                continue

            add_len = len(c.text or "")
            if total + add_len > budget:
                remain = budget - total
                if remain > 400:
                    c.text = (c.text or "")[:remain]
                    kept.append(c)
                    total += len(c.text or "")
                break

            kept.append(c)
            total += add_len

        # Preserve narrative order within bank for final prompt coherence
        kept.sort(key=lambda x: (x.source_id, x.order_key))
        return kept

    packed_all: List[RetrievedChunk] = []

    # Pack RBI first (if present)
    for rk in rbi_keys:
        packed_all.extend(pack_bank(rk, by_bank[rk], rbi_budget))

    # Pack other banks
    for bk in other_banks:
        packed_all.extend(pack_bank(bk, by_bank[bk], per_other_budget))

    # Final de-dupe across banks by chunk_id
    seen = set()
    uniq: List[RetrievedChunk] = []
    for c in packed_all:
        if c.chunk_id in seen:
            continue
        uniq.append(c)
        seen.add(c.chunk_id)

    # Merge back by score desc (your final quota selector will still enforce coverage)
    uniq.sort(key=lambda x: x.score, reverse=True)

    # Fallback: if balanced packing under-fills (common on short corpora),
    # backfill with best remaining chunks (score order) while respecting max_chars.
    target_min = min(12, len(chunks))  # safe default for top_k_final up to ~8–10
    if len(uniq) < target_min:
        used = {c.chunk_id for c in uniq}
        total_chars = sum(len(c.text or "") for c in uniq)

        # candidates in score order
        remaining = sorted(chunks, key=lambda x: x.score, reverse=True)
        for c in remaining:
            if c.chunk_id in used:
                continue

            # keep dedupe against global uniq
            dup = False
            for k in uniq:
                if jaccard((c.text or "")[:1200], (k.text or "")[:1200]) >= dedupe_jaccard:
                    dup = True
                    break
            if dup:
                continue

            add_len = len(c.text or "")
            if total_chars + add_len > max_chars:
                break

            uniq.append(c)
            used.add(c.chunk_id)
            total_chars += add_len

            if len(uniq) >= target_min:
                break
    uniq.sort(key=lambda x: x.score, reverse=True)
    return uniq




class RouterRetriever:
    """
    Hybrid Router Retrieval:
      - Dense (Milvus) per collection
      - Lexical (BM25) per collection
      - Fuse (RRF)
      - Rerank (CrossEncoder)
      - Expand adjacency (cheap, within retrieved set)
      - Compress (dedupe + char budget)
      - Enforce per-bank quota in final selection
    """

    def __init__(self, embedder, 
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 enable_reranker: bool = True):
        self.backend = get_vector_backend()
        self.embedder = embedder
        self.reranker = Reranker(reranker_model) if enable_reranker else None

    def retrieve(
        self,
        query: str,
        bank_collections: List[str],
        top_k_dense_each: int = 8,
        top_k_final: int = 6,
        include_rbi: bool = True,
        # CPU latency controls:
        max_rerank_candidates: int = 40,
        per_collection_fused_extra: int = 10,
        # Coverage control:
        min_per_bank: int = 2,
        return_trace: bool = False,
        use_reranker: bool = True,
    ) -> Union[List[RetrievedChunk], Tuple[List[RetrievedChunk], Dict[str, Any]]]:
        trace = {"per_collection": {}, "global": {}}
        qv = self.embedder.embed(query)

        collections: List[str] = []
        if include_rbi:
            collections.append("rbi_faq_chunks")
        collections.extend(bank_collections)

        hits_all: List[RetrievedChunk] = []

        for col in collections:
            dense_hits = self.backend.search(
                col,
                qv,
                top_k=top_k_dense_each,
                output_fields=[
                    "chunk_id", "text", "source_id", "doc_id", "bank",
                    "page_start", "page_end", "order_key", "section_type", "policy_topic",
                ],
            )

            try:
                _ensure_bm25_built(self.backend, col, limit=16384)
                lex_hits = bm25_manager.search(col, query, top_k=top_k_dense_each)
            except Exception:
                lex_hits = []

            fused_top_k = max(1, top_k_dense_each + int(per_collection_fused_extra))
            fused_hits = _rrf_fuse(
                dense_hits=dense_hits,
                lex_hits=lex_hits,
                top_k=fused_top_k,
                k=60,
            )
            if return_trace:
                trace["per_collection"][col] = {
                    "dense_count": len(dense_hits),
                    "bm25_count": len(lex_hits),
                    "fused_count": len(fused_hits),
                    "top_dense_ids": [h.get("chunk_id") for h in dense_hits[:5]],
                    "top_bm25_ids": [h.get("chunk_id") for h in lex_hits[:5]],
                    "top_fused_ids": [h.get("chunk_id") for h in fused_hits[:5]],
                }

            for h in fused_hits:
                hits_all.append(
                    RetrievedChunk(
                        chunk_id=h.get("chunk_id", ""),
                        text=h.get("text", "") or "",
                        score=float(h.get("fused_score", h.get("score", 0.0))),
                        source_id=h.get("source_id", "") or "",
                        bank=h.get("bank", "") or "",
                        page_start=int(h.get("page_start", 0) or 0),
                        page_end=int(h.get("page_end", 0) or 0),
                        order_key=h.get("order_key", "") or "",
                        section_type=h.get("section_type", "") or "",
                        policy_topic=h.get("policy_topic", "") or "",
                    )
                )

        if not hits_all:
            return ([], trace) if return_trace else []

        # Cap reranker workload (critical on CPU)
        hits_all.sort(key=lambda c: c.score, reverse=True)  # fused_score pre-rerank
        hits_all = hits_all[:max_rerank_candidates]

        # Rerank (optional)
        if use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(query, hits_all, top_k=top_k_final * 4)
            reranked.sort(key=lambda x: x.score, reverse=True)
            # Cheap adjacency continuity (within retrieved set)
            
        else:
            # FAST PATH: keep fused_score ordering (already in c.score)
            reranked = hits_all
        reranked = _expand_adjacent(self.backend, reranked, collections, per_seed=1)
        reranked.sort(key=lambda x: x.score, reverse=True)
        # Select BEFORE compression (ensures we can fill top_k_final)
        final_pre = _select_with_bank_quota(
            reranked,
            top_k_final=top_k_final,
            min_per_bank=min_per_bank,
        )
        final_pre.sort(key=lambda x: x.score, reverse=True)

        # Compress AFTER selection (balanced)
        final = compress_context_balanced(final_pre, max_chars=9000, rbi_ratio=0.35)
        final = final[:top_k_final]

        if return_trace:
            trace["global"] = {
                "hits_all_pre_cap": "n/a",
                "rerank_in": len(hits_all),
                "rerank_out": len(reranked),
                "compressed_out": len(final),
                "final_out": len(final),
                "top_final_ids": [h.chunk_id for h in final[:8]],
            }
            return final, trace

        return final
