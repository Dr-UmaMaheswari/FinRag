import json
from typing import Dict, List
import numpy as np

from rag_starterkit.rag.embeddings import get_default_embedder


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def build_relations(
    chunks: List[Dict],
    out_path: str,
    build_similarity: bool = True,
    sim_threshold: float = 0.78,
    max_similarity_edges_per_chunk: int = 5,
    max_chunks_for_similarity: int = 2000,
):
    """
    Build a lightweight relations graph across ingested chunks.

    chunks: list of dicts with keys:
      - chunk_id (str)
      - doc_id (str)
      - text (str)
      - concept_tags (list[str], optional)
      - bank (optional)

    Output JSON:
      {
        "concept_edges": [...],
        "similarity_edges": [...]
      }

    Notes:
    - Similarity is computed across different documents only.
    - For scale safety, similarity edges are capped per chunk and optionally
      capped by a max chunk count to avoid O(N^2) explosion.
    """
    rel = {"concept_edges": [], "similarity_edges": []}

    # -------------------------
    # Concept edges (shared tag)
    # -------------------------
    concept_map: Dict[str, List[str]] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            continue
        for tag in c.get("concept_tags", []) or []:
            concept_map.setdefault(tag, []).append(cid)

    for concept, ids in concept_map.items():
        if len(ids) < 2:
            continue
        # Star pattern to keep edge count smaller
        hub = ids[0]
        for other in ids[1:]:
            rel["concept_edges"].append(
                {"type": "CONCEPT_SHARED", "concept": concept, "from": hub, "to": other}
            )

    # ---------------------------------------
    # Similarity edges (cross-document only)
    # ---------------------------------------
    if build_similarity and len(chunks) > 2:
        # Scale guard: prevent accidental O(N^2) on large corpora
        work_chunks = chunks
        if len(chunks) > max_chunks_for_similarity:
            work_chunks = chunks[:max_chunks_for_similarity]

        texts = [(c.get("text") or "")[:1200] for c in work_chunks]

        embedder = get_default_embedder()
        vecs = np.array(embedder.embed_batch(texts), dtype=np.float32)

        # Build top edges per chunk (avoid huge graph)
        # We'll compute all pair scores, but only keep best few per i.
        # For large N, replace with ANN; for now this is adequate.
        for i in range(len(work_chunks)):
            best: List[Dict] = []
            for j in range(i + 1, len(work_chunks)):
                if work_chunks[i].get("doc_id") == work_chunks[j].get("doc_id"):
                    continue

                s = _cos(vecs[i], vecs[j])
                if s < sim_threshold:
                    continue

                best.append({
                    "type": "SIMILAR_TO",
                    "score": float(s),
                    "from": work_chunks[i]["chunk_id"],
                    "to": work_chunks[j]["chunk_id"],
                })

            if best:
                best.sort(key=lambda x: x["score"], reverse=True)
                rel["similarity_edges"].extend(best[:max_similarity_edges_per_chunk])

    # Write JSON
    out_dir = out_path.rsplit("/", 1)[0] if "/" in out_path else ""
    if out_dir:
        # best-effort; caller may already create directory
        try:
            import os
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)

    return rel
