"""
Evaluation harness for the Bank Router RAG pipeline.

What it evaluates (per test case):
- Retrieval quality: recall@k over gold chunk_ids and/or doc_ids
- Coverage: banks present in retrieved contexts vs expected banks
- Generation quality: judge groundedness/confidence/hallucination_risk + rejection
- Latency: retrieve_ms, generate_ms, total_ms
- Optional: simple "expected substring" checks on answers

It uses the same core components as the API route:
- RouterRetriever.retrieve(...)  fileciteturn2file1L35-L89
- generate_answer(...)           fileciteturn2file0L10-L65
- judge_answer(...)              fileciteturn2file4L8-L66

Dataset format (JSONL):
Each line is one test case, e.g.

{"id":"T1",
 "query":"What is the policy on immediate credit for outstation cheques?",
 "banks":["tmb"],                       // optional (bank hints; can be bank keys or collection names)
 "collections":["tmb_policy_chunks"],   // optional alternative to banks
 "gold_chunk_ids":["doc:abcd1234..."],  // optional
 "gold_doc_ids":["tmb_cheque_collection_policy"],  // optional
 "expected_banks":["tmb","rbi"],        // optional
 "expected_answer_contains":["immediate credit","outstation"],  // optional
 "notes":"..."

}

Outputs:
- results.jsonl (per test case)
- summary.json (aggregate metrics)
- results.csv (flat view for Excel)

Usage:
python -m rag_starterkit.eval.evaluate_pipeline --dataset data/eval.jsonl --out_dir data/eval_runs/run1

Environment:
- OLLAMA_URL, RAG_LLM_MODEL (generator/judge)
- MILVUS_* envs as per your backend

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

# --- Import from your project ---
from rag_starterkit.rag.router_retriever import RouterRetriever
from rag_starterkit.rag.generator import generate_answer
from rag_starterkit.rag.embeddings import get_default_embedder


# -----------------------------
# Helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _as_collections(banks: Optional[List[str]], collections: Optional[List[str]]) -> List[str]:
    """
    Accept either:
      - collections: explicit Milvus collection names (ending with _chunks)
      - banks: bank keys (e.g., "tmb", "axis", "union") OR collection names
    """
    cols: List[str] = []
    for token in (_safe_list(collections) + _safe_list(banks)):
        if not token:
            continue
        t = str(token).strip()
        if not t:
            continue
        # If the user passed a collection, keep it
        if t.endswith("_chunks"):
            cols.append(t)
        else:
            # A light normalization: append default suffix if not a collection
            # If your API uses a richer mapping, you can import it; here we keep it simple.
            cols.append(f"{t.lower().replace(' ', '_')}_policy_chunks")
    # de-dup while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def _retrieved_to_contexts(hits) -> List[Dict[str, Any]]:
    """
    Converts RouterRetriever RetrievedChunk objects to the "contexts" format expected by prompt/citations/judge.
    Matches your FastAPI route behavior. fileciteturn2file2L74-L90
    """
    contexts = []
    for h in hits:
        contexts.append(
            {
                "id": getattr(h, "source_id", None) or getattr(h, "chunk_id", None),
                "text": getattr(h, "text", None),
                "bank": getattr(h, "bank", None),
                "source_id": getattr(h, "source_id", None),
                "chunk_id": getattr(h, "chunk_id", None),
                "doc_id": getattr(h, "doc_id", None),
                "page_start": getattr(h, "page_start", None),
                "page_end": getattr(h, "page_end", None),
                "order_key": getattr(h, "order_key", None),
                "section_type": getattr(h, "section_type", None),
                "policy_topic": getattr(h, "policy_topic", None),
                "score": getattr(h, "score", None),
            }
        )
    return contexts

def _recall_at_k(retrieved: List[str], gold: List[str]) -> float:
    if not gold:
        return float("nan")
    g = set(gold)
    r = set(retrieved)
    return 1.0 if len(g & r) > 0 else 0.0

def _contains_all(answer: str, must_contain: List[str]) -> Tuple[bool, List[str]]:
    missing = []
    a = (answer or "").lower()
    for s in must_contain or []:
        if str(s).lower() not in a:
            missing.append(str(s))
    return (len(missing) == 0), missing


# -----------------------------
# Core eval per case
# -----------------------------
def run_one(
    rr: RouterRetriever,
    query: str,
    bank_collections: List[str],
    top_k_final: int,
    *,
    expected_banks: Optional[List[str]] = None,
    gold_chunk_ids: Optional[List[str]] = None,
    gold_doc_ids: Optional[List[str]] = None,
    expected_answer_contains: Optional[List[str]] = None,
    include_rbi: bool = True,
    return_trace: bool = True,
) -> Dict[str, Any]:

    t0 = _now_ms()
    # Retrieval
    t1 = _now_ms()
    out = rr.retrieve(
        query=query,
        bank_collections=bank_collections,
        top_k_dense_each=8,
        top_k_final=top_k_final,
        include_rbi=include_rbi,
        return_trace=return_trace,
    )
    if return_trace:
        hits, trace = out
    else:
        hits, trace = out, None
    t2 = _now_ms()

    contexts = _retrieved_to_contexts(hits)

    # Generation (+ judge inside generate_answer)
    t3 = _now_ms()
    answer, citations, quality = generate_answer(query, contexts)
    t4 = _now_ms()

    # Metrics
    retrieved_chunk_ids = [c.get("chunk_id") for c in contexts if c.get("chunk_id")]
    retrieved_doc_ids = [c.get("doc_id") for c in contexts if c.get("doc_id")]
    retrieved_banks = [c.get("bank") for c in contexts if c.get("bank")]

    chunk_recall = _recall_at_k(retrieved_chunk_ids, _safe_list(gold_chunk_ids))
    doc_recall = _recall_at_k(retrieved_doc_ids, _safe_list(gold_doc_ids))

    bank_ok = float("nan")
    expected_banks_norm = [b.lower() for b in _safe_list(expected_banks)]
    if expected_banks_norm:
        retrieved_banks_norm = [str(b).lower() for b in retrieved_banks]
        bank_ok = 1.0 if any(b in retrieved_banks_norm for b in expected_banks_norm) else 0.0

    contains_ok, missing = _contains_all(answer, _safe_list(expected_answer_contains))

    result = {
        "query": query,
        "collections": ["rbi_faq_chunks"] + list(bank_collections) if include_rbi else list(bank_collections),
        "top_k_final": top_k_final,
        "timing_ms": {
            "retrieve_ms": t2 - t1,
            "generate_ms": t4 - t3,
            "total_ms": t4 - t0,
        },
        "retrieval": {
            "retrieved_chunk_ids": retrieved_chunk_ids[:top_k_final],
            "retrieved_doc_ids": retrieved_doc_ids[:top_k_final],
            "retrieved_banks": retrieved_banks[:top_k_final],
            "chunk_recall@k": chunk_recall,
            "doc_recall@k": doc_recall,
            "bank_hit@k": bank_ok,
        },
        "answer": {
            "text": answer,
            "contains_ok": bool(contains_ok),
            "missing_substrings": missing,
        },
        "quality": getattr(quality, "model_dump", lambda: asdict(quality))(),
        "citations": [getattr(c, "model_dump", lambda: c.__dict__)() for c in citations],
        "trace": trace,
    }
    return result


# -----------------------------
# Aggregate summary
# -----------------------------
def _mean_ignore_nan(xs: List[float]) -> float:
    vals = [x for x in xs if x == x]  # NaN check
    return float(statistics.mean(vals)) if vals else float("nan")

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_recalls = [r["retrieval"]["chunk_recall@k"] for r in results]
    doc_recalls = [r["retrieval"]["doc_recall@k"] for r in results]
    bank_hits = [r["retrieval"]["bank_hit@k"] for r in results]

    groundedness = [float(r["quality"].get("groundedness", 0.0)) for r in results]
    confidence = [float(r["quality"].get("confidence", 0.0)) for r in results]
    halluc = [str(r["quality"].get("hallucination_risk", "unknown")) for r in results]
    rejected = [bool(r["quality"].get("rejected", False)) for r in results]

    total_ms = [int(r["timing_ms"]["total_ms"]) for r in results]
    retrieve_ms = [int(r["timing_ms"]["retrieve_ms"]) for r in results]
    generate_ms = [int(r["timing_ms"]["generate_ms"]) for r in results]

    halluc_counts: Dict[str, int] = {}
    for h in halluc:
        halluc_counts[h] = halluc_counts.get(h, 0) + 1

    return {
        "n": len(results),
        "retrieval": {
            "mean_chunk_recall@k": _mean_ignore_nan(chunk_recalls),
            "mean_doc_recall@k": _mean_ignore_nan(doc_recalls),
            "mean_bank_hit@k": _mean_ignore_nan(bank_hits),
        },
        "answer_quality": {
            "mean_groundedness": float(statistics.mean(groundedness)) if groundedness else 0.0,
            "mean_confidence": float(statistics.mean(confidence)) if confidence else 0.0,
            "hallucination_risk_counts": halluc_counts,
            "rejection_rate": float(sum(1 for x in rejected if x) / max(1, len(rejected))),
        },
        "latency_ms": {
            "p50_total": int(statistics.median(total_ms)) if total_ms else 0,
            "p95_total": int(statistics.quantiles(total_ms, n=20)[18]) if len(total_ms) >= 20 else (max(total_ms) if total_ms else 0),
            "p50_retrieve": int(statistics.median(retrieve_ms)) if retrieve_ms else 0,
            "p50_generate": int(statistics.median(generate_ms)) if generate_ms else 0,
        },
    }


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    # A flat-ish view for Excel
    fieldnames = [
        "id","query","collections","top_k_final",
        "chunk_recall@k","doc_recall@k","bank_hit@k",
        "groundedness","confidence","hallucination_risk","rejected",
        "retrieve_ms","generate_ms","total_ms",
        "contains_ok","missing_substrings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            q = r.get("quality", {})
            w.writerow({
                "id": r.get("id"),
                "query": r.get("query"),
                "collections": ",".join(r.get("collections", [])),
                "top_k_final": r.get("top_k_final"),
                "chunk_recall@k": r["retrieval"]["chunk_recall@k"],
                "doc_recall@k": r["retrieval"]["doc_recall@k"],
                "bank_hit@k": r["retrieval"]["bank_hit@k"],
                "groundedness": q.get("groundedness"),
                "confidence": q.get("confidence"),
                "hallucination_risk": q.get("hallucination_risk"),
                "rejected": q.get("rejected", False),
                "retrieve_ms": r["timing_ms"]["retrieve_ms"],
                "generate_ms": r["timing_ms"]["generate_ms"],
                "total_ms": r["timing_ms"]["total_ms"],
                "contains_ok": r["answer"]["contains_ok"],
                "missing_substrings": "|".join(r["answer"]["missing_substrings"] or []),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to eval dataset JSONL")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--top_k", type=int, default=6, help="top_k_final to use in RouterRetriever")
    ap.add_argument("--include_rbi", action="store_true", help="Include RBI in retrieval (default off)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cases = load_jsonl(args.dataset)

    embedder = get_default_embedder()
    rr = RouterRetriever(embedder=embedder)

    results = []
    for case in cases:
        cid = case.get("id")
        query = case["query"]
        bank_cols = _as_collections(case.get("banks"), case.get("collections"))
        if not bank_cols:
            bank_cols = ["generic_policy_chunks"]

        res = run_one(
            rr=rr,
            query=query,
            bank_collections=bank_cols,
            top_k_final=int(case.get("top_k", args.top_k)),
            expected_banks=case.get("expected_banks"),
            gold_chunk_ids=case.get("gold_chunk_ids"),
            gold_doc_ids=case.get("gold_doc_ids"),
            expected_answer_contains=case.get("expected_answer_contains"),
            include_rbi=bool(case.get("include_rbi", args.include_rbi)),
            return_trace=bool(case.get("return_trace", True)),
        )
        res["id"] = cid
        res["notes"] = case.get("notes")
        results.append(res)

    # Write outputs
    write_jsonl(os.path.join(args.out_dir, "results.jsonl"), results)
    write_csv(os.path.join(args.out_dir, "results.csv"), results)

    summary = summarize(results)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
