# src/rag_starterkit/rag/bm25_index.py
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_RANK_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_RANK_BM25 = False


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


@dataclass
class _DocStore:
    docs: List[Dict]                 # raw doc dicts (chunk_id, text, metadata...)
    tokens: List[List[str]]          # tokenized docs
    bm25: object                     # BM25Okapi or fallback scorer


class _BM25Fallback:
    """
    Minimal BM25 (Okapi) scorer.
    Not as fast as rank-bm25 but works without extra deps.
    """
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = (sum(len(d) for d in corpus_tokens) / self.N) if self.N else 0.0

        # document frequencies
        self.df: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.tf: List[Dict[str, int]] = []

        for doc in corpus_tokens:
            freqs: Dict[str, int] = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1
            self.tf.append(freqs)
            self.doc_lens.append(len(doc))

            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        # BM25 idf with smoothing
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if self.N else 0.0

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not query_tokens or self.N == 0:
            return scores

        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
            doc_tf = self.tf[i]

            s = 0.0
            for t in query_tokens:
                f = doc_tf.get(t, 0)
                if f <= 0:
                    continue
                idf = self._idf(t)
                num = f * (self.k1 + 1)
                den = f + denom_norm
                s += idf * (num / den)
            scores[i] = s

        return scores


class BM25IndexManager:
    """
    In-memory BM25 index cache keyed by Milvus collection name.
    You build once per collection (or rebuild on demand).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._stores: Dict[str, _DocStore] = {}

    def has_index(self, collection_name: str) -> bool:
        with self._lock:
            return collection_name in self._stores

    def drop_index(self, collection_name: str) -> None:
        with self._lock:
            if collection_name in self._stores:
                del self._stores[collection_name]

    def build_index(
        self,
        collection_name: str,
        docs: List[Dict],
        min_text_len: int = 10,
    ) -> int:
        """
        Build BM25 index for a collection from a list of doc dicts.
        Returns number of indexed docs.
        """
        # Clean docs
        cleaned: List[Dict] = []
        tokens: List[List[str]] = []
        for d in docs:
            t = (d.get("text") or "").strip()
            if len(t) < min_text_len:
                continue
            cleaned.append(d)
            tokens.append(_tokenize(t))

        if _HAS_RANK_BM25:
            bm25 = BM25Okapi(tokens)  # type: ignore
        else:
            bm25 = _BM25Fallback(tokens)

        store = _DocStore(docs=cleaned, tokens=tokens, bm25=bm25)

        with self._lock:
            self._stores[collection_name] = store

        return len(cleaned)

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Return top_k docs with lexical_score.
        """
        q_tokens = _tokenize(query)

        with self._lock:
            store = self._stores.get(collection_name)

        if store is None:
            raise RuntimeError(
                f"BM25 index not built for collection '{collection_name}'. "
                f"Call build_index() first."
            )

        # Score
        if _HAS_RANK_BM25:
            scores = store.bm25.get_scores(q_tokens)  # type: ignore
        else:
            scores = store.bm25.get_scores(q_tokens)  # type: ignore

        # Rank
        scored: List[Tuple[int, float]] = [(i, float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)

        out: List[Dict] = []
        for idx, s in scored[: max(top_k * 3, top_k)]:  # grab extra then threshold-filter
            if score_threshold is not None and s < score_threshold:
                continue
            d = store.docs[idx]
            out.append({
                "chunk_id": d.get("chunk_id"),
                "text": d.get("text", ""),
                "source_id": d.get("source_id", ""),
                "doc_id": d.get("doc_id", ""),
                "bank": d.get("bank", ""),
                "page_start": d.get("page_start", 0),
                "page_end": d.get("page_end", 0),
                "order_key": d.get("order_key", ""),
                "section_type": d.get("section_type", ""),
                "policy_topic": d.get("policy_topic", ""),
                "lexical_score": s,
            })
            if len(out) >= top_k:
                break

        return out


# Singleton manager (simple pattern)
bm25_manager = BM25IndexManager()
