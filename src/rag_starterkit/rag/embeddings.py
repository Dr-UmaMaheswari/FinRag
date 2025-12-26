from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

from sentence_transformers import SentenceTransformer
from rag_starterkit.llm.device import get_torch_device
logging.info(f"Embedding model using device: {get_torch_device()}")

_MODEL = None
_MODEL_DEVICE = None

def get_embedding_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL, _MODEL_DEVICE

    device = get_torch_device()

    # Reuse model only if device matches
    if _MODEL is None or _MODEL_DEVICE != device:
        _MODEL = SentenceTransformer(name, device=device)
        _MODEL_DEVICE = device

    return _MODEL


@dataclass
class Embedder:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        self.model = get_embedding_model(self.model_name)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def get_default_embedder() -> Embedder:
    return Embedder(model_name="all-MiniLM-L6-v2")


# -----------------------------
# Backward-compatible function
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Backward compatibility for older code paths (e.g., relations_builder.py)
    that expect `embed_texts(texts)`.

    Returns a list of normalized embedding vectors (list[float]) for each text.
    """
    return get_default_embedder().embed_batch(texts)
