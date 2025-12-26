import os
from rag_starterkit.rag.milvus_backend import MilvusBackend

def get_vector_backend():
    # backend = os.getenv("VECTOR_BACKEND", "milvus").lower()
    # if backend == "chroma":
    #     return ChromaBackend()
    return MilvusBackend()
