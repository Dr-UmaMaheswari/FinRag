from rag_starterkit.rag.vectorstore import query_documents
from rag_starterkit.rag.backend_factory import get_vector_backend

def retrieve_context(query: str, top_k: int = 4):
    backend = get_vector_backend()
    return backend.query(query, top_k=top_k)

