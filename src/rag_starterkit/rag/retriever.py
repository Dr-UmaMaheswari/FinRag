from rag_starterkit.rag.vectorstore import query_documents

def retrieve_context(query: str, top_k: int = 4):
    return query_documents(query, top_k=top_k)
