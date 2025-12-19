from fastapi import FastAPI
from rag_starterkit.api.routes import router
from rag_starterkit.core.logging import configure_logging
from rag_starterkit.api.debug_routes import debug_router
configure_logging()
app = FastAPI(title="RAG Enterprise Starterkit", version="0.1.0")
app.include_router(debug_router)
app.include_router(router)
