from pydantic import BaseModel, Field
from typing import List,Literal,Optional

class IngestRequest(BaseModel):
    path: str = Field(..., description="Local folder path containing documents to ingest.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class Citation(BaseModel):
    source_id: str
    snippet: str

class AnswerQuality(BaseModel):
    groundedness: Optional[float] = None
    confidence: Optional[float] = None
    hallucination_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    unsupported_points: list[str] = []
    error: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    quality: AnswerQuality | None = None
