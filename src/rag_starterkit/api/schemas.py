from pydantic import BaseModel, Field
from typing import List,Literal

class IngestRequest(BaseModel):
    path: str = Field(..., description="Local folder path containing documents to ingest.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class Citation(BaseModel):
    source_id: str
    snippet: str

class AnswerQuality(BaseModel):
    groundedness: float
    confidence: float
    hallucination_risk: Literal["low", "medium", "high"]
    unsupported_points: list[str] = []
    rejected: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    quality: AnswerQuality | None = None
