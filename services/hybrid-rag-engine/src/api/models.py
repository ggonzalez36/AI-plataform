from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentIngestRequest(BaseModel):
    title: str = Field(..., example="Corporate Security & Compliance Policy")
    content: str = Field(..., example="All audit logs must be retained for 7 years according to SOX guidelines.")
    category: str = Field("compliance", example="compliance")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentIngestResponse(BaseModel):
    id: str
    status: str
    message: str
    collection: str

class DocumentQueryRequest(BaseModel):
    query: str = Field(..., example="What are the compliance guidelines for data retention?")
    top_k: int = Field(3, ge=1, le=20)
    category_filter: Optional[str] = Field(None, example="compliance")
    dense_weight: float = Field(0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(0.5, ge=0.0, le=1.0)

class DocumentMatch(BaseModel):
    id: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float
    title: str
    snippet: str
    category: str

class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(..., description="Fact grounding score [0, 1]")
    answer_relevancy: float = Field(..., description="Query-Answer relevance [0, 1]")
    context_precision: float = Field(..., description="Retrieval ranking precision [0, 1]")

class DocumentQueryResponse(BaseModel):
    query: str
    matches: List[DocumentMatch]
    synthesized_answer: str
    retrieval_strategy: str = "Hybrid (Dense HNSW + Sparse BM25 with RRF Fusion)"
    latency_ms: float
    collection: str
    metrics: Optional[EvaluationMetrics] = None
    trace_id: Optional[str] = None
