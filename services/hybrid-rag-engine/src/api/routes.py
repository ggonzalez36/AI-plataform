import time
import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import config
from src.core.qdrant_store import vector_store
from src.eval.metrics import evaluator
from src.api.models import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentQueryRequest,
    DocumentQueryResponse,
    DocumentMatch,
)

router = APIRouter()

# Prometheus Telemetry Metrics
RAG_REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests processed",
    ["endpoint", "status"]
)
RAG_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Latency of RAG operations in seconds",
    ["operation"]
)

@router.get("/healthz")
def healthz():
    return {
        "status": "UP",
        "service": "hybrid-rag-engine",
        "qdrant_connected": vector_store.connected,
        "qdrant_host": config.qdrant_host,
        "qdrant_port": config.qdrant_port,
        "collection": config.collection_name,
        "indexed_documents": vector_store.count(),
    }

@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.post("/api/v1/documents/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(req: DocumentIngestRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()
    try:
        with RAG_LATENCY.labels(operation="ingest").time():
            doc_id = vector_store.ingest(
                title=req.title,
                content=req.content,
                category=req.category,
                metadata=req.metadata,
            )
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/ingest", status="201").inc()
        return DocumentIngestResponse(
            id=doc_id,
            status="SUCCESS",
            message=f"Document '{req.title}' successfully indexed with dense and sparse representations.",
            collection=config.collection_name,
        )
    except Exception as e:
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/ingest", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.post("/api/v1/documents/query", response_model=DocumentQueryResponse)
def query_documents(req: DocumentQueryRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()
    try:
        with RAG_LATENCY.labels(operation="query").time():
            raw_matches = vector_store.hybrid_search(
                query=req.query,
                top_k=req.top_k,
                category_filter=req.category_filter,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
            )

        matches = [
            DocumentMatch(
                id=m["id"],
                score=m["score"],
                dense_score=m.get("dense_score"),
                sparse_score=m.get("sparse_score"),
                rrf_score=m.get("rrf_score", m["score"]),
                title=m["title"],
                snippet=m["snippet"],
                category=m.get("category", "general"),
            )
            for m in raw_matches
        ]

        # Context synthesis
        if matches:
            top_match = matches[0]
            synthesized_answer = (
                f"Based on {top_match.category} knowledge ('{top_match.title}'): {top_match.snippet}"
            )
            snippets = [m.snippet for m in matches]
        else:
            synthesized_answer = "No matching documents found in corporate knowledge base for the given query."
            snippets = []

        # Ragas evaluation
        eval_metrics = evaluator.evaluate(
            query=req.query,
            retrieved_snippets=snippets,
            synthesized_answer=synthesized_answer,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/query", status="200").inc()

        return DocumentQueryResponse(
            query=req.query,
            matches=matches,
            synthesized_answer=synthesized_answer,
            retrieval_strategy="Hybrid (Dense HNSW + Sparse BM25 with RRF Fusion)",
            latency_ms=latency_ms,
            collection=config.collection_name,
            metrics=eval_metrics,
            trace_id=x_trace_id,
        )
    except Exception as e:
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/query", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")
