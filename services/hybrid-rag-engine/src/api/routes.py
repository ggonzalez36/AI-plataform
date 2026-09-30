import time
import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import config
from src.core.qdrant_store import vector_store
from src.eval.metrics import evaluator
from src.security.guardrails import guardrails
from src.security.dlp import redactor
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
SECURITY_EVENTS = Counter(
    "rag_security_events_total",
    "Security guardrail events triggered",
    ["type", "rule"]
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
        "guardrails_active": True,
        "dlp_redactor_active": True,
    }

@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.post("/api/v1/documents/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(req: DocumentIngestRequest, x_trace_id: Optional[str] = Header(None)):
    try:
        # 1. Apply Sensitive Data Loss Prevention (PII Sanitization)
        sanitized_content, redaction_counts = redactor.sanitize(req.content)
        sanitized_title, _ = redactor.sanitize(req.title)

        if redaction_counts:
            SECURITY_EVENTS.labels(type="pii_redaction", rule="INGEST_DLP").inc()

        # 2. Ingest sanitized document into Vector Store
        with RAG_LATENCY.labels(operation="ingest").time():
            doc_id = vector_store.ingest(
                title=sanitized_title,
                content=sanitized_content,
                category=req.category,
                metadata={**(req.metadata or {}), "redactions": redaction_counts},
            )

        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/ingest", status="201").inc()
        msg = f"Document '{sanitized_title}' indexed successfully."
        if redaction_counts:
            msg += f" (DLP sanitized: {redaction_counts})"

        return DocumentIngestResponse(
            id=doc_id,
            status="SUCCESS",
            message=msg,
            collection=config.collection_name,
        )
    except Exception as e:
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/ingest", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.post("/api/v1/documents/query", response_model=DocumentQueryResponse)
def query_documents(req: DocumentQueryRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()

    # 1. OWASP LLM01: Prompt Injection & Jailbreak Guardrail
    is_blocked, threat_score, rule = guardrails.inspect(req.query)
    if is_blocked:
        SECURITY_EVENTS.labels(type="prompt_injection", rule=rule or "UNKNOWN").inc()
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/query", status="400").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Security Guardrail Violation",
                "code": "PROMPT_INJECTION_DETECTED",
                "message": f"Query blocked by AI Security Guardrails ({rule})",
                "threat_score": threat_score,
                "rule": rule,
                "trace_id": x_trace_id,
            }
        )

    # 2. Sanitize query input from potential PII leakage
    sanitized_query, _ = redactor.sanitize(req.query)

    try:
        with RAG_LATENCY.labels(operation="query").time():
            raw_matches = vector_store.hybrid_search(
                query=sanitized_query,
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
            raw_answer = f"Based on {top_match.category} knowledge ('{top_match.title}'): {top_match.snippet}"
            snippets = [m.snippet for m in matches]
        else:
            raw_answer = "No matching documents found in corporate knowledge base for the given query."
            snippets = []

        # 3. Post-synthesis DLP check: ensure no sensitive data leaks in output
        sanitized_answer, _ = redactor.sanitize(raw_answer)

        # Ragas evaluation
        eval_metrics = evaluator.evaluate(
            query=sanitized_query,
            retrieved_snippets=snippets,
            synthesized_answer=sanitized_answer,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/query", status="200").inc()

        return DocumentQueryResponse(
            query=sanitized_query,
            matches=matches,
            synthesized_answer=sanitized_answer,
            retrieval_strategy="Hybrid (Dense HNSW + Sparse BM25 with RRF Fusion)",
            latency_ms=latency_ms,
            collection=config.collection_name,
            metrics=eval_metrics,
            trace_id=x_trace_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        RAG_REQUEST_COUNT.labels(endpoint="/api/v1/documents/query", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")
