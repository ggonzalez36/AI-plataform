import time
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import config
from src.engine.onnx_runner import onnx_runner
from src.monitoring.drift import drift_detector
from src.api.models import (
    RiskScoreRequest,
    RiskScoreResponse,
    BatchRiskScoreRequest,
    BatchRiskScoreResponse,
    DriftReportResponse,
)

router = APIRouter()

# Prometheus Metrics
INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Total inference scoring requests",
    ["model", "tier", "status"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Execution latency of model scoring in seconds",
    ["mode"]
)
DRIFT_EVENTS = Counter(
    "mlops_drift_detected_total",
    "Total detected data drift events"
)

@router.get("/healthz")
def healthz():
    return {
        "status": "UP",
        "service": "mlops-inference",
        "engine": onnx_runner.engine_name,
        "model_loaded": onnx_runner.loaded,
        "model_path": config.model_path,
        "model_version": config.model_version,
    }

@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.post("/api/v1/predictions/risk-score", response_model=RiskScoreResponse)
def predict_risk(req: RiskScoreRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()
    try:
        with INFERENCE_LATENCY.labels(mode="single").time():
            score = onnx_runner.predict(req.features)
            drift_detector.record_observation(req.features)

        tier = onnx_runner.classify_tier(score)
        drift_status = "DRIFT_DETECTED" if drift_detector.has_drift() else "STABLE"
        if drift_status == "DRIFT_DETECTED":
            DRIFT_EVENTS.inc()

        latency_ms = round((time.time() - start_time) * 1000, 2)
        model_version = req.model_version or config.model_version

        INFERENCE_COUNT.labels(model=model_version, tier=tier, status="200").inc()

        return RiskScoreResponse(
            risk_score=score,
            risk_tier=tier,
            model_version=model_version,
            engine=onnx_runner.engine_name,
            drift_status=drift_status,
            latency_ms=latency_ms,
            trace_id=x_trace_id,
        )
    except Exception as e:
        INFERENCE_COUNT.labels(model=req.model_version or "unknown", tier="ERROR", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

@router.post("/api/v1/predictions/batch", response_model=BatchRiskScoreResponse)
def predict_batch(req: BatchRiskScoreRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()
    try:
        with INFERENCE_LATENCY.labels(mode="batch").time():
            scores = onnx_runner.predict_batch(req.batch)
            for row in req.batch:
                drift_detector.record_observation(row)

        model_version = req.model_version or config.model_version
        drift_status = "DRIFT_DETECTED" if drift_detector.has_drift() else "STABLE"

        predictions = []
        for s in scores:
            tier = onnx_runner.classify_tier(s)
            INFERENCE_COUNT.labels(model=model_version, tier=tier, status="200").inc()
            predictions.append(RiskScoreResponse(
                risk_score=s,
                risk_tier=tier,
                model_version=model_version,
                engine=onnx_runner.engine_name,
                drift_status=drift_status,
                latency_ms=0.0,
                trace_id=x_trace_id,
            ))

        total_latency = (time.time() - start_time) * 1000
        mean_latency = round(total_latency / max(1, len(req.batch)), 3)

        return BatchRiskScoreResponse(
            predictions=predictions,
            batch_size=len(req.batch),
            mean_latency_ms=mean_latency,
            model_version=model_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

@router.get("/api/v1/predictions/drift", response_model=DriftReportResponse)
def get_drift_report():
    return drift_detector.compute_drift_report()
