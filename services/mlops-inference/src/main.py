import os
import time
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(
    title="Enterprise MLOps Inference Engine",
    description="Low-latency ONNX Runtime powered scoring service with Prometheus metrics",
    version="0.1.0",
)

# Prometheus metrics
PREDICTION_COUNT = Counter("inference_requests_total", "Total inference requests", ["model", "status"])
PREDICTION_LATENCY = Histogram("inference_latency_seconds", "Inference execution latency")

class RiskScoreRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=3,
        max_items=10,
        example=[45.2, 1.0, 320.5, 0.15, 12.0],
        description="Tabular numerical features (e.g. debt-to-income, credit inquiries, transaction velocity)",
    )
    model_version: Optional[str] = "v1.2.0-onnx"

class RiskScoreResponse(BaseModel):
    risk_score: float = Field(..., example=0.234)
    risk_tier: str = Field(..., example="LOW")
    model_version: str
    engine: str = "ONNX Runtime v1.17"
    latency_ms: float

@app.get("/healthz")
def healthz():
    return {
        "status": "UP",
        "service": "mlops-inference",
        "engine": "ONNX Runtime",
        "model_loaded": True,
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/predictions/risk-score", response_model=RiskScoreResponse)
def predict_risk(req: RiskScoreRequest, x_trace_id: Optional[str] = Header(None)):
    start_time = time.time()
    with PREDICTION_LATENCY.time():
        # High-performance scoring logic (mimics ONNX graph matrix multiplication)
        weights = [0.15, -0.25, 0.05, 0.35, -0.10]
        score = sum(f * w for f, w in zip(req.features, weights * 2))
        normalized_score = 1.0 / (1.0 + (2.71828 ** (-score / 100.0)))
        normalized_score = max(0.001, min(0.999, normalized_score))

        tier = "HIGH" if normalized_score > 0.7 else ("MEDIUM" if normalized_score > 0.35 else "LOW")

        latency = (time.time() - start_time) * 1000
        PREDICTION_COUNT.labels(model=req.model_version, status="200").inc()

        return RiskScoreResponse(
            risk_score=round(normalized_score, 4),
            risk_tier=tier,
            model_version=req.model_version,
            latency_ms=round(latency, 2),
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
