from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class RiskScoreRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=3,
        max_length=10,
        example=[45.2, 1.0, 320.5, 0.15, 12.0],
        description="Tabular numerical features: [debt_to_income, credit_inquiries, tx_velocity, rev_utilization, account_age]",
    )
    model_version: Optional[str] = "v1.2.0-onnx"

class RiskScoreResponse(BaseModel):
    risk_score: float = Field(..., example=0.2341)
    risk_tier: str = Field(..., example="LOW")
    model_version: str
    engine: str = "ONNX Runtime v1.17"
    drift_status: str = "STABLE"
    latency_ms: float
    trace_id: Optional[str] = None

class BatchRiskScoreRequest(BaseModel):
    batch: List[List[float]] = Field(
        ...,
        min_length=1,
        max_length=500,
        example=[
            [45.2, 1.0, 320.5, 0.15, 12.0],
            [88.5, 4.0, 1250.0, 0.85, 2.0],
        ],
    )
    model_version: Optional[str] = "v1.2.0-onnx"

class BatchRiskScoreResponse(BaseModel):
    predictions: List[RiskScoreResponse]
    batch_size: int
    mean_latency_ms: float
    model_version: str

class DriftFeatureStat(BaseModel):
    ks_statistic: float
    p_value: float
    drift_detected: bool

class DriftReportResponse(BaseModel):
    feature_count: int
    sample_count: int
    overall_drift_detected: bool
    features: Dict[str, DriftFeatureStat]
    status: str
