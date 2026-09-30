import os
from dataclasses import dataclass

@dataclass
class MLOpsConfig:
    port: int = int(os.getenv("PORT", "8002"))
    model_path: str = os.getenv("MODEL_PATH", "model/risk_model.onnx")
    model_version: str = os.getenv("MODEL_VERSION", "v1.2.0-onnx")
    drift_threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
    environment: str = os.getenv("ENVIRONMENT", "development")

config = MLOpsConfig()
