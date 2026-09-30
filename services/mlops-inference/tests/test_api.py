from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_healthz_endpoint():
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert data["service"] == "mlops-inference"

def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "inference_requests_total" in res.text or "python_gc_objects_collected_total" in res.text

def test_predict_risk_endpoint():
    payload = {
        "features": [50.0, 2.0, 400.0, 0.35, 8.0],
        "model_version": "v1.2.0-onnx",
    }
    headers = {"X-Trace-Id": "trace-test-uuid-5678"}
    res = client.post("/api/v1/predictions/risk-score", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["trace_id"] == "trace-test-uuid-5678"
    assert data["drift_status"] in ["STABLE", "DRIFT_DETECTED"]

def test_predict_batch_endpoint():
    payload = {
        "batch": [
            [40.0, 1.0, 250.0, 0.15, 12.0],
            [90.0, 6.0, 1500.0, 0.88, 1.5],
        ]
    }
    res = client.post("/api/v1/predictions/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2

def test_drift_report_endpoint():
    res = client.get("/api/v1/predictions/drift")
    assert res.status_code == 200
    data = res.json()
    assert "feature_count" in data
    assert "features" in data
