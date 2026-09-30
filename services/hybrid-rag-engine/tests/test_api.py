from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["service"] == "hybrid-rag-engine"
    assert data["indexed_documents"] >= 4

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text or "python_gc_objects_collected_total" in response.text

def test_query_documents_hybrid():
    payload = {
        "query": "What are the rules for SOX compliance and data retention?",
        "top_k": 2,
    }
    headers = {"X-Trace-Id": "trace-test-uuid-1234"}
    response = client.post("/api/v1/documents/query", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["matches"]) <= 2
    assert "synthesized_answer" in data
    assert data["trace_id"] == "trace-test-uuid-1234"
    assert "metrics" in data
    assert data["metrics"]["faithfulness"] > 0.0

def test_ingest_and_query_document():
    new_doc = {
        "title": "Incident Management Protocol",
        "content": "All Sev-1 outages must be escalated to the on-call Site Reliability Engineer within 5 minutes.",
        "category": "sre",
    }
    ingest_res = client.post("/api/v1/documents/ingest", json=new_doc)
    assert ingest_res.status_code == 201
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "SUCCESS"
    assert "id" in ingest_data

    # Query the ingested document
    query_res = client.post("/api/v1/documents/query", json={"query": "Sev-1 outage escalation protocol", "top_k": 3})
    assert query_res.status_code == 200
    matches = query_res.json()["matches"]
    match_titles = [m["title"] for m in matches]
    assert "Incident Management Protocol" in match_titles
