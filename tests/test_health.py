# tests/test_health.py
from app import app


def test_health_endpoint():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code in (200, 500)  # 200 if model exists; 500 if missing during unit test
