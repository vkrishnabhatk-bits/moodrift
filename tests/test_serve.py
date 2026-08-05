import pytest
from fastapi.testclient import TestClient
from src.serve.app import app

def test_health_endpoint():
    """Verify that the health check route returns HTTP 200 OK."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_predict_endpoint_success():
    """Verify that a valid text payload returns a successful prediction."""
    with TestClient(app) as client:
        payload = {"text": "The item was delivered fast and works great!"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_score" in data
        assert "probabilities" in data
        assert "latency_ms" in data