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
    """Verify that a valid text payload returns the full schemas.PredictResponse shape."""
    with TestClient(app) as client:
        payload = {"text": "The item was delivered fast and works great!"}
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "latency_ms" in data
        prediction = data["prediction"]
        assert 1 <= prediction["stars"] <= 5
        assert set(prediction["probabilities"].keys()) == {"1", "2", "3", "4", "5"}
        assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-3
        assert prediction["feature_source"] == "live"
        assert prediction["text_hash"]
        assert data["model"]["name"] == "moodrift-classifier"


def test_predict_endpoint_echoes_request_id():
    """A client-supplied request_id is echoed back rather than replaced."""
    with TestClient(app) as client:
        payload = {"text": "Fine, nothing special.", "request_id": "client-123"}
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json()["request_id"] == "client-123"


def test_predict_endpoint_rejects_whitespace_only():
    """Empty/whitespace-only text is a 422, per schemas.PredictRequest."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422


def test_predict_batch_endpoint_success():
    """A small batch returns one Prediction per input, in order."""
    with TestClient(app) as client:
        payload = {"texts": ["Loved it, five stars.", "Terrible, broke on day one."]}
        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        for prediction in data["predictions"]:
            assert 1 <= prediction["stars"] <= 5


def test_predict_batch_endpoint_rejects_oversize_batch():
    """More than limits.max_batch items is a 413, not a 422."""
    with TestClient(app) as client:
        payload = {"texts": ["fine"] * 65}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 413


def test_predict_batch_endpoint_rejects_empty_list():
    """An empty texts list is a 422 - pydantic's min_length=1 on the field."""
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422
