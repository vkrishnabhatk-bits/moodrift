import json
import uuid

from fastapi.testclient import TestClient

from src.config import load_config, resolve
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
        # "live" on a cold store, "store" once a prior run has cached this exact text -
        # the feature store persists across test runs, so either is a legitimate result.
        assert prediction["feature_source"] in ("live", "store")
        assert prediction["text_hash"]
        assert data["model"]["name"] == "moodrift-classifier"


def test_predict_endpoint_feature_store_read_through():
    """A never-before-seen text misses (live) once, then hits (store) on a repeat."""
    unique_text = f"Cache read-through probe {uuid.uuid4()}"
    with TestClient(app) as client:
        first = client.post("/predict", json={"text": unique_text}).json()
        second = client.post("/predict", json={"text": unique_text}).json()

    assert first["prediction"]["feature_source"] == "live"
    assert second["prediction"]["feature_source"] == "store"
    assert first["prediction"]["text_hash"] == second["prediction"]["text_hash"]


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


def test_predict_writes_a_prediction_log_line():
    """Every prediction appends one JSONL line matching docs/api_contract.md's schema."""
    log_path = resolve(load_config("serve")["logging"]["path"])
    request_id = f"log-test-{uuid.uuid4()}"

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Decent, arrived on time.", "request_id": request_id})
    predicted = response.json()["prediction"]

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    matches = [line for line in lines if line["request_id"] == request_id]
    assert len(matches) == 1

    logged = matches[0]
    assert logged["stars"] == predicted["stars"]
    assert logged["text_hash"] == predicted["text_hash"]
    assert logged["feature_source"] == predicted["feature_source"]
    assert logged["token_count"] > 0
    assert logged["char_count"] > 0
    assert "timestamp" in logged
    # Raw text is opt-in (conf/serve.yaml logging.log_raw_text) and off by default.
    assert "text" not in logged


def test_predict_batch_writes_one_log_line_per_item():
    """A batch of N logs N lines sharing one request_id, not one line for the batch."""
    log_path = resolve(load_config("serve")["logging"]["path"])
    request_id = f"log-test-batch-{uuid.uuid4()}"

    with TestClient(app) as client:
        client.post(
            "/predict/batch",
            json={"texts": ["First review.", "Second review.", "Third review."], "request_id": request_id},
        )

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    matches = [line for line in lines if line["request_id"] == request_id]
    assert len(matches) == 3
