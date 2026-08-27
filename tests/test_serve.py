import json
import uuid

from fastapi.testclient import TestClient

from src.config import load_config, resolve
from src.serve.app import app


def test_health_endpoint_does_not_touch_the_model():
    """Liveness only: status 'ok' and an uptime, nothing about model state."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] >= 0
        assert "model_loaded" not in data


def test_ready_endpoint_when_model_loaded():
    """A loaded model and a reachable store report 200 and ready: true."""
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["feature_store_reachable"] is True


def test_model_info_endpoint():
    """/model/info reports the whole traceability story: model, tier, hashes, store, runtime."""
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model"]["name"] == "moodrift-classifier"
        assert data["model"]["alias"] != "latest"
        assert data["tier"]
        assert data["data_hash"]
        assert data["feature_store"]["rows"] > 0
        assert set(data["runtime"].keys()) >= {"onnx", "quantised", "threads", "device"}
        # threads is an int (INTRA_OP_THREADS); the schema's runtime dict must accept
        # int alongside str/bool or pydantic silently coerces 1 -> True on the wire.
        assert data["runtime"]["threads"] == 1
        assert data["served_since"]


def test_metrics_endpoint_returns_prometheus_exposition():
    """/metrics is Prometheus text exposition, and reflects predictions already served."""
    with TestClient(app) as client:
        client.post("/predict", json={"text": "Metrics probe review, works as expected."})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "moodrift_predictions_total" in body
        assert "moodrift_model_loaded 1.0" in body


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


def test_predict_batch_matches_single_item_predictions():
    """``/predict/batch`` must always agree with ``/predict`` on identical text - true
    for both graphs, but for different reasons: fp32 shares one padded tensor across
    items (app.py's ``_batched_logits``) and is bit-exact regardless; the INT8 graph
    loops single-row ONNX calls (``_looped_logits``) specifically *because* a real batched
    forward pass measurably disagrees with it (5% star flips, measured on 200 held-out
    rows - see the module docstring on ``_looped_logits``). This test is what would catch
    a regression that started batching the quantized graph again.
    """
    texts = [
        "Loved it, five stars, would buy again.",
        "Terrible, broke on day one, total waste of money.",
        "It's fine. Does what it says on the tin.",
    ]
    with TestClient(app) as client:
        batch_response = client.post("/predict/batch", json={"texts": texts})
        assert batch_response.status_code == 200
        batch_predictions = batch_response.json()["predictions"]

        for text, batched in zip(texts, batch_predictions, strict=True):
            single_response = client.post("/predict", json={"text": text})
            assert single_response.status_code == 200
            single = single_response.json()["prediction"]
            assert batched["stars"] == single["stars"]
            assert batched["confidence"] == single["confidence"]
            assert batched["probabilities"] == single["probabilities"]
            assert batched["text_hash"] == single["text_hash"]


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


def test_predict_flags_truncated_text_beyond_the_token_window():
    """Long input is served (200), not rejected, with truncated: true."""
    long_text = "This product is absolutely wonderful and I would recommend it to everyone. " * 20
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 200
        assert response.json()["prediction"]["flags"]["truncated"] is True


def test_predict_flags_low_signal_for_emoji_only_input():
    """Emoji-only input is served (200), not rejected, with low_signal: true."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "\U0001F600\U0001F600\U0001F600\U0001F600"})
        assert response.status_code == 200
        flags = response.json()["prediction"]["flags"]
        assert flags["low_signal"] is True
        assert flags["out_of_domain"] is False  # skipped once already low_signal


def test_predict_flags_low_signal_for_bare_url():
    """A bare URL reduces to the <url> placeholder token, which carries no real signal."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "https://example.com/a-product-page"})
        assert response.status_code == 200
        assert response.json()["prediction"]["flags"]["low_signal"] is True


def test_predict_flags_out_of_domain_for_non_english_text():
    """Confidently-detected non-English text is served (200) with out_of_domain: true."""
    spanish = "Este producto es fantastico y funciona muy bien todos los dias, lo recomiendo totalmente."
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": spanish})
        assert response.status_code == 200
        flags = response.json()["prediction"]["flags"]
        assert flags["out_of_domain"] is True
        assert flags["low_signal"] is False


def test_predict_flags_default_false_for_ordinary_english_text():
    """A plain, in-window, in-domain review carries no flags at all."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Arrived on time and tasted great, will buy again."})
        assert response.status_code == 200
        flags = response.json()["prediction"]["flags"]
        assert flags == {"truncated": False, "low_signal": False, "out_of_domain": False}
