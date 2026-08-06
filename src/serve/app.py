import json
import time
import uuid

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from src.config import load_config, resolve
from src.features import embed
from src.features.clean import normalise
from src.features.store import FeatureStore
from src.monitor.logger import log_prediction
from src.provenance import text_key
from src.serve.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Flags,
    ModelRef,
    PredictRequest,
    PredictResponse,
    Prediction,
    limits,
    oversized_batch,
    oversized_text,
)

app = FastAPI(
    title="MooDrift REST API",
    version="1.0.0",
    description="Sentiment analysis inference using ONNX Runtime."
)

tokenizer = None
ort_session = None
model_info: dict | None = None
feature_store: FeatureStore | None = None
tier2_cfg: dict | None = None
quantised = False

MODEL_DIR = resolve("data/artifacts/model")
ONNX_PATH = resolve("data/artifacts/model.onnx")
INT8_PATH = resolve("data/artifacts/model.int8.onnx")
MANIFEST_PATH = resolve("data/artifacts/model_manifest.json")
MAX_TOKENS = int(limits()["max_tokens"])


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@app.on_event("startup")
def load_model_and_tokenizer():
    """Load the pre-exported artifact from disk. Never falls back to a substitute model.

    The serving container never talks to MLflow directly - resolving the registry alias
    and exporting to ONNX is `python -m src.serve.export_onnx`'s job, run ahead of time.
    If that hasn't happened, the correct behaviour is to stay unloaded (503 on /predict),
    not to quietly serve a different model with a different label space.
    """
    global tokenizer, ort_session, model_info, feature_store, tier2_cfg, quantised

    config_path = MODEL_DIR / "config.json"
    if not config_path.exists() or not ONNX_PATH.exists():
        print(
            f"[serve] no exported model found at {MODEL_DIR} / {ONNX_PATH}. "
            f"Run `python -m src.serve.export_onnx` first. Staying unloaded."
        )
        return

    if MANIFEST_PATH.exists():
        model_info = json.loads(MANIFEST_PATH.read_text())
        print(f"[serve] serving {model_info['registered_model']} v{model_info['version']} "
              f"(@{model_info['alias']}, run {model_info['run_id']})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Prefer the INT8 build when it exists: task 4 measured the accuracy delta at
    # +0.0011 macro-F1 / +0.0028 macro-MAE on 1,000 held-out rows - noise-level - for a
    # 75% smaller artifact (313 -> 79 MB) and ~3x faster batched CPU inference. Falls
    # back to fp32 for anyone who ran a plain export without `--quantize`.
    active_path = INT8_PATH if INT8_PATH.exists() else ONNX_PATH
    quantised = active_path == INT8_PATH

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(str(active_path), sess_options=opts)
    print(f"[serve] ONNX runtime: {active_path.name} (quantised={quantised})")

    # The feature store's tier-2 embedding, not the champion's own weights - the champion
    # doesn't consume it on its inference path (it has its own tokeniser/forward pass), so
    # this read-through is for cache growth / monitoring features, not a serving speed-up.
    # See §"The feature store's speed-up..." in the Week 2 write-up.
    tier2_cfg = load_config("model_tier2")
    emb_cfg = tier2_cfg["embedding"]
    feature_store = FeatureStore(model=emb_cfg["model"], dimension=int(emb_cfg["dimension"]))
    embed.get_encoder(emb_cfg["model"], int(emb_cfg["max_seq_length"]))  # warm, off the request path
    print(f"[serve] feature store ready: {feature_store.stats()['rows']} rows at {feature_store.path}")


def _model_ref() -> ModelRef:
    """Provenance block for every response - built from export_onnx.py's manifest."""
    info = model_info or {}
    return ModelRef(
        name=info.get("registered_model", "moodrift-classifier"),
        version=str(info.get("version", "unknown")),
        alias=info.get("alias", "unknown"),
        run_id=info.get("run_id", "unknown"),
        git_sha=info.get("git_sha", "unknown"),
    )


def _feature_source(key: str, normalised: str) -> str:
    """Feature-store read-through: hit reuses the cached embedding, miss computes and
    writes back with source='online' - so a repeated identical request hits next time.

    The embedding itself isn't used by the champion's own forward pass (see the startup
    comment above); this exists to keep the store growing and accurate for monitoring,
    and to report `feature_source` honestly instead of the placeholder "live" task 2 left.
    """
    if feature_store.get(key) is not None:
        return "store"
    vectors = embed.encode([normalised], tier2_cfg)
    feature_store.write_many([key], vectors, source="online")
    return "live"


def _predict_one(text: str, request_id: str, model_ref: ModelRef) -> Prediction:
    """Normalise -> tokenise -> ONNX forward pass -> one schema-shaped Prediction.

    Normalising with the same `src/features/clean.normalise` the batch pipeline uses
    (rather than tokenising the raw payload, as PR #1 originally did) matters for two
    reasons: it's the train/serve skew guard - the champion was trained on normalised
    text - and it's what makes `text_hash` here match the feature store's key for the
    same string, which the read-through below depends on.

    Takes `request_id`/`model_ref` (rather than being a pure function of `text`) because
    it also writes the prediction log line - one row per prediction, not one per HTTP
    request, so a batch call logs once per item while sharing the batch's request_id.
    """
    started = time.perf_counter()
    normalised = normalise(text)
    key = text_key(normalised)
    feature_source = _feature_source(key, normalised)

    tokens = tokenizer(
        normalised,
        padding="max_length",
        max_length=MAX_TOKENS,
        truncation=True,
        return_tensors="np",
    )
    onnx_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    logits = ort_session.run(None, onnx_inputs)[0]
    probs = softmax(logits)[0]
    stars = int(np.argmax(probs) + 1)
    confidence = round(float(probs[stars - 1]), 4)
    probabilities = {i + 1: round(float(p), 4) for i, p in enumerate(probs)}
    # flags stay at their all-False defaults here: truncated/low_signal/out_of_domain
    # detection is task B in moodrift-implementation.html's Week 3 task board.
    flags = Flags()
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    log_prediction(
        request_id=request_id,
        text=normalised,
        text_hash=key,
        token_count=int(tokens["attention_mask"].sum()),
        stars=stars,
        confidence=confidence,
        probabilities=probabilities,
        flags=flags.model_dump(),
        feature_source=feature_source,
        model_version=model_ref.version,
        run_id=model_ref.run_id,
        git_sha=model_ref.git_sha,
        latency_ms=latency_ms,
    )

    return Prediction(
        stars=stars,
        confidence=confidence,
        probabilities=probabilities,
        flags=flags,
        feature_source=feature_source,
        text_hash=key,
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the export step first.")
    if oversized_text(payload.text):
        raise HTTPException(status_code=413, detail="text exceeds limits.max_input_chars")

    request_id = payload.request_id or str(uuid.uuid4())
    model_ref = _model_ref()

    start_time = time.perf_counter()
    prediction = _predict_one(payload.text, request_id, model_ref)
    latency = (time.perf_counter() - start_time) * 1000.0

    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        model=model_ref,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the export step first.")
    if oversized_batch(payload.texts):
        raise HTTPException(status_code=413, detail=f"batch exceeds limits.max_batch ({limits()['max_batch']})")
    for text in payload.texts:
        if oversized_text(text):
            raise HTTPException(status_code=413, detail="one or more texts exceed limits.max_input_chars")

    request_id = payload.request_id or str(uuid.uuid4())
    model_ref = _model_ref()

    start_time = time.perf_counter()
    predictions = [_predict_one(text, request_id, model_ref) for text in payload.texts]
    latency = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictResponse(
        request_id=request_id,
        count=len(predictions),
        predictions=predictions,
        model=model_ref,
        latency_ms=round(latency, 2),
    )
