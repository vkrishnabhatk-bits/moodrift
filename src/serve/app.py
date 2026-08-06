import json
import time
import uuid

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from src.config import resolve
from src.features.clean import normalise
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

MODEL_DIR = resolve("data/artifacts/model")
ONNX_PATH = resolve("data/artifacts/model.onnx")
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
    global tokenizer, ort_session, model_info

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

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts)


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


def _predict_one(text: str) -> Prediction:
    """Normalise -> tokenise -> ONNX forward pass -> one schema-shaped Prediction.

    Normalising with the same `src/features/clean.normalise` the batch pipeline uses
    (rather than tokenising the raw payload, as PR #1 originally did) matters for two
    reasons: it's the train/serve skew guard - the champion was trained on normalised
    text - and it's what makes `text_hash` here match the feature store's key for the
    same string, which task 3's read-through depends on.
    """
    normalised = normalise(text)
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

    return Prediction(
        stars=stars,
        confidence=round(float(probs[stars - 1]), 4),
        probabilities={i + 1: round(float(p), 4) for i, p in enumerate(probs)},
        # flags stay at their all-False defaults here: truncated/low_signal/out_of_domain
        # detection is task B in moodrift-implementation.html's Week 3 task board.
        flags=Flags(),
        # Always "live" until task 3 wires up the feature-store read-through - there's no
        # store lookup on this path yet, so nothing is ever actually a cache hit.
        feature_source="live",
        text_hash=text_key(normalised),
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

    start_time = time.perf_counter()
    prediction = _predict_one(payload.text)
    latency = (time.perf_counter() - start_time) * 1000.0

    return PredictResponse(
        request_id=payload.request_id or str(uuid.uuid4()),
        prediction=prediction,
        model=_model_ref(),
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

    start_time = time.perf_counter()
    predictions = [_predict_one(text) for text in payload.texts]
    latency = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictResponse(
        request_id=payload.request_id or str(uuid.uuid4()),
        count=len(predictions),
        predictions=predictions,
        model=_model_ref(),
        latency_ms=round(latency, 2),
    )
