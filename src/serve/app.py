import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from transformers import AutoTokenizer

from src.config import load_config, resolve
from src.features import embed
from src.features.clean import EMAIL_TOKEN, PRODUCT_TOKEN, URL_TOKEN, normalise
from src.features.language_filter import detect as detect_language
from src.features.store import FeatureStore
from src.monitor.logger import log_prediction
from src.provenance import text_key
from src.serve.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureSource,
    FeatureStoreInfo,
    Flags,
    HealthResponse,
    ModelInfoResponse,
    ModelRef,
    Prediction,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    limits,
    oversized_batch,
    oversized_text,
)

tokenizer = None
ort_session = None
model_info: dict | None = None
feature_store: FeatureStore | None = None
tier2_cfg: dict | None = None
quantised = False
served_since: str | None = None
_started_at = time.perf_counter()

MODEL_DIR = resolve("data/artifacts/model")
ONNX_PATH = resolve("data/artifacts/model.onnx")
INT8_PATH = resolve("data/artifacts/model.int8.onnx")
MANIFEST_PATH = resolve("data/artifacts/model_manifest.json")
MAX_TOKENS = int(limits()["max_tokens"])
INTRA_OP_THREADS = 1
FLAGS_CFG = load_config("serve")["flags"]
# The three normalise() substitution tokens, stripped so a placeholder itself (e.g. a
# bare URL becoming "<url>") never counts as user-supplied signal - see _is_low_signal.
_PLACEHOLDER_TOKENS = tuple(t.strip() for t in (URL_TOKEN, EMAIL_TOKEN, PRODUCT_TOKEN))

# HF's fast (Rust) tokenizer mutates shared internal state on every call to configure
# truncation/padding, which is not safe for concurrent callers on one instance - FastAPI
# runs sync routes in a thread pool, and under load this reproducibly raised
# `RuntimeError: Already borrowed` inside `tokenizers`. The lock covers tokenisation only
# (microseconds); the ONNX forward pass, which dominates latency, runs outside it and is
# safe to call concurrently (onnxruntime's InferenceSession.run is thread-safe).
_TOKENIZER_LOCK = threading.Lock()

PREDICTIONS_TOTAL = Counter(
    "moodrift_predictions_total", "Predictions served, by feature source", ["feature_source"]
)
REQUEST_LATENCY_MS = Histogram(
    "moodrift_request_latency_ms",
    "End-to-end request latency in milliseconds, by route",
    ["route"],
    buckets=(5, 10, 20, 43.7, 60, 86, 130, 200, 500, 1000),
)
MODEL_LOADED = Gauge("moodrift_model_loaded", "1 if the ONNX model is loaded and serving, else 0")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def load_model_and_tokenizer():
    """Load the pre-exported artifact from disk, once, at process startup.

    Never falls back to a substitute model. The serving container never talks to MLflow
    directly - resolving the registry alias and exporting to ONNX is
    `python -m src.serve.export_onnx`'s job, run ahead of time. If that hasn't happened,
    the correct behaviour is to stay unloaded (503 on /predict), not to quietly serve a
    different model with a different label space.
    """
    global tokenizer, ort_session, model_info, feature_store, tier2_cfg, quantised, served_since

    config_path = MODEL_DIR / "config.json"
    if not config_path.exists() or not ONNX_PATH.exists():
        print(
            f"[serve] no exported model found at {MODEL_DIR} / {ONNX_PATH}. "
            f"Run `python -m src.serve.export_onnx` first. Staying unloaded."
        )
        MODEL_LOADED.set(0)
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
    opts.intra_op_num_threads = INTRA_OP_THREADS
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

    served_since = datetime.now(UTC).isoformat(timespec="seconds")
    MODEL_LOADED.set(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()  # runs once, before the app starts accepting requests
    yield


app = FastAPI(
    title="MooDrift REST API",
    version="1.0.0",
    description="Sentiment analysis inference using ONNX Runtime.",
    lifespan=lifespan,
)


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


def _feature_source(key: str, normalised: str) -> FeatureSource:
    """Feature-store read-through: hit reuses the cached embedding, miss computes and
    writes back with source='online' - so a repeated identical request hits next time.

    The embedding itself isn't used by the champion's own forward pass (see the startup
    comment above); this exists to keep the store growing and accurate for monitoring,
    and to report `feature_source` honestly instead of the placeholder "live" task 2 left.
    """
    assert feature_store is not None  # only called after load_model_and_tokenizer()
    if feature_store.get(key) is not None:
        return "store"
    vectors = embed.encode([normalised], tier2_cfg)
    feature_store.write_many([key], vectors, source="online")
    return "live"


def _is_low_signal(normalised: str) -> bool:
    """True when the input carries too little content to classify honestly.

    Placeholder tokens (``<url>``, ``<email>``, ``<product>``) are stripped first, so a
    bare URL or email address - already reduced to a token by ``normalise()`` - doesn't
    count as letters the model actually has an opinion about.
    """
    stripped = normalised
    for token in _PLACEHOLDER_TOKENS:
        stripped = stripped.replace(token, "")
    alpha_count = sum(ch.isalpha() for ch in stripped)
    return alpha_count < int(FLAGS_CFG["low_signal_min_alpha_chars"])


def _is_out_of_domain(normalised: str) -> bool:
    """True when confidently detected as a language other than the training corpus's.

    Reuses the same tier-0 detector (`src.features.language_filter.detect`) the sample
    stage uses to build the English-only corpus, so "out of domain" means the same thing
    here as it does in training. The confidence bar is deliberately high
    (`conf/serve.yaml` `flags.out_of_domain_min_confidence`) so ambiguous short text is
    served plainly rather than mislabelled - the effect the batch pipeline gets from its
    separate short-text carve-out, without needing a second length threshold here.
    """
    language, confidence = detect_language(normalised)
    return (
        language != FLAGS_CFG["out_of_domain_language"]
        and confidence >= float(FLAGS_CFG["out_of_domain_min_confidence"])
    )


def _detect_flags(normalised: str, truncated: bool) -> Flags:
    """What the API notices about one already-normalised input.

    Order matters: low-signal text (emoji, a bare URL) is skipped for language
    detection entirely - langid on "<url>" or three exclamation marks is meaningless,
    and would otherwise risk a spurious ``out_of_domain`` alongside ``low_signal``.
    """
    low_signal = _is_low_signal(normalised)
    out_of_domain = False if low_signal else _is_out_of_domain(normalised)
    return Flags(truncated=truncated, low_signal=low_signal, out_of_domain=out_of_domain)


def _tokenize(normalised: str) -> tuple[dict, bool]:
    """Tokenise once, under ``_TOKENIZER_LOCK``: the padded/truncated tensors the ONNX
    graph needs, plus whether the real (untruncated) token count actually overflowed
    the model's window - the ``truncated`` flag reports what the tokeniser did, not a
    character-count guess.
    """
    assert tokenizer is not None  # only called after load_model_and_tokenizer()
    with _TOKENIZER_LOCK:
        full_ids = tokenizer.encode(normalised, add_special_tokens=True)
        tokens = tokenizer(
            normalised,
            padding="max_length",
            max_length=MAX_TOKENS,
            truncation=True,
            return_tensors="np",
        )
    return tokens, len(full_ids) > MAX_TOKENS


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
    assert ort_session is not None  # only called after load_model_and_tokenizer()
    started = time.perf_counter()
    normalised = normalise(text)
    key = text_key(normalised)
    feature_source = _feature_source(key, normalised)

    tokens, truncated = _tokenize(normalised)
    onnx_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    logits = ort_session.run(None, onnx_inputs)[0]
    probs = softmax(logits)[0]
    stars = int(np.argmax(probs) + 1)
    confidence = round(float(probs[stars - 1]), 4)
    probabilities = {i + 1: round(float(p), 4) for i, p in enumerate(probs)}
    flags = _detect_flags(normalised, truncated)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    PREDICTIONS_TOTAL.labels(feature_source=feature_source).inc()

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


def _tokenize_batch(normalised_list: list[str]) -> tuple[dict, list[bool]]:
    """Batched counterpart to ``_tokenize``: one call for the whole list, so
    ``/predict/batch`` builds one padded tensor instead of looping ``_tokenize`` N times.
    Padding is already fixed at ``MAX_TOKENS`` per item, so batching changes nothing about
    per-item truncation - each row's real length is still checked against its own
    untruncated encoding.
    """
    assert tokenizer is not None  # only called after load_model_and_tokenizer()
    with _TOKENIZER_LOCK:
        full_ids = [tokenizer.encode(t, add_special_tokens=True) for t in normalised_list]
        tokens = tokenizer(
            normalised_list,
            padding="max_length",
            max_length=MAX_TOKENS,
            truncation=True,
            return_tensors="np",
        )
    truncated = [len(ids) > MAX_TOKENS for ids in full_ids]
    return tokens, truncated


def _batched_logits(tokens: dict) -> np.ndarray:
    """One ONNX call across the whole batch. Verified bit-exact against N single-item
    calls for the fp32 graph - safe to use whenever ``quantised`` is False.
    """
    assert ort_session is not None  # only called after load_model_and_tokenizer()
    onnx_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    return ort_session.run(None, onnx_inputs)[0]


def _looped_logits(tokens: dict) -> np.ndarray:
    """N single-row ONNX calls, stacked. Used for the INT8 graph only.

    ONNX Runtime's dynamic quantization recomputes its activation scale from whatever
    tensor it's actually given, so a batch-of-N forward pass is *not* numerically
    equivalent to N batch-of-1 calls on the quantized model the way it is on fp32 -
    measured on 200 held-out test rows: batching flipped the predicted star on 5% of
    them versus the single-item path. Looping here keeps ``/predict/batch`` agreeing with
    ``/predict`` on identical text, at the cost of the batching speed-up this model
    doesn't get to keep. See docs/api_contract.md's "Open for Week 3" note and
    PROJECT_PLAN.md's Week 3 retro for the measurement. A static/calibrated quantization
    would remove this constraint but was out of scope for this pass.
    """
    assert ort_session is not None  # only called after load_model_and_tokenizer()
    rows = []
    for i in range(tokens["input_ids"].shape[0]):
        single = {
            "input_ids": tokens["input_ids"][i : i + 1].astype(np.int64),
            "attention_mask": tokens["attention_mask"][i : i + 1].astype(np.int64),
        }
        rows.append(ort_session.run(None, single)[0][0])
    return np.stack(rows)


def _predict_many(texts: list[str], request_id: str, model_ref: ModelRef) -> list[Prediction]:
    """Batched counterpart to ``_predict_one``: one tokenisation call always, and one
    ONNX forward pass for the whole batch when that's numerically safe (fp32) - closes
    most of the "Open for Week 3" gap in docs/api_contract.md, where ``/predict/batch``
    measured ~120 ms x N because it looped the single-item path unconditionally. The
    quantized model still loops its forward pass; see ``_looped_logits``. Per-item work
    that was already cheap (normalisation, feature-store lookup, flag detection, logging)
    stays per-item either way.
    """
    started = time.perf_counter()
    normalised_list = [normalise(t) for t in texts]
    keys = [text_key(n) for n in normalised_list]
    feature_sources = [_feature_source(k, n) for k, n in zip(keys, normalised_list, strict=True)]

    tokens, truncated = _tokenize_batch(normalised_list)
    logits = _looped_logits(tokens) if quantised else _batched_logits(tokens)
    probs = softmax(logits)
    # Tokenisation + the forward pass are genuinely shared across the batch; amortise
    # that cost evenly across items for the prediction log's latency_ms field rather than
    # recording 0 for work that did happen.
    shared_ms = (time.perf_counter() - started) * 1000.0 / len(texts)

    predictions = []
    for i, normalised in enumerate(normalised_list):
        row = probs[i]
        stars = int(np.argmax(row) + 1)
        confidence = round(float(row[stars - 1]), 4)
        probabilities = {j + 1: round(float(p), 4) for j, p in enumerate(row)}
        flags = _detect_flags(normalised, truncated[i])
        token_count = int(tokens["attention_mask"][i].sum())
        PREDICTIONS_TOTAL.labels(feature_source=feature_sources[i]).inc()

        log_prediction(
            request_id=request_id,
            text=normalised,
            text_hash=keys[i],
            token_count=token_count,
            stars=stars,
            confidence=confidence,
            probabilities=probabilities,
            flags=flags.model_dump(),
            feature_source=feature_sources[i],
            model_version=model_ref.version,
            run_id=model_ref.run_id,
            git_sha=model_ref.git_sha,
            latency_ms=round(shared_ms, 2),
        )
        predictions.append(
            Prediction(
                stars=stars,
                confidence=confidence,
                probabilities=probabilities,
                flags=flags,
                feature_source=feature_sources[i],
                text_hash=keys[i],
            )
        )
    return predictions


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness only - deliberately does not touch the model.

    A liveness probe that fails while the model reloads would have the orchestrator
    restart a process that was about to be fine; that distinction is exactly why
    ``/ready`` (below) is a separate endpoint. ``uptime_seconds`` counts from process
    start, not model load - use ``/ready`` and ``/model/info`` for anything about the
    model itself.
    """
    return HealthResponse(uptime_seconds=round(time.perf_counter() - _started_at, 2))


@app.get("/ready", response_model=ReadyResponse)
def readiness():
    """Readiness: model loaded and the feature store reachable. 503 when either is not."""
    model_loaded = ort_session is not None
    store_reachable = False
    if feature_store is not None:
        try:
            feature_store.stats()
            store_reachable = True
        except Exception:  # noqa: BLE001 - any store failure means "not reachable"
            store_reachable = False

    ready = model_loaded and store_reachable
    detail = None
    if not model_loaded:
        detail = "model not loaded - run `python -m src.serve.export_onnx` and restart"
    elif not store_reachable:
        detail = "feature store unreachable"

    body = ReadyResponse(
        ready=ready, model_loaded=model_loaded, feature_store_reachable=store_reachable, detail=detail
    )
    if not ready:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info_endpoint():
    """What is actually running right now - the whole traceability story in one call."""
    if model_info is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the export step first.")

    store_stats = feature_store.stats()
    return ModelInfoResponse(
        model=_model_ref(),
        tier=model_info.get("tier", "unknown"),
        data_hash=model_info.get("data_hash", "unknown"),
        image_tag=os.environ.get("IMAGE_TAG"),
        feature_store=FeatureStoreInfo(
            rows=store_stats["rows"],
            by_source=store_stats["by_source"],
            last_write=store_stats["last_write"],
        ),
        runtime={
            "onnx": True,
            "quantised": quantised,
            "threads": INTRA_OP_THREADS,
            "device": "cpu",
        },
        served_since=served_since or "unknown",
    )


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    REQUEST_LATENCY_MS.labels(route="/predict").observe(latency)

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
    predictions = _predict_many(payload.texts, request_id, model_ref)
    latency = (time.perf_counter() - start_time) * 1000.0
    REQUEST_LATENCY_MS.labels(route="/predict/batch").observe(latency)

    return BatchPredictResponse(
        request_id=request_id,
        count=len(predictions),
        predictions=predictions,
        model=model_ref,
        latency_ms=round(latency, 2),
    )
