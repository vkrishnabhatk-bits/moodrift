"""Request and response schemas - the API contract, expressed as code.

Written in Week 2, ahead of the implementation, deliberately. The schemas are the part of
the contract that is cheap to get right early and expensive to change once an app, a test
suite, a Postman collection and a monitoring log all depend on it. ``docs/api_contract.md``
is the prose version of this file; if the two ever disagree, this one is right.

Three decisions encoded here rather than left to the route handlers:

* **Empty or whitespace-only text is a validation error**, not a prediction. There is no
  honest star rating for "   ", and returning one would put noise into the drift
  detectors downstream. FastAPI turns this into a 422.
* **Oversize is not validated here.** A batch of 65 texts is a well-formed request that is
  simply too large, which is 413 - and pydantic can only produce 422. The cap lives in
  ``conf/serve.yaml`` and is enforced in the route, with :func:`oversized_batch` and
  :func:`oversized_text` shared so the limit is read from one place.
* **Every prediction carries its provenance**: which model version produced it, whether
  the embedding came from the feature store or was computed live, and which flags fired.
  A prediction log without that is unusable for debugging a drift alert three weeks later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import load_config

STARS = (1, 2, 3, 4, 5)
FeatureSource = Literal["store", "live"]


def limits() -> dict[str, int]:
    """Serving limits from ``conf/serve.yaml`` - one source of truth for the caps."""
    return load_config("serve")["limits"]


def oversized_text(text: str) -> bool:
    """True when a single input exceeds the absolute character cap (-> 413)."""
    return len(text) > int(limits()["max_input_chars"])


def oversized_batch(texts: list[str]) -> bool:
    """True when a batch exceeds the item cap (-> 413)."""
    return len(texts) > int(limits()["max_batch"])


def _require_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("text must contain at least one non-whitespace character")
    return text


# ------------------------------------------------------------------- requests


class PredictRequest(BaseModel):
    """A single review to rate."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(description="Review text. Whitespace-only is rejected with 422.")
    request_id: str | None = Field(
        default=None,
        description="Client-supplied correlation ID. Generated server-side when absent, and "
        "echoed into both the response and the prediction log.",
    )

    @field_validator("text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_text(value)


class BatchPredictRequest(BaseModel):
    """Up to ``limits.max_batch`` reviews. Oversize is a 413 from the route, not a 422."""

    model_config = ConfigDict(protected_namespaces=())

    texts: list[str] = Field(min_length=1, description="Non-empty list of review texts.")
    request_id: str | None = None

    @field_validator("texts")
    @classmethod
    def _non_empty_items(cls, value: list[str]) -> list[str]:
        return [_require_text(text) for text in value]


# ------------------------------------------------------------------ responses


class Flags(BaseModel):
    """What the API noticed about the input. All three are served, none are refusals."""

    truncated: bool = Field(
        default=False, description="Input exceeded the model's token window and was cut."
    )
    low_signal: bool = Field(
        default=False, description="Almost no letters - emoji-only, punctuation-only, a bare URL."
    )
    out_of_domain: bool = Field(
        default=False, description="Detected as non-English; the model was trained English-only."
    )


class ModelRef(BaseModel):
    """Which exact model answered. Every response carries this, not just /model/info."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    alias: str = Field(description="Resolved alias, e.g. 'production' - never 'latest'.")
    run_id: str = Field(description="MLflow run that produced the artifact.")
    git_sha: str = Field(description="Commit the training code was at.")


class Prediction(BaseModel):
    """One rated review."""

    model_config = ConfigDict(protected_namespaces=())

    stars: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the returned class.")
    probabilities: dict[int, float] = Field(description="Full 1-5 distribution.")
    flags: Flags = Flags()
    feature_source: FeatureSource = Field(
        description="'store' if the embedding came from features.db, 'live' if computed on "
        "the request path and written back."
    )
    text_hash: str = Field(description="Feature-store key: sha256 of the normalised text.")

    @field_validator("probabilities")
    @classmethod
    def _full_distribution(cls, value: dict[int, float]) -> dict[int, float]:
        if tuple(sorted(value)) != STARS:
            raise ValueError(f"probabilities must cover exactly {list(STARS)}, got {sorted(value)}")
        total = sum(value.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"probabilities must sum to 1.0, got {total:.6f}")
        return value


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    prediction: Prediction
    model: ModelRef
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    count: int
    predictions: list[Prediction]
    model: ModelRef
    latency_ms: float


class FeatureStoreInfo(BaseModel):
    """Feature-store health, reported next to the model's provenance."""

    rows: int
    by_source: dict[str, int] = Field(description="Rows written by the batch job vs online.")
    last_write: str | None = None


class ModelInfoResponse(BaseModel):
    """Everything needed to answer "what is actually running right now?"."""

    model_config = ConfigDict(protected_namespaces=())

    model: ModelRef
    tier: str
    data_hash: str = Field(description="DVC hash of the splits the model was trained on.")
    image_tag: str | None = Field(default=None, description="moodrift-serve:<git-sha>.")
    feature_store: FeatureStoreInfo
    runtime: dict[str, str | bool] = Field(
        description="Execution details: onnx, quantised, threads, device."
    )
    served_since: str = Field(description="ISO timestamp of model load, not process start.")


class HealthResponse(BaseModel):
    """Liveness only. Deliberately does not touch the model - see the contract doc."""

    status: Literal["ok"] = "ok"
    uptime_seconds: float


class ReadyResponse(BaseModel):
    """Readiness: can this process actually serve a request right now? 503 when not."""

    ready: bool
    model_loaded: bool
    feature_store_reachable: bool
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error body, so clients parse one shape regardless of status code."""

    error: str = Field(description="Machine-readable code, e.g. 'payload_too_large'.")
    detail: str
    request_id: str | None = None
