"""Prediction logger: one structlog JSONL line per served prediction.

Ships alongside the serving app in Week 3, not deferred to the monitoring build - so by
the time the drift detectors are written, there is already real accumulated data for them
to read instead of an empty log (PROJECT_PLAN.md §6/§8).

Raw review text is **not** logged by default (`logging.log_raw_text` in
``conf/serve.yaml``, default ``false``): the log ships with the repo during evaluation,
and `text_hash` is enough to detect repeats and join back to the feature store. The flag
exists for drift debugging, not routine use - documented here rather than left as a
`print` someone adds later.

Deliberately takes plain values (strings, dicts), not `src.serve.schemas` models: the
monitoring plane reads this log independently of the serving process, and shouldn't need
to import the serving plane's request/response types to know what a log line means.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.config import load_config, resolve

_configured = False


def _logger() -> Any:
    """Configure structlog for append-only JSONL on first use, then return it."""
    global _configured
    if not _configured:
        log_path = resolve(load_config("serve")["logging"]["path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.WriteLoggerFactory(file=log_path.open("a", encoding="utf-8")),
            cache_logger_on_first_use=True,
        )
        _configured = True
    return structlog.get_logger("prediction")


def log_prediction(
    *,
    request_id: str,
    text: str,
    text_hash: str,
    token_count: int,
    stars: int,
    confidence: float,
    probabilities: dict[int, float],
    flags: dict[str, bool],
    feature_source: str,
    model_version: str,
    run_id: str,
    git_sha: str,
    latency_ms: float,
) -> None:
    """Write one JSONL line. Fields match docs/api_contract.md's "Prediction log" section
    exactly - add a field in both places at once, not just here."""
    cfg = load_config("serve")["logging"]
    event: dict[str, Any] = {
        "request_id": request_id,
        "text_hash": text_hash,
        "char_count": len(text),
        "token_count": token_count,
        "stars": stars,
        "confidence": confidence,
        "probabilities": probabilities,
        "flags": flags,
        "feature_source": feature_source,
        "model_version": model_version,
        "run_id": run_id,
        "git_sha": git_sha,
        "latency_ms": latency_ms,
    }
    if cfg.get("log_raw_text", False):
        event["text"] = text
    _logger().info("prediction", **event)
