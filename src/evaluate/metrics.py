"""Metrics for 5-class ordinal star-rating prediction.

Accuracy is deliberately *not* the headline number. With ~64% of this corpus sitting at
5 stars, a model that predicts 5 unconditionally scores ~0.64 accuracy while being
useless. Two metrics carry the evaluation instead:

* **macro-F1** - averages per-class F1, so the rare 2- and 3-star classes count as much
  as the dominant 5-star one.
* **macro-averaged MAE** - the ordinal dimension. macro-F1 treats "predicted 4, actually
  5" and "predicted 1, actually 5" as equally wrong; for a star rating they are not.
  Averaging MAE *per true class* rather than over all rows keeps the majority class from
  hiding errors on the minority ones.

Quadratic weighted kappa is reported alongside as the standard ordinal-agreement measure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

LABELS = [1, 2, 3, 4, 5]


def macro_mae(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> float:
    """Mean absolute error in stars, averaged over true classes rather than rows.

    Row-averaged MAE on this corpus is dominated by the 5-star class. Averaging per class
    first gives every rating equal weight, matching how macro-F1 treats them.
    """
    labels = labels or LABELS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = [
        float(np.abs(y_pred[y_true == label] - label).mean())
        for label in labels
        if np.any(y_true == label)
    ]
    return float(np.mean(per_class)) if per_class else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> dict[str, Any]:
    """Full metric set for one model on one split."""
    labels = labels or LABELS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_mae": macro_mae(y_true, y_pred, labels),
        "row_mae": float(np.abs(y_pred - y_true).mean()),
        # Quadratic weights: being two stars off is penalised four times as much as one.
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        # Adjacent accuracy: how often we are within one star. A practical usefulness
        # measure that exact accuracy understates on an ordinal task.
        "adjacent_accuracy": float((np.abs(y_pred - y_true) <= 1).mean()),
    }
    for label in labels:
        entry = report.get(str(label), {})
        metrics[f"f1_class_{label}"] = float(entry.get("f1-score", 0.0))
        metrics[f"precision_class_{label}"] = float(entry.get("precision", 0.0))
        metrics[f"recall_class_{label}"] = float(entry.get("recall", 0.0))
        metrics[f"support_class_{label}"] = int(entry.get("support", 0))
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric.

    A bare point estimate invites over-reading a 0.5-point gap between two models. The
    interval is what tells you whether a difference is real.
    """
    metric_fn = metric_fn or (
        lambda t, p: float(f1_score(t, p, labels=LABELS, average="macro", zero_division=0))
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> np.ndarray:
    """Raw confusion matrix, rows = true, columns = predicted."""
    return confusion_matrix(y_true, y_pred, labels=labels or LABELS)


def confusion_markdown(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> str:
    """Confusion matrix as a markdown table, for logging as an MLflow artifact."""
    labels = labels or LABELS
    cm = confusion(y_true, y_pred, labels)
    header = "| true \\ pred | " + " | ".join(str(c) for c in labels) + " |"
    divider = "|---" * (len(labels) + 1) + "|"
    rows = [f"| **{labels[i]}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |" for i in range(len(labels))]
    return "\n".join([header, divider, *rows])


def worst_misclassifications(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray | None = None,
    n: int = 20,
) -> list[dict[str, Any]]:
    """The n most badly wrong predictions - confidently wrong, and far off in stars.

    Reading these is the fastest way to find a broken preprocessing step or a genuine
    ambiguity in the labels; a metric alone will never point at either.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    distance = np.abs(y_pred - y_true).astype(float)
    severity = distance * (np.asarray(confidences) if confidences is not None else 1.0)

    order = np.argsort(-severity)[:n]
    return [
        {
            "true": int(y_true[i]),
            "predicted": int(y_pred[i]),
            "stars_off": int(distance[i]),
            "confidence": float(confidences[i]) if confidences is not None else None,
            "text": texts[i][:300],
        }
        for i in order
        if distance[i] > 0
    ]
