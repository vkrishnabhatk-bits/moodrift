"""Tier 1: TF-IDF (word + char) into a multinomial logistic regression.

The baseline every other tier has to beat. It trains on CPU in a couple of minutes, has
no GPU dependency, and its coefficients are directly readable - which makes it both the
honest comparison point and the fallback champion if the transformer never happens.

The vectoriser and classifier are a single sklearn ``Pipeline``, so the fitted vocabulary
travels with the model and serving cannot accidentally pair a model with the wrong
vectoriser.

Run with ``python -m src.train.tier1``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.evaluate.metrics import (
    LABELS,
    bootstrap_ci,
    compute_metrics,
    confusion_markdown,
    worst_misclassifications,
)
from src.features.sample import LABEL, MODEL_INPUT, load_split
from src.features.tfidf import build_vectorizer
from src.provenance import set_seeds
from src.train import registry

MODEL_ARTIFACT = "model"


def train() -> dict[str, Any]:
    """Train, evaluate and log tier 1. Returns the test metrics."""
    cfg = load_config("model_tier1")
    data_cfg = load_config("data")
    set_seeds(int(cfg["seed"]))

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"[tier1] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    pipeline = Pipeline(
        [
            ("features", build_vectorizer(cfg)),
            (
                "clf",
                LogisticRegression(
                    random_state=int(cfg["seed"]),
                    **cfg["classifier"],
                ),
            ),
        ]
    )

    with registry.start_run("tier1_tfidf_logreg", tags={"tier": "1", "family": "linear"}) as run:
        registry.log_flat_params(cfg)
        registry.log_data_provenance(data_cfg["paths"])
        mlflow.log_param("seed", cfg["seed"])

        print("[tier1] fitting (vectoriser is fit on TRAIN ONLY - no test vocabulary)")
        pipeline.fit(train_df[MODEL_INPUT], train_df[LABEL])

        n_features = len(pipeline.named_steps["features"].get_feature_names_out())
        mlflow.log_param("n_features", n_features)
        print(f"[tier1] vocabulary: {n_features:,} features")

        results = {}
        for name, frame in (("val", val_df), ("test", test_df)):
            proba = pipeline.predict_proba(frame[MODEL_INPUT])
            y_pred = pipeline.classes_[proba.argmax(axis=1)]
            y_true = frame[LABEL].to_numpy()

            metrics = compute_metrics(y_true, y_pred)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            results[name] = metrics

            if name == "test":
                low, high = bootstrap_ci(y_true, y_pred, seed=int(cfg["seed"]))
                mlflow.log_metrics({"test_macro_f1_ci_low": low, "test_macro_f1_ci_high": high})
                _log_diagnostics(y_true, y_pred, proba, frame[MODEL_INPUT].tolist(), metrics)

            print(
                f"[tier1] {name}: macro_f1={metrics['macro_f1']:.4f} "
                f"macro_mae={metrics['macro_mae']:.4f} acc={metrics['accuracy']:.4f}"
            )

        registry.log_model(pipeline, MODEL_ARTIFACT, flavor="sklearn")
        results["run_id"] = run.info.run_id

    print(f"[tier1] logged run {results['run_id']}")
    return results


def _log_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    texts: list[str],
    metrics: dict[str, Any],
) -> None:
    """Confusion matrix, per-class table and the 20 worst predictions.

    Metrics tell you *that* a model is wrong; these tell you *how*, which is what
    actually drives the next iteration.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        confusion_path = tmpdir / "confusion_matrix.md"
        per_class = "\n".join(
            f"| {label} | {metrics[f'precision_class_{label}']:.3f} | "
            f"{metrics[f'recall_class_{label}']:.3f} | {metrics[f'f1_class_{label}']:.3f} | "
            f"{metrics[f'support_class_{label}']:,} |"
            for label in LABELS
        )
        confusion_path.write_text(
            "# Test-set diagnostics\n\n"
            "## Confusion matrix\n\n"
            f"{confusion_markdown(y_true, y_pred)}\n\n"
            "## Per-class performance\n\n"
            "| Star | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n"
            f"{per_class}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(confusion_path))

        worst_path = tmpdir / "worst_misclassifications.json"
        worst = worst_misclassifications(texts, y_true, y_pred, proba.max(axis=1), n=20)
        worst_path.write_text(json.dumps(worst, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(worst_path))


if __name__ == "__main__":
    train()
