"""Tier 2: frozen MiniLM embeddings into a gradient-boosted head.

This tier never calls the encoder. Vectors come out of the **feature store**, keyed by a
hash of the normalised text - the same store the serving path reads in Week 3. That is
the whole point of having one: training and serving consume byte-identical features, so
a train/serve skew bug has nowhere to hide.

A cache miss here is treated as an error rather than silently recomputed: if the store is
incomplete at training time, the feature-store stage did not run and the pipeline is out
of order.

Run with ``python -m src.train.tier2``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from src.config import load_config
from src.evaluate.metrics import (
    LABELS,
    bootstrap_ci,
    compute_metrics,
    confusion_markdown,
    worst_misclassifications,
)
from src.features.sample import LABEL, MODEL_INPUT, TEXT_KEY, load_split
from src.features.store import FeatureStore
from src.provenance import set_seeds
from src.train import registry

MODEL_ARTIFACT = "model"


def _features(store: FeatureStore, frame: pd.DataFrame, split: str) -> np.ndarray:
    """Pull this split's vectors from the feature store, insisting on a full hit."""
    matrix, missing = store.matrix(frame[TEXT_KEY].tolist())
    if missing:
        raise RuntimeError(
            f"{len(missing):,} of {len(frame):,} {split} vectors are missing from the feature "
            "store. Run `python -m src.features.embed` before training tier 2."
        )
    return matrix


def train() -> dict[str, Any]:
    """Train, evaluate and log tier 2. Returns the test metrics."""
    cfg = load_config("model_tier2")
    data_cfg = load_config("data")
    set_seeds(int(cfg["seed"]))

    emb_cfg = cfg["embedding"]
    store = FeatureStore(model=emb_cfg["model"], dimension=int(emb_cfg["dimension"]))

    train_df, val_df, test_df = (load_split(s) for s in ("train", "val", "test"))
    X_train = _features(store, train_df, "train")
    X_val = _features(store, val_df, "val")
    X_test = _features(store, test_df, "test")
    print(f"[tier2] features from store: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    params = dict(cfg["classifier"]["params"])
    model = LGBMClassifier(random_state=int(cfg["seed"]), **params)

    with registry.start_run("tier2_minilm_lightgbm", tags={"tier": "2", "family": "embedding+gbm"}) as run:
        registry.log_flat_params(cfg)
        registry.log_data_provenance(data_cfg["paths"])
        mlflow.log_params(
            {
                "seed": cfg["seed"],
                "feature_source": "feature_store",
                "feature_store_rows": store.stats()["rows"],
            }
        )

        print("[tier2] fitting with early stopping on the validation split")
        model.fit(
            X_train,
            train_df[LABEL],
            eval_set=[(X_val, val_df[LABEL])],
            eval_metric="multi_logloss",
            callbacks=[
                early_stopping(int(cfg["classifier"]["early_stopping_rounds"]), verbose=False),
                log_evaluation(period=100),
            ],
        )
        mlflow.log_metric("best_iteration", int(model.best_iteration_ or params["n_estimators"]))
        print(f"[tier2] best iteration: {model.best_iteration_}")

        results = {}
        for name, X, frame in (("val", X_val, val_df), ("test", X_test, test_df)):
            # LightGBM's sklearn wrapper types predict_proba loosely; pin it so the
            # argmax below (and the diagnostics helper) get a real ndarray.
            proba = np.asarray(model.predict_proba(X))
            y_pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
            y_true = frame[LABEL].to_numpy()

            metrics = compute_metrics(y_true, y_pred)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            results[name] = metrics

            if name == "test":
                low, high = bootstrap_ci(y_true, y_pred, seed=int(cfg["seed"]))
                mlflow.log_metrics({"test_macro_f1_ci_low": low, "test_macro_f1_ci_high": high})
                _log_diagnostics(y_true, y_pred, proba, frame[MODEL_INPUT].tolist(), metrics)

            print(
                f"[tier2] {name}: macro_f1={metrics['macro_f1']:.4f} "
                f"macro_mae={metrics['macro_mae']:.4f} acc={metrics['accuracy']:.4f}"
            )

        registry.log_model(model, MODEL_ARTIFACT, flavor="sklearn")
        results["run_id"] = run.info.run_id

    print(f"[tier2] logged run {results['run_id']}")
    return results


def _log_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    texts: list[str],
    metrics: dict[str, Any],
) -> None:
    """Same diagnostics as tier 1, so the two runs are directly comparable in MLflow."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        per_class = "\n".join(
            f"| {label} | {metrics[f'precision_class_{label}']:.3f} | "
            f"{metrics[f'recall_class_{label}']:.3f} | {metrics[f'f1_class_{label}']:.3f} | "
            f"{metrics[f'support_class_{label}']:,} |"
            for label in LABELS
        )
        path = tmpdir / "confusion_matrix.md"
        path.write_text(
            "# Test-set diagnostics\n\n## Confusion matrix\n\n"
            f"{confusion_markdown(y_true, y_pred)}\n\n## Per-class performance\n\n"
            "| Star | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n"
            f"{per_class}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(path))

        worst = tmpdir / "worst_misclassifications.json"
        worst.write_text(
            json.dumps(worst_misclassifications(texts, y_true, y_pred, proba.max(axis=1), n=20), indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(worst))


if __name__ == "__main__":
    train()
