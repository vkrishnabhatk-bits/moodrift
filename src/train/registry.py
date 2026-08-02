"""All MLflow interaction: tracking setup, run logging, and the model registry.

Two things worth knowing before reading further.

**The tracking store is SQLite, not the default file store.** The MLflow Model Registry
is only available on a database-backed store; ``mlruns/`` alone cannot register a model.
The database holds metadata, ``mlruns/`` holds artifacts.

**Promotion uses aliases, not stages.** MLflow 3.x deprecated stage transitions in favour
of *aliases* - movable named pointers to a version - so a model is referenced as
``models:/moodrift-classifier@production`` rather than by a ``Production`` stage. The
gating logic is unchanged; only the mechanism differs. Recorded in ADR-0004.

Alias meanings:

* ``@candidate`` - registered, passed the M3 evaluation gate, not yet serving.
* ``@champion``  - current best model by the comparison report.
* ``@production``- what the serving container actually loads (set in Week 3, after the
  smoke and load tests pass).
"""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import mlflow

from src.config import REPO_ROOT, resolve
from src.provenance import file_digest, git_sha

REGISTERED_MODEL = "moodrift-classifier"
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_ROOT = (REPO_ROOT / "mlruns").as_uri()


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def setup(experiment: str | None = None) -> str:
    """Point MLflow at the local SQLite store and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri())
    name = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "moodrift")
    existing = mlflow.get_experiment_by_name(name)
    if existing is None:
        mlflow.create_experiment(name, artifact_location=DEFAULT_ARTIFACT_ROOT)
    mlflow.set_experiment(name)
    return name


@contextmanager
def start_run(run_name: str, tags: dict[str, Any] | None = None):
    """Start an MLflow run with provenance tags already attached."""
    setup()
    base_tags = {"git_sha": git_sha(), "pipeline": "moodrift"}
    base_tags.update(tags or {})
    with mlflow.start_run(run_name=run_name, tags=base_tags) as run:
        yield run


def log_data_provenance(paths: dict[str, str], keys: tuple[str, ...] = ("train", "val", "test")) -> None:
    """Log a content hash per split so a run can be tied to exact data.

    This is the half of reproducibility that config alone does not give you: the same
    config over different data is a different experiment.
    """
    for key in keys:
        path = resolve(paths[key])
        if path.exists():
            mlflow.set_tag(f"data.{key}.sha256", file_digest(path)[:16])
            mlflow.log_param(f"data_{key}_rows", _parquet_rows(path))


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def log_flat_params(config: dict[str, Any], prefix: str = "") -> None:
    """Log a nested config dict as flat MLflow params (``vectorizer.word.min_df``)."""
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            log_flat_params(value, prefix=f"{name}.")
        else:
            mlflow.log_param(name, value)


def log_model(model: Any, name: str, flavor: str = "sklearn", **kwargs: Any) -> None:
    """Log a model, absorbing two MLflow 3.x differences.

    1. ``artifact_path`` was renamed to ``name``.
    2. The sklearn flavor now defaults to ``skops`` serialisation, which refuses to
       save types it does not recognise as sklearn-native. That is a sensible default
       for untrusted artifacts, but it rejects a LightGBM booster wrapped in the sklearn
       API - and it fails *after* the metrics are logged, leaving a run marked FAILED
       with no model attached. cloudpickle (the previous default) handles both tiers and
       keeps them loadable through one code path in Week 3.
    """
    module = getattr(mlflow, flavor)
    if flavor == "sklearn":
        kwargs.setdefault("serialization_format", "cloudpickle")
    if flavor == "transformers":
        # MLflow infers requirements by importing what the flavor references, which pulls
        # in torchvision - not installed here, and not needed for text classification.
        # Declaring the requirements explicitly skips that inference entirely.
        kwargs.setdefault("pip_requirements", ["torch", "transformers"])
    try:
        module.log_model(model, name=name, **kwargs)
    except TypeError:
        module.log_model(model, artifact_path=name, **kwargs)


def register(run_id: str, artifact_name: str, alias: str | None = None) -> Any:
    """Register a run's model and optionally point an alias at the new version."""
    setup()
    client = mlflow.tracking.MlflowClient()
    with suppress(mlflow.exceptions.MlflowException):
        client.create_registered_model(REGISTERED_MODEL)  # already exists on later runs

    version = mlflow.register_model(f"runs:/{run_id}/{artifact_name}", REGISTERED_MODEL)
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version.version)
        print(f"[registry] {REGISTERED_MODEL} v{version.version} -> @{alias}")
    return version


def set_alias(alias: str, version: str | int) -> None:
    """Move an alias to a specific version - this is how promotion happens."""
    setup()
    mlflow.tracking.MlflowClient().set_registered_model_alias(REGISTERED_MODEL, alias, str(version))


def model_uri(alias: str = "production") -> str:
    """The URI the serving container loads. Never 'latest' - always an explicit alias."""
    return f"models:/{REGISTERED_MODEL}@{alias}"


def describe(alias: str = "champion") -> dict[str, Any] | None:
    """Version, run and tags behind an alias, for ``/model/info`` and the demo."""
    setup()
    client = mlflow.tracking.MlflowClient()
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    except mlflow.exceptions.MlflowException:
        return None
    return {
        "name": version.name,
        "version": version.version,
        "alias": alias,
        "run_id": version.run_id,
        "description": version.description,
        "tags": dict(version.tags or {}),
    }
