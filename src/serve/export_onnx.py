"""Export the serving model to ONNX.

The model is always resolved from the MLflow registry alias configured in
``conf/serve.yaml`` (``models:/moodrift-classifier@<alias>``) - never a local directory
guess and never a fallback to an unrelated pretrained model. An alias that is not set is
a hard failure here, not a silent substitution: an unregistered model has no business
being exported and served.

Run with ``python -m src.serve.export_onnx``.
"""

from __future__ import annotations

import json

import mlflow
import torch

from src.config import load_config, resolve
from src.provenance import git_sha
from src.train import registry

LOCAL_MODEL_DIR = resolve("data/artifacts/model")
ONNX_OUTPUT_PATH = resolve("data/artifacts/model.onnx")
MANIFEST_PATH = resolve("data/artifacts/model_manifest.json")

# Trace shape only - dynamic_axes below makes sequence_length variable at inference time.
TRACE_MAX_LENGTH = 128


def _resolve_model():
    """Load tokenizer + model from the registry alias in conf/serve.yaml, or raise."""
    alias = load_config("serve")["model"]["alias"]
    info = registry.describe(alias)
    if info is None:
        raise RuntimeError(
            f"No model registered under @{alias} ({registry.REGISTERED_MODEL}). "
            f"Refusing to export - there is nothing to serve yet. "
            f"Set the alias first (e.g. via `make register`), then re-run this export."
        )

    uri = registry.model_uri(alias)
    print(f"[export] resolving {uri} (run {info['run_id']}, version {info['version']})")
    pipeline = mlflow.transformers.load_model(uri)
    # Export target is CPU (the serving deployment target, per the Week 2 latency
    # benchmark) - mlflow.transformers auto-places the pipeline on MPS/CUDA if available,
    # which would otherwise leave the traced graph's weights on a different device than
    # the CPU dummy input below.
    model = pipeline.model.to("cpu")
    return pipeline.tokenizer, model, info


def export_model() -> None:
    tokenizer, model, info = _resolve_model()
    model.eval()

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    # Config only (id2label, num_labels, ...) - not the weights, which is what the ONNX
    # graph below is for. app.py's startup check for "was the export step run" looks for
    # this file, so it has to actually be written, not just the tokenizer's own files.
    model.config.save_pretrained(LOCAL_MODEL_DIR)

    manifest = {
        "registered_model": registry.REGISTERED_MODEL,
        "alias": info["alias"],
        "version": info["version"],
        "run_id": info["run_id"],
        "git_sha": git_sha(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[export] manifest written -> {MANIFEST_PATH}")

    dummy_text = "Great product, highly recommended!"
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=TRACE_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    print(f"[export] exporting to {ONNX_OUTPUT_PATH}...")
    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(ONNX_OUTPUT_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        # 18, not 14: this torch version's dynamo-based exporter targets opset 18 natively
        # (RobertaForSequenceClassification's LayerNormalization has no opset-14 form), and
        # attempting the downgrade fails loudly even though the export itself succeeds.
        opset_version=18,
    )

    print(f"[export] done: {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    export_model()
