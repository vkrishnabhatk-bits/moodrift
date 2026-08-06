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
INT8_OUTPUT_PATH = resolve("data/artifacts/model.int8.onnx")
MANIFEST_PATH = resolve("data/artifacts/model_manifest.json")

# Trace shape only - dynamic_axes below makes sequence_length variable at inference time.
TRACE_MAX_LENGTH = 128
DEFAULT_ACCURACY_SAMPLES = 1000


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
        opset_version=17,
        # This torch version defaults to the new torch.export-based ("dynamo") exporter,
        # which produces a graph onnx's shape_inference chokes on ("Inferred shape and
        # existing shape differ in dimension 0: (768) vs (5)" on the classification head) -
        # onnxruntime.quantization calls that internally and fails hard on it. The legacy
        # TorchScript-based exporter (dynamo=False) doesn't have the bug and verified
        # byte-for-byte identical predictions against the dynamo export before this switch.
        dynamo=False,
    )

    print(f"[export] done: {ONNX_OUTPUT_PATH}")


def _onnx_size_mb(path) -> float:
    """On-disk size, including the external-data sidecar this exporter always writes.

    torch's dynamo exporter puts the actual weights in ``<name>.data`` next to a small
    graph-only ``.onnx`` file (910 KB vs 328 MB on the champion) - measuring only the
    ``.onnx`` file's size would understate the model's real footprint by ~350x.
    """
    total = path.stat().st_size
    data_file = path.with_name(path.name + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)


def quantize_model() -> None:
    """INT8 dynamic quantisation of the fp32 export.

    Dynamic, not static: static quantisation needs a calibration dataset and buys a bit
    more speed, but the plan only ever scoped this as a size optimisation (§10 - measured
    in Week 2, the champion has ~3x latency headroom even at fp32), so the simpler,
    calibration-free approach is the right amount of engineering for the actual goal.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if not ONNX_OUTPUT_PATH.exists():
        raise RuntimeError(f"{ONNX_OUTPUT_PATH} not found - run the export first.")

    # Recommended before quantize_dynamic regardless of exporter: folds constants and
    # infers shapes ahead of time, rather than quantize_dynamic doing it implicitly.
    preprocessed_path = resolve("data/artifacts/model.preprocessed.onnx")
    print(f"[quantize] preprocessing (shape inference + optimisation) -> {preprocessed_path}")
    quant_pre_process(
        input_model=str(ONNX_OUTPUT_PATH),
        output_model_path=str(preprocessed_path),
        save_as_external_data=True,
    )

    print(f"[quantize] INT8 dynamic quantisation -> {INT8_OUTPUT_PATH}")
    quantize_dynamic(
        model_input=str(preprocessed_path),
        model_output=str(INT8_OUTPUT_PATH),
        weight_type=QuantType.QUInt8,
    )
    fp32_mb, int8_mb = _onnx_size_mb(ONNX_OUTPUT_PATH), _onnx_size_mb(INT8_OUTPUT_PATH)
    print(f"[quantize] size: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({(1 - int8_mb / fp32_mb) * 100:.0f}% smaller)")


def measure_accuracy_delta(n_samples: int = DEFAULT_ACCURACY_SAMPLES, batch_size: int = 32) -> dict:
    """Compare fp32 vs INT8 on real held-out data instead of assuming quantisation is free.

    Runs both ONNX builds over the same sample of the real test split and compares
    macro-F1 / macro-MAE with `src.evaluate.metrics.compute_metrics` - the identical
    function every tier's own evaluation uses, so this number is comparable to the
    numbers already in docs/model_comparison.md rather than a one-off approximation.
    """
    import time

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from src.evaluate.metrics import compute_metrics
    from src.features.sample import LABEL, MODEL_INPUT, load_split

    if not INT8_OUTPUT_PATH.exists():
        raise RuntimeError(f"{INT8_OUTPUT_PATH} not found - run quantize_model() first.")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    max_tokens = int(load_config("serve")["limits"]["max_tokens"])

    test = load_split("test")
    sample = test.sample(n=min(n_samples, len(test)), random_state=42)
    texts = sample[MODEL_INPUT].tolist()
    y_true = sample[LABEL].to_numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # matches the serving configuration in app.py

    def predict_all(path) -> tuple[np.ndarray, float]:
        session = ort.InferenceSession(str(path), sess_options=opts)
        predictions = []
        started = time.perf_counter()
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            tokens = tokenizer(
                chunk, padding="max_length", max_length=max_tokens, truncation=True, return_tensors="np"
            )
            logits = session.run(
                None,
                {
                    "input_ids": tokens["input_ids"].astype(np.int64),
                    "attention_mask": tokens["attention_mask"].astype(np.int64),
                },
            )[0]
            predictions.append(np.argmax(logits, axis=-1) + 1)
        elapsed = time.perf_counter() - started
        return np.concatenate(predictions), elapsed

    print(f"[quantize] measuring accuracy delta on {len(texts):,} test-split rows...")
    fp32_preds, fp32_seconds = predict_all(ONNX_OUTPUT_PATH)
    int8_preds, int8_seconds = predict_all(INT8_OUTPUT_PATH)

    fp32_metrics = compute_metrics(y_true, fp32_preds)
    int8_metrics = compute_metrics(y_true, int8_preds)

    result = {
        "n_samples": len(texts),
        "fp32_macro_f1": fp32_metrics["macro_f1"],
        "int8_macro_f1": int8_metrics["macro_f1"],
        "macro_f1_delta": int8_metrics["macro_f1"] - fp32_metrics["macro_f1"],
        "fp32_macro_mae": fp32_metrics["macro_mae"],
        "int8_macro_mae": int8_metrics["macro_mae"],
        "macro_mae_delta": int8_metrics["macro_mae"] - fp32_metrics["macro_mae"],
        "fp32_seconds": fp32_seconds,
        "int8_seconds": int8_seconds,
        "fp32_mb": _onnx_size_mb(ONNX_OUTPUT_PATH),
        "int8_mb": _onnx_size_mb(INT8_OUTPUT_PATH),
    }

    print(
        f"[quantize] macro-F1  fp32={result['fp32_macro_f1']:.4f}  int8={result['int8_macro_f1']:.4f}"
        f"  delta={result['macro_f1_delta']:+.4f}"
    )
    print(
        f"[quantize] macro-MAE fp32={result['fp32_macro_mae']:.4f}  int8={result['int8_macro_mae']:.4f}"
        f"  delta={result['macro_mae_delta']:+.4f}"
    )
    print(
        f"[quantize] size      fp32={result['fp32_mb']:.1f} MB  int8={result['int8_mb']:.1f} MB"
        f"  ({fp32_seconds:.1f}s vs {int8_seconds:.1f}s over {len(texts)} rows, batch={batch_size})"
    )

    _log_accuracy_delta(result)
    return result


def _log_accuracy_delta(result: dict) -> None:
    """Record the measured delta on the model's own MLflow run.

    Logged onto the run rather than a side file: anyone auditing this model later reads
    it from the same place as every other metric, and it can't drift out of sync with
    which run/version was actually measured.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    client = mlflow.tracking.MlflowClient()
    for key, value in result.items():
        if key == "n_samples":
            continue
        client.log_metric(manifest["run_id"], f"onnx_{key}", value)
    print(f"[quantize] logged onnx_* metrics to run {manifest['run_id']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantize", action="store_true",
        help="Also produce an INT8 build and measure its accuracy delta against fp32.",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_ACCURACY_SAMPLES,
        help="Test-split rows to evaluate the accuracy delta on (default: %(default)s).",
    )
    args = parser.parse_args()

    export_model()
    if args.quantize:
        quantize_model()
        measure_accuracy_delta(args.samples)
