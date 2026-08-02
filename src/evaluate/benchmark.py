"""Serving-cost benchmark: per-tier inference latency and artifact size.

Macro-F1 alone cannot justify a champion pick. A model that wins by 8 points of macro-F1
while being 20x slower and 20x larger is a different decision from one that wins for free,
and the comparison report is only honest if it shows all three numbers together. This
module produces the two that training does not: **latency** and **size**.

Four choices worth stating:

* **Batch size 1, on one core.** That is the serving target (p95 < 130 ms, batch=1, CPU),
  not the batched throughput a training loop would measure. Tier 3 is timed on CPU even
  though it was trained on the GPU. Threads are pinned to 1 for every tier - see the
  OpenMP note below - which makes the tiers directly comparable and the numbers a
  conservative bound rather than a best case.
* **End to end, including feature extraction.** Each tier is timed through the whole path
  a request would take - vectorising for tier 1, encoding for tier 2, tokenising for
  tier 3 - because that is what the API's p95 will actually contain. Timing only the
  classifier head would flatter tier 2 enormously and mean nothing.
* **Tier 2 is timed twice**, once with a feature-store miss (encode live) and once with a
  hit (read the cached vector). The gap is the concrete argument for the feature store,
  and it is measured rather than asserted.
* **Each tier runs in its own subprocess.** Partly isolation - one tier's threading
  settings cannot leak into another's numbers - and partly necessity, below.

**The OpenMP conflict.** PyTorch and LightGBM each ship their own libomp. Initialising
both in one process on macOS/arm64 kills it: with torch first the process segfaults
(SIGSEGV) inside LightGBM, with LightGBM first it deadlocks inside torch's forward pass.
Both failures are silent - no traceback, no OMP #15 message. Pinning ``OMP_NUM_THREADS=1``
makes the two runtimes coexist, which is why every subprocess is launched with it set.
Tier 2 is the only tier that needs both libraries at once (MiniLM encoder + LightGBM
head), so this is a live constraint on serving tier 2, not just on measuring it - noted
for Week 3, where the champion (tier 3) does not hit it.

Results are logged back onto the originating MLflow run, so ``compare.py`` keeps reading
one source of truth instead of a side file that can drift.

Run with ``python -m src.evaluate.benchmark``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from src.config import load_config
from src.features.sample import MODEL_INPUT, TEXT_KEY, load_split
from src.train import registry

MODEL_ARTIFACT = "model"
DEFAULT_SAMPLES = 200
DEFAULT_WARMUP = 20

# Metric names are prefixed `serve_` so they sort away from the `test_`/`val_` metrics
# training logs, and read unambiguously as a property of serving rather than of fit.
P50_METRIC = "serve_latency_p50_ms"
P95_METRIC = "serve_latency_p95_ms"
P99_METRIC = "serve_latency_p99_ms"
SIZE_METRIC = "serve_artifact_mb"
CACHED_METRIC = "serve_latency_p95_ms_cache_hit"

# Single-threaded, deterministic-ish, and the only configuration in which torch and
# LightGBM survive in one process (see the module docstring).
PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 of a list of per-request timings."""
    array = np.asarray(samples_ms, dtype=float)
    return {
        P50_METRIC: float(np.percentile(array, 50)),
        P95_METRIC: float(np.percentile(array, 95)),
        P99_METRIC: float(np.percentile(array, 99)),
    }


def time_calls(predict_one: Callable[[int], Any], n: int, warmup: int) -> list[float]:
    """Time ``predict_one(i)`` for each of ``n`` requests, discarding ``warmup`` first.

    The warm-up matters more than it looks: the first calls pay lazy imports, one-off
    tokeniser setup and cold caches, and including them turns a p99 into a measurement of
    process start-up rather than of inference.
    """
    for i in range(warmup):
        predict_one(i % n)

    timings = []
    for i in range(n):
        started = time.perf_counter()
        predict_one(i)
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def artifact_mb(run_id: str, artifact_name: str = MODEL_ARTIFACT) -> float:
    """On-disk size of a run's logged model, in MB.

    The whole artifact directory - weights, tokeniser, the MLflow wrapper, the pinned
    requirements - because that is what a serving image has to carry, not just the
    weights file.
    """
    local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name))
    total = sum(
        (Path(root) / name).stat().st_size for root, _, files in os.walk(local) for name in files
    )
    return total / (1024 * 1024)


# --------------------------------------------------------------- per-tier timing


def bench_tier1(run_id: str, texts: list[str], keys: list[str], n: int, warmup: int) -> dict[str, float]:
    """TF-IDF + logistic regression: vectorise and predict in one sklearn pipeline."""
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT}")
    return _percentiles(time_calls(lambda i: model.predict([texts[i]]), n, warmup))


def bench_tier2(run_id: str, texts: list[str], keys: list[str], n: int, warmup: int) -> dict[str, float]:
    """MiniLM + LightGBM, measured on both feature-store paths.

    The headline p50/p95/p99 are the **cache-miss** path (encode live), because that is
    the honest worst case for text the system has never seen. The cache-hit p95 is
    reported alongside as the payoff for having a feature store at all.
    """
    # Import order is load-bearing, not stylistic. LightGBM must claim the OpenMP runtime
    # before torch does: torch-first segfaults the process the moment LightGBM loads, even
    # with OMP_NUM_THREADS=1. LightGBM-first plus the pinned thread count is the one
    # combination that survives (module docstring has the full finding).
    import lightgbm  # noqa: F401

    cfg = load_config("model_tier2")["embedding"]
    head = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT}")

    from sentence_transformers import SentenceTransformer

    from src.features.store import FeatureStore

    encoder = SentenceTransformer(cfg["model"], device="cpu")
    encoder.max_seq_length = int(cfg["max_seq_length"])
    store = FeatureStore(model=cfg["model"], dimension=int(cfg["dimension"]))

    def miss(i: int) -> Any:
        vector = encoder.encode(
            [texts[i]], normalize_embeddings=bool(cfg["normalize"]), show_progress_bar=False
        )
        return head.predict(np.asarray(vector))

    def hit(i: int) -> Any:
        vector = store.get(keys[i])
        if vector is None:  # in serving this falls through to the miss path
            return miss(i)
        return head.predict(vector.reshape(1, -1))

    # Confirm the hit path is actually hitting before timing it: a store that silently
    # missed would make the "cached" numbers a copy of the uncached ones.
    hits = sum(1 for key in keys[:n] if store.get(key) is not None)
    print(f"[bench] tier 2 feature-store hits: {hits}/{n}")

    results = _percentiles(time_calls(miss, n, warmup))
    results[CACHED_METRIC] = _percentiles(time_calls(hit, n, warmup))[P95_METRIC]
    return results


def bench_tier3(run_id: str, texts: list[str], keys: list[str], n: int, warmup: int) -> dict[str, float]:
    """Fine-tuned DistilRoBERTa on CPU - the tier the latency target is really about."""
    import torch

    torch.set_num_threads(1)
    pipeline = mlflow.transformers.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT}", device=-1)
    max_length = int(load_config("model_tier3")["max_length"])

    def predict(i: int) -> Any:
        return pipeline(texts[i], truncation=True, max_length=max_length)

    return _percentiles(time_calls(predict, n, warmup))


BENCHMARKS: dict[str, Callable[[str, list[str], list[str], int, int], dict[str, float]]] = {
    "1": bench_tier1,
    "2": bench_tier2,
    "3": bench_tier3,
}


# --------------------------------------------------------------------- sampling


def sample_requests(n: int, seed: int = 42) -> tuple[list[str], list[str]]:
    """Draw the request texts (and their feature-store keys) from the test split.

    Real reviews, not synthetic strings: length drives tokenisation and vectorisation
    cost, so a benchmark on toy inputs would understate every tier by a different amount.
    """
    test = load_split("test")
    sample = test.sample(n=min(n, len(test)), random_state=seed)
    return sample[MODEL_INPUT].tolist(), sample[TEXT_KEY].tolist()


def _measure_one(tier: str, run_id: str, n: int, warmup: int) -> dict[str, float]:
    """Time one tier in this process. Only ever called inside a pinned subprocess."""
    texts, keys = sample_requests(n)
    measured = BENCHMARKS[tier](run_id, texts, keys, len(texts), warmup)
    measured[SIZE_METRIC] = artifact_mb(run_id)
    return measured


def _measure_in_subprocess(tier: str, run_id: str, n: int, warmup: int) -> dict[str, float]:
    """Run ``_measure_one`` in a thread-pinned child process and read back its results."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"tier{tier}.json"
        subprocess.run(
            [
                sys.executable, "-m", "src.evaluate.benchmark",
                "--tier", tier, "--run-id", run_id,
                "--samples", str(n), "--warmup", str(warmup), "--out", str(out),
            ],
            check=True,
            env={**os.environ, **PINNED_ENV},
        )
        return json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ main


def benchmark(n: int = DEFAULT_SAMPLES, warmup: int = DEFAULT_WARMUP) -> dict[str, dict[str, float]]:
    """Benchmark the latest finished run of every tier and log the results back."""
    from src.evaluate.compare import fetch_runs

    runs = fetch_runs()
    if runs.empty:
        raise RuntimeError("no finished runs to benchmark - run `make train` first")

    client = mlflow.tracking.MlflowClient()
    results: dict[str, dict[str, float]] = {}

    for _, run in runs.sort_values("tags.tier").iterrows():
        tier, run_id = str(run["tags.tier"]), str(run["run_id"])
        if tier not in BENCHMARKS:
            print(f"[bench] tier {tier}: no benchmark defined, skipping")
            continue

        print(f"[bench] tier {tier} ({run_id[:8]}): {n} requests, batch=1, 1 thread")
        started = time.time()
        measured = _measure_in_subprocess(tier, run_id, n, warmup)

        for key, value in measured.items():
            client.log_metric(run_id, key, value)
        client.set_tag(run_id, "bench.samples", str(n))
        client.set_tag(run_id, "bench.threads", "1")
        client.set_tag(run_id, "bench.machine", f"{platform.machine()} {platform.system()}")

        results[tier] = measured
        print(
            f"[bench] tier {tier}: p50={measured[P50_METRIC]:.1f}ms "
            f"p95={measured[P95_METRIC]:.1f}ms p99={measured[P99_METRIC]:.1f}ms "
            f"size={measured[SIZE_METRIC]:.1f}MB ({time.time() - started:.0f}s)"
        )
        if CACHED_METRIC in measured:
            print(f"[bench] tier {tier}: p95 on a feature-store hit={measured[CACHED_METRIC]:.1f}ms")

    return results


def main() -> Any:
    parser = argparse.ArgumentParser(description="Per-tier serving latency and artifact size.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="requests per tier")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="untimed calls first")
    # The three below put this module in child mode: measure one tier, write JSON, exit.
    parser.add_argument("--tier", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    registry.setup()
    if args.tier:
        measured = _measure_one(args.tier, args.run_id, args.samples, args.warmup)
        Path(args.out).write_text(json.dumps(measured), encoding="utf-8")
        return measured
    return benchmark(n=args.samples, warmup=args.warmup)


if __name__ == "__main__":
    main()
