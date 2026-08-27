"""Re-run a logged experiment and prove it lands in the same place.

``make reproduce RUN_ID=<id>`` is the claim "this project is reproducible" turned into a
command that either passes or fails. It does four things in order, and stops at the first
one that is not satisfied:

1. **Identify the run.** Tier, git SHA, data hashes and the recorded metric all come from
   the MLflow run - nothing is passed in by hand.
2. **Check the code.** If HEAD is not the commit that produced the run, say so loudly.
   Re-running different code and matching the old number would be luck, not reproduction,
   so this is a warning the operator has to see, not a silent detail.
3. **Check the data.** Every split's content hash must match what the run recorded. This
   is the half of reproducibility a config file cannot give you: the same config over
   different data is a different experiment. ``--restore-data`` will try to put the
   working tree back on the run's data version via ``dvc checkout``.
4. **Re-train and compare.** The tier trains again from the current config, and the new
   metric is compared against the logged one within the tolerance in
   ``conf/evaluation.yaml``.

**Tolerance is not a loophole.** CPU tiers are checked at exactly 0.0 - they either
reproduce bit-for-bit or they do not. Tier 3 trained on Apple's MPS backend, whose
floating-point ops are not bit-deterministic: two runs from an identical seed produced
0.6017 and 0.6001. Asserting equality there would assert something false, so that tier
gets a documented band and the report says exactly why.

Run with ``python -m src.train.reproduce <run_id>`` (``make reproduce RUN_ID=<id>``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import mlflow

from src.config import REPO_ROOT, load_config, resolve
from src.provenance import file_digest, git_sha
from src.train import registry

# Same modules `make train` uses - reproduction runs the real training path, not a copy.
TIER_MODULES = {
    "1": "src.train.tier1",
    "2": "src.train.tier2",
    "3": "src.train.tier3",
}
SPLITS = ("train", "val", "test")


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a config the way ``registry.log_flat_params`` logged it, for comparison."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def check_code(run_sha: str) -> list[str]:
    """Compare HEAD against the commit that produced the run."""
    current = git_sha()
    if current == run_sha:
        return []
    if current.endswith("-dirty") and current[: -len("-dirty")] == run_sha:
        return [f"working tree is dirty at the run's commit {run_sha[:8]} - uncommitted changes"]
    return [f"HEAD is {current[:8]}, the run was produced by {run_sha[:8]}"]


def check_data(run: Any) -> list[str]:
    """Compare each split's content hash against the hash the run recorded."""
    paths = load_config("data")["paths"]
    problems = []
    for split in SPLITS:
        recorded = run.data.tags.get(f"data.{split}.sha256")
        if recorded is None:
            problems.append(f"{split}: the run recorded no data hash")
            continue
        path = resolve(paths[split])
        if not path.exists():
            problems.append(f"{split}: {path} is missing - run `make data`")
            continue
        actual = file_digest(path)[:16]
        status = "match" if actual == recorded else "DIFFERENT"
        print(f"[reproduce]   {split:<6} {recorded} vs {actual}  {status}")
        if actual != recorded:
            problems.append(f"{split}: data has changed since the run ({recorded} -> {actual})")
    return problems


def check_config(tier: str, run: Any) -> list[str]:
    """Compare the current tier config against the params the run logged.

    A drifted config is the most common reason a re-run does not match, and it is
    invisible unless you look - the training script happily uses whatever is in
    ``conf/`` today.
    """
    current = _flatten(load_config(f"model_tier{tier}"))
    problems = []
    for key, value in current.items():
        logged = run.data.params.get(key)
        if logged is None:
            problems.append(f"{key}: not logged by the run (config gained a key)")
        elif logged != value:
            problems.append(f"{key}: run used {logged!r}, current config says {value!r}")
    return problems


def restore_data(run_sha: str) -> None:
    """Best-effort restore of the run's data version: its dvc.lock, then ``dvc checkout``.

    Only ``dvc.lock`` is touched, and only from the run's own commit. The DVC cache is
    local, so this succeeds when the version was built on this machine and fails clearly
    when it was not - rather than pretending to have restored something.

    ``--force`` is required: plain ``dvc checkout`` refuses whenever a DVC-tracked
    output has *any* local drift (a corrupted split, or the feature store's legitimate
    online-write accretions from serving) - which is exactly the situation this function
    exists to fix. Found by executing this path for the first time (it had never run
    before this): without --force it raised ``CalledProcessError`` and restored nothing,
    so the whole function silently didn't work despite reading and type-checking fine.
    """
    print(f"[reproduce] restoring data version from {run_sha[:8]}")
    print(
        "[reproduce] WARNING: overwriting local drift in DVC-tracked outputs "
        "(e.g. feature_store/features.db's online-write rows) with the run's version"
    )
    subprocess.run(["git", "checkout", run_sha, "--", "dvc.lock"], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "-m", "dvc", "checkout", "--force"], cwd=REPO_ROOT, check=True)


def reproduce(run_id: str, restore: bool = False, tolerance: float | None = None) -> dict[str, Any]:
    """Re-run ``run_id`` and assert the metric matches within tolerance."""
    registry.setup()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    tier = run.data.tags.get("tier")
    if tier not in TIER_MODULES:
        raise SystemExit(f"[reproduce] run {run_id[:8]} has no known tier tag (got {tier!r})")

    cfg = load_config("evaluation")["reproduce"]
    metric_key = str(cfg["metric"])
    if tolerance is None:
        tolerance = float(cfg["tolerance"].get(tier, cfg["tolerance"]["default"]))
    original = run.data.metrics.get(metric_key)
    if original is None:
        raise SystemExit(f"[reproduce] run {run_id[:8]} never logged {metric_key}")

    run_sha = run.data.tags.get("git_sha", "unknown").removesuffix("-dirty")
    print(f"[reproduce] run {run_id[:8]}: tier {tier}, {metric_key}={original:.4f}")
    print(f"[reproduce] recorded git SHA: {run_sha[:8]}, tolerance: {tolerance}")

    for warning in check_code(run_sha):
        print(f"[reproduce] WARNING: {warning}")

    print("[reproduce] data hashes (recorded vs on disk):")
    if restore:
        restore_data(run_sha)
    data_problems = check_data(run)
    if data_problems:
        for problem in data_problems:
            print(f"[reproduce] ERROR: {problem}")
        raise SystemExit(
            "[reproduce] data does not match the run. Re-run with --restore-data, or "
            f"`git checkout {run_sha[:8]} -- dvc.lock && dvc checkout` by hand."
        )

    config_problems = check_config(tier, run)
    if config_problems:
        for problem in config_problems:
            print(f"[reproduce] ERROR: {problem}")
        raise SystemExit(
            f"[reproduce] conf/model_tier{tier}.yaml has changed since the run. Check it out "
            f"from {run_sha[:8]} before reproducing."
        )
    print(f"[reproduce] config matches the run's logged params ({len(_flatten(load_config(f'model_tier{tier}')))} keys)")

    print(f"[reproduce] retraining tier {tier} - this runs the real training path")
    module = __import__(TIER_MODULES[tier], fromlist=["train"])
    results = module.train()

    new_run_id = results["run_id"]
    client.set_tag(new_run_id, "reproduces", run_id)
    achieved = float(results["test"][metric_key.removeprefix("test_")])
    delta = abs(achieved - original)
    passed = delta <= tolerance

    print(
        f"[reproduce] {metric_key}: original {original:.6f}, reproduced {achieved:.6f}, "
        f"delta {delta:.6f} (tolerance {tolerance})"
    )
    print(f"[reproduce] {'PASS' if passed else 'FAIL'} - new run {new_run_id[:8]}")
    if not passed:
        raise SystemExit(1)
    return {
        "run_id": run_id,
        "new_run_id": new_run_id,
        "tier": tier,
        "original": original,
        "reproduced": achieved,
        "delta": delta,
        "tolerance": tolerance,
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Re-run a logged experiment and check the metric.")
    parser.add_argument("run_id", help="MLflow run ID to reproduce")
    parser.add_argument(
        "--restore-data",
        action="store_true",
        help="check out the run's dvc.lock and `dvc checkout` before comparing hashes",
    )
    parser.add_argument(
        "--tolerance", type=float, default=None, help="override the per-tier tolerance"
    )
    args = parser.parse_args()
    return reproduce(args.run_id, restore=args.restore_data, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
