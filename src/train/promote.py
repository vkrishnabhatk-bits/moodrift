"""Register the winning run in the MLflow Model Registry, behind the promotion gates.

Registration is not "save the best number". It is a decision that must be defensible
later, so this script does three things a manual `mlflow.register_model` call would not:

1. **It refuses to promote a model that fails a gate.** The thresholds live in
   ``conf/evaluation.yaml`` and are evaluated by ``src/evaluate/gates.py`` - the same
   code that prints the gate table in the comparison report, so the two can never
   disagree. ``--force`` exists for the case where you have a reason, and it records
   that the promotion was forced rather than hiding it.
2. **It is idempotent.** A run that is already registered is not registered again; the
   aliases simply move. Re-running this is safe and does not litter the registry with
   identical versions.
3. **It writes down why.** The version description carries the metrics, the serving cost,
   the run ID and the git SHA, so a reviewer opening the registry six weeks later can see
   the argument without reconstructing it from the tracking store.

Aliases, not stages (ADR-0004). This sets ``@candidate`` and ``@champion``; ``@production``
is deliberately left alone - it moves in Week 3, once the API smoke and load tests pass.

Run with ``python -m src.train.promote`` (``make register``).
"""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
import pandas as pd

from src.evaluate import gates
from src.train import registry

MODEL_ARTIFACT = "model"
CANDIDATE, CHAMPION = "candidate", "champion"


def _existing_version(client: mlflow.tracking.MlflowClient, run_id: str) -> Any | None:
    """The registered version for ``run_id``, if this run was already registered."""
    versions = client.search_model_versions(f"name='{registry.REGISTERED_MODEL}'")
    return next((v for v in versions if v.run_id == run_id), None)


def _description(champion: pd.Series, checks: list[gates.Gate], forced: bool) -> str:
    """The argument for this model, stored next to the model itself."""
    tier = str(champion["tags.tier"])
    lines = [
        f"Tier {tier} ({champion.get('tags.mlflow.runName', '?')}) - champion as of "
        f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.",
        "",
        f"macro-F1 {float(champion['metrics.test_macro_f1']):.4f}, "
        f"macro-MAE {float(champion['metrics.test_macro_mae']):.4f} on the held-out test split.",
    ]
    p95 = champion.get("metrics.serve_latency_p95_ms")
    size = champion.get("metrics.serve_artifact_mb")
    if pd.notna(p95) and pd.notna(size):
        lines.append(
            f"Serving cost: p95 {float(p95):.1f} ms at batch=1 on one CPU thread, "
            f"{float(size):.1f} MB on disk."
        )
    lines += [
        "",
        "Gates: " + "; ".join(f"{g.label} = {g.value} ({'PASS' if g.passed else 'FAIL'})" for g in checks),
        "",
        f"Run: {champion['run_id']}",
        f"Git SHA: {champion.get('tags.git_sha', 'unknown')}",
    ]
    if forced:
        lines += ["", "PROMOTED WITH --force DESPITE A FAILING GATE."]
    return "\n".join(lines)


def promote(force: bool = False) -> dict[str, Any]:
    """Register the highest-macro-F1 run and point @candidate and @champion at it."""
    from src.evaluate.compare import fetch_runs

    runs = fetch_runs()
    if runs.empty:
        raise RuntimeError("no finished runs to promote - run `make train` first")

    champion = runs.loc[runs["metrics.test_macro_f1"].idxmax()]
    run_id, tier = str(champion["run_id"]), str(champion["tags.tier"])
    checks = gates.evaluate(runs, champion)

    print(f"[promote] candidate: tier {tier}, run {run_id[:8]}")
    for gate in checks:
        print(f"[promote]   {'PASS' if gate.passed else 'FAIL'}  {gate.label} = {gate.value}")

    if not gates.all_passed(checks) and not force:
        failed = [gate.label for gate in checks if not gate.passed]
        raise SystemExit(
            "[promote] refusing to promote: " + "; ".join(failed) + "\n"
            "[promote] fix the model, or re-run with --force if you have a reason "
            "(it will be recorded on the version)."
        )

    client = mlflow.tracking.MlflowClient()
    registry.setup()

    existing = _existing_version(client, run_id)
    if existing is not None:
        version = existing.version
        print(f"[promote] run already registered as v{version} - moving aliases only")
    else:
        version = registry.register(run_id, MODEL_ARTIFACT).version
        print(f"[promote] registered {registry.REGISTERED_MODEL} v{version}")

    client.update_model_version(
        registry.REGISTERED_MODEL, version, description=_description(champion, checks, force)
    )
    for key, value in {
        "tier": tier,
        "run_id": run_id,
        "git_sha": str(champion.get("tags.git_sha", "unknown")),
        "test_macro_f1": f"{float(champion['metrics.test_macro_f1']):.4f}",
        "promoted_with_force": str(force).lower(),
    }.items():
        client.set_model_version_tag(registry.REGISTERED_MODEL, version, key, value)

    for alias in (CANDIDATE, CHAMPION):
        registry.set_alias(alias, version)
        print(f"[promote] @{alias} -> v{version}")

    print(
        "[promote] @production deliberately unchanged - it moves in Week 3, "
        "after the API smoke and load tests."
    )
    return {"version": version, "run_id": run_id, "tier": tier}


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Register the champion model.")
    parser.add_argument(
        "--force", action="store_true", help="promote even if a gate fails (recorded on the version)"
    )
    args = parser.parse_args()
    return promote(force=args.force)


if __name__ == "__main__":
    main()
