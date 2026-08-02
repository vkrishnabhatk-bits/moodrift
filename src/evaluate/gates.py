"""The promotion gates: one definition, used by both the report and the registry.

A model is promoted when it clears the thresholds in ``conf/evaluation.yaml``. Those
checks are written once, here, because the alternative - the comparison report deciding
one thing and the registration script another - is the kind of inconsistency nobody
notices until a model is already serving.

Gates are evaluated against MLflow run metrics, so a gate can only test something that was
actually measured and logged. A gate whose metric is missing is skipped rather than
silently passed: the caller sees a shorter list, and ``make bench`` fills it in.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import load_config


@dataclass(frozen=True)
class Gate:
    """One threshold, its measured value, and whether the model cleared it."""

    label: str
    passed: bool
    value: str


def evaluate(runs: pd.DataFrame, champion: pd.Series) -> list[Gate]:
    """Every gate that can be evaluated for ``champion`` given the logged runs."""
    cfg = load_config("evaluation")
    targets, baseline_cfg = cfg["targets"], cfg["baseline"]

    f1 = float(champion["metrics.test_macro_f1"])
    mae = float(champion["metrics.test_macro_mae"])
    gates = [
        Gate(f"Macro-F1 >= {targets['macro_f1_min']}", f1 >= float(targets["macro_f1_min"]), f"{f1:.4f}"),
        Gate(f"Macro-MAE <= {targets['macro_mae_max']}", mae <= float(targets["macro_mae_max"]), f"{mae:.4f}"),
    ]

    baseline_tier = str(baseline_cfg["tier"])
    baseline = runs[runs["tags.tier"] == baseline_tier]
    min_gain = float(baseline_cfg["min_macro_f1_gain"])
    if not baseline.empty and str(champion["tags.tier"]) != baseline_tier:
        gain = f1 - float(baseline.iloc[0]["metrics.test_macro_f1"])
        gates.append(
            Gate(
                f"Beats the tier-{baseline_tier} baseline by >= {min_gain:.2f} macro-F1",
                gain >= min_gain,
                f"+{gain:.4f}",
            )
        )

    p95 = champion.get("metrics.serve_latency_p95_ms")
    if pd.notna(p95):
        limit = float(targets["latency_p95_ms_max"])
        gates.append(
            Gate(
                f"p95 latency < {limit:.0f} ms (batch=1, 1 thread)",
                float(p95) < limit,
                f"{float(p95):.1f} ms",
            )
        )
    return gates


def all_passed(gates: list[Gate]) -> bool:
    return all(gate.passed for gate in gates)
