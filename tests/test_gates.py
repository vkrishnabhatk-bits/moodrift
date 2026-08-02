"""The promotion gates decide what is allowed to serve, so they get tested directly.

Every case here is built from a synthetic runs frame rather than the live tracking store:
a gate test that depends on today's MLflow contents would pass or fail for reasons that
have nothing to do with the gate logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import gates

TARGETS = load_config("evaluation")["targets"]
BASELINE = load_config("evaluation")["baseline"]


def _runs(champion_f1: float = 0.60, champion_mae: float = 0.42, p95: float | None = 40.0,
          baseline_f1: float = 0.51) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        [
            {
                "run_id": "baseline",
                "tags.tier": "1",
                "metrics.test_macro_f1": baseline_f1,
                "metrics.test_macro_mae": 0.63,
                "metrics.serve_latency_p95_ms": 2.0,
            },
            {
                "run_id": "champion",
                "tags.tier": "3",
                "metrics.test_macro_f1": champion_f1,
                "metrics.test_macro_mae": champion_mae,
                "metrics.serve_latency_p95_ms": p95,
            },
        ]
    )
    return frame, frame.iloc[1]


def _by_prefix(checks: list[gates.Gate], prefix: str) -> gates.Gate:
    return next(gate for gate in checks if gate.label.startswith(prefix))


class TestEvaluate:
    def test_a_good_model_passes_every_gate(self):
        runs, champion = _runs()
        checks = gates.evaluate(runs, champion)
        assert gates.all_passed(checks)
        assert len(checks) == 4  # f1, mae, baseline gain, latency

    def test_macro_f1_below_target_fails(self):
        runs, champion = _runs(champion_f1=float(TARGETS["macro_f1_min"]) - 0.01)
        assert not _by_prefix(gates.evaluate(runs, champion), "Macro-F1").passed

    def test_macro_f1_exactly_at_target_passes(self):
        # The gate is >=, so the boundary is inside. Pinned because an off-by-one here
        # silently rejects a model that met the published bar.
        runs, champion = _runs(champion_f1=float(TARGETS["macro_f1_min"]))
        assert _by_prefix(gates.evaluate(runs, champion), "Macro-F1").passed

    def test_macro_mae_above_target_fails(self):
        runs, champion = _runs(champion_mae=float(TARGETS["macro_mae_max"]) + 0.01)
        assert not _by_prefix(gates.evaluate(runs, champion), "Macro-MAE").passed

    def test_too_small_a_gain_over_the_baseline_fails(self):
        gain = float(BASELINE["min_macro_f1_gain"])
        runs, champion = _runs(champion_f1=0.60, baseline_f1=0.60 - gain / 2)
        assert not _by_prefix(gates.evaluate(runs, champion), "Beats the tier-1").passed

    def test_slow_model_fails_the_latency_gate(self):
        runs, champion = _runs(p95=float(TARGETS["latency_p95_ms_max"]) + 1)
        assert not _by_prefix(gates.evaluate(runs, champion), "p95 latency").passed
        assert not gates.all_passed(gates.evaluate(runs, champion))

    def test_unmeasured_latency_is_skipped_not_passed(self):
        # A gate whose metric was never logged must not count as satisfied - `make bench`
        # has simply not run yet, and silently passing would promote an unmeasured model.
        runs, champion = _runs(p95=None)
        checks = gates.evaluate(runs, champion)
        assert not any(gate.label.startswith("p95 latency") for gate in checks)
        assert len(checks) == 3

    def test_the_baseline_is_not_compared_against_itself(self):
        runs, _ = _runs()
        checks = gates.evaluate(runs, runs.iloc[0])
        assert not any(gate.label.startswith("Beats the tier-1") for gate in checks)


class TestAllPassed:
    @pytest.mark.parametrize(
        ("flags", "expected"),
        [((True, True), True), ((True, False), False), ((False, False), False), ((), True)],
    )
    def test_all_passed(self, flags, expected):
        checks = [gates.Gate(f"gate {i}", flag, "-") for i, flag in enumerate(flags)]
        assert gates.all_passed(checks) is expected
