"""Four-tier retraining-trigger policy: WATCH -> CANDIDATE -> FIRE -> PROMOTE.

Design: `docs/decisions/ADR-0007-retraining-trigger-design.md`, `docs/drift_design.md`.

Rule-based, not learned - a trigger policy is the one decision in this system that most
needs to be explainable, and four rules with thresholds anyone can read serve that better
than a model that would itself need drift-incident training data this project doesn't
have. Nothing here trains anything: this module turns detector output into a decision and
a reason a human can read; a real retraining job would be the thing that consumes a FIRE
event, kept deliberately separate so the decision stays auditable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import pandera as pa

from src.config import load_config

Tier = Literal["none", "watch", "candidate", "fire"]

# The FIRE gate's data-validation check: a minimal schema for a *monitoring window*, not
# the full ingest schema (src.ingest.schema.build_schema) - that one's column contract
# (ProductId, Helpfulness*, ...) describes the raw CSV, which a window of already-processed
# predictions was never going to match.
_WINDOW_SCHEMA = pa.DataFrameSchema(
    {
        "Text": pa.Column(
            str,
            checks=pa.Check(lambda s: s.str.strip().str.len() > 0, element_wise=False, name="non_blank"),
            nullable=False,
        ),
        "Score": pa.Column(int, checks=pa.Check.isin([1, 2, 3, 4, 5]), nullable=False, coerce=True),
    },
    strict=False,
)


def validate_window(window: pd.DataFrame) -> tuple[bool, str | None]:
    """True, None on a clean window; False, a short failure summary otherwise."""
    try:
        _WINDOW_SCHEMA.validate(window, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as exc:
        return False, str(exc.failure_cases.head(5).to_dict("records"))


@dataclass
class TriggerState:
    """Rolling state across windows - what makes WATCH's "sustained N windows" and
    FIRE's cooldown possible without re-reading every prior window's raw detector output.
    One `TriggerState` per monitored stream (e.g. one per drift-simulation scenario).
    """

    consecutive_statistical_alerts: int = 0
    windows_since_last_fire: int | None = None  # None = never fired
    history: list[dict[str, Any]] = field(default_factory=list)


def evaluate_window(
    state: TriggerState,
    input_drift: dict[str, Any],
    concept_drift: dict[str, Any],
    performance_drift: dict[str, Any],
    window: pd.DataFrame,
) -> dict[str, Any]:
    """One window's decision. Mutates `state` in place and returns the decision, so a
    caller can fold the next window's detector output straight back in.
    """
    cfg = load_config("monitor")["trigger"]

    statistical_alert = bool(input_drift["alert"] or concept_drift["alert"])
    state.consecutive_statistical_alerts = (
        state.consecutive_statistical_alerts + 1 if statistical_alert else 0
    )
    watch = state.consecutive_statistical_alerts >= int(cfg["watch"]["sustained_windows"])

    candidate = bool(performance_drift.get("alert"))

    cooldown_windows = int(cfg["fire"]["cooldown_days"])
    cooldown_elapsed = (
        state.windows_since_last_fire is None or state.windows_since_last_fire >= cooldown_windows
    )
    schema_ok, schema_reason = validate_window(window) if candidate else (True, None)
    fire = candidate and cooldown_elapsed and schema_ok

    tier: Tier = "fire" if fire else "candidate" if candidate else "watch" if watch else "none"

    reasons: list[str] = []
    if watch:
        reasons.append(f"input/concept drift sustained {state.consecutive_statistical_alerts} windows")
    if candidate:
        reasons.append(f"performance drift: macro_f1_drop={performance_drift.get('macro_f1_drop', 0):.4f}")
        if not fire:
            if not cooldown_elapsed:
                reasons.append(
                    f"FIRE suppressed: cooldown active ({state.windows_since_last_fire} "
                    f"windows since last fire, needs {cooldown_windows})"
                )
            elif not schema_ok:
                reasons.append(f"FIRE suppressed: window failed schema validation ({schema_reason})")

    decision = {
        "tier": tier,
        "watch": watch,
        "candidate": candidate,
        "fire": fire,
        "reasons": reasons,
        "input_drift_alert": bool(input_drift["alert"]),
        "concept_drift_alert": bool(concept_drift["alert"]),
        "performance_drift_alert": bool(performance_drift.get("alert")),
    }

    state.windows_since_last_fire = 0 if fire else (
        state.windows_since_last_fire + 1 if state.windows_since_last_fire is not None else None
    )
    state.history.append(decision)
    return decision


def evaluate_promotion(
    challenger_macro_f1: float,
    champion_macro_f1: float,
    challenger_slice_f1: dict[int, float] | None = None,
    champion_slice_f1: dict[int, float] | None = None,
) -> dict[str, Any]:
    """The PROMOTE gate: a challenger earns `@production` only by beating the champion by
    the configured margin, with no per-slice regression. Called after a FIRE-triggered
    retrain produces a challenger - this project's demo simulates the gate's logic on two
    scored models, not a live retraining job (see ADR-0007's scope note).
    """
    cfg = load_config("monitor")["trigger"]["promote"]
    gain = challenger_macro_f1 - champion_macro_f1
    beats_champion = gain >= float(cfg["min_macro_f1_gain"])

    regressed: dict[int, float] = {}
    if not bool(cfg["allow_slice_regression"]) and challenger_slice_f1 and champion_slice_f1:
        regressed = {
            label: challenger_slice_f1[label] - champion_slice_f1[label]
            for label in champion_slice_f1
            if challenger_slice_f1.get(label, 0.0) < champion_slice_f1[label]
        }

    promote = beats_champion and not regressed
    return {
        "promote": promote,
        "macro_f1_gain": gain,
        "regressed_slices": regressed,
    }
