import pandas as pd
import pytest

from src.monitor import trigger

NO_ALERT = {"alert": False}
ALERT = {"alert": True}
PERF_OK = {"alert": False}
PERF_BAD = {"alert": True, "macro_f1_drop": 0.08}


@pytest.fixture
def window() -> pd.DataFrame:
    return pd.DataFrame({"Text": ["A perfectly fine review.", "Another one."], "Score": [4, 5]})


def test_validate_window_accepts_clean_data(window):
    ok, reason = trigger.validate_window(window)
    assert ok is True
    assert reason is None


def test_validate_window_rejects_invalid_score(window):
    bad = window.copy()
    bad.loc[0, "Score"] = 9
    ok, reason = trigger.validate_window(bad)
    assert ok is False
    assert reason is not None


def test_validate_window_rejects_blank_text(window):
    bad = window.copy()
    bad.loc[0, "Text"] = "   "
    ok, _ = trigger.validate_window(bad)
    assert ok is False


def test_watch_after_sustained_alerts(window):
    state = trigger.TriggerState()
    decisions = [
        trigger.evaluate_window(state, ALERT, NO_ALERT, PERF_OK, window) for _ in range(3)
    ]
    assert [d["tier"] for d in decisions] == ["none", "none", "watch"]


def test_watch_resets_on_a_clean_window(window):
    state = trigger.TriggerState()
    trigger.evaluate_window(state, ALERT, NO_ALERT, PERF_OK, window)
    trigger.evaluate_window(state, ALERT, NO_ALERT, PERF_OK, window)
    trigger.evaluate_window(state, NO_ALERT, NO_ALERT, PERF_OK, window)  # breaks the streak
    decision = trigger.evaluate_window(state, ALERT, NO_ALERT, PERF_OK, window)
    assert decision["tier"] == "none"  # only 1 consecutive alert again, not 3


def test_candidate_without_prior_watch(window):
    state = trigger.TriggerState()
    decision = trigger.evaluate_window(state, NO_ALERT, NO_ALERT, PERF_BAD, window)
    # No prior fire, so cooldown is trivially elapsed and the window is valid -> fires.
    assert decision["tier"] == "fire"
    assert decision["candidate"] is True


def test_fire_is_suppressed_during_cooldown(window):
    state = trigger.TriggerState()
    first = trigger.evaluate_window(state, NO_ALERT, NO_ALERT, PERF_BAD, window)
    second = trigger.evaluate_window(state, NO_ALERT, NO_ALERT, PERF_BAD, window)
    assert first["tier"] == "fire"
    assert second["tier"] == "candidate"
    assert any("cooldown" in r for r in second["reasons"])


def test_fire_is_suppressed_by_failed_schema_validation():
    state = trigger.TriggerState()
    bad_window = pd.DataFrame({"Text": ["fine"], "Score": [9]})
    decision = trigger.evaluate_window(state, NO_ALERT, NO_ALERT, PERF_BAD, bad_window)
    assert decision["candidate"] is True
    assert decision["fire"] is False
    assert any("schema validation" in r for r in decision["reasons"])


def test_promote_requires_the_minimum_gain():
    assert trigger.evaluate_promotion(0.62, 0.6001)["promote"] is True
    assert trigger.evaluate_promotion(0.605, 0.6001)["promote"] is False


def test_promote_blocks_on_a_per_slice_regression():
    result = trigger.evaluate_promotion(
        0.65, 0.6001, challenger_slice_f1={1: 0.3, 2: 0.4}, champion_slice_f1={1: 0.5, 2: 0.3}
    )
    assert result["promote"] is False
    assert 1 in result["regressed_slices"]
