import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve
from src.monitor import detectors


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 1000))
    assert detectors.psi(reference, reference) < 1e-6


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 1000))
    shifted = pd.Series(rng.normal(5, 1, 1000))
    assert detectors.psi(reference, shifted) > 1.0


def test_input_drift_is_quiet_on_a_null_case():
    """Two random halves of the frozen reference window should not alert each other."""
    ref = pd.read_parquet(resolve(load_config("monitor")["reference"]["path"]))
    half_a = ref.sample(n=2500, random_state=1)
    half_b = ref.drop(half_a.index)
    result = detectors.input_drift(half_a, half_b)
    assert result["alert"] is False


def test_input_drift_alerts_on_a_real_shift():
    """Text stretched to 5x length should blow past the char_count/token_count PSI bar."""
    ref = pd.read_parquet(resolve(load_config("monitor")["reference"]["path"])).sample(n=500, random_state=1)
    shifted = ref.copy()
    shifted["Text"] = shifted["Text"] * 5
    result = detectors.input_drift(ref, shifted)
    assert result["alert"] is True
    by_feature = {f["feature"]: f for f in result["features"]}
    assert bool(by_feature["char_count"]["alert"]) is True


def test_reference_vocabulary_contains_common_words():
    vocab = detectors.reference_vocabulary()
    assert "the" in vocab
    assert "and" in vocab
    assert len(vocab) > 100


def test_concept_drift_is_near_chance_on_a_null_case():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=(300, 16))
    current = rng.normal(0, 1, size=(300, 16))
    result = detectors.concept_drift(reference, current)
    assert result["auc"] == pytest.approx(0.5, abs=0.15)
    assert result["alert"] is False


def test_concept_drift_detects_a_real_separation():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=(300, 16))
    current = rng.normal(4, 1, size=(300, 16))  # clearly separable cluster
    result = detectors.concept_drift(reference, current)
    assert result["auc"] > 0.9
    assert result["alert"] is True


def test_performance_drift_below_min_samples_is_not_evaluated():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    result = detectors.performance_drift(y_true, y_pred)
    assert result["evaluated"] is False
    assert result["alert"] is False


def test_performance_drift_matches_baseline_when_perfect():
    y_true = np.tile([1, 2, 3, 4, 5], 200)
    y_pred = y_true.copy()
    result = detectors.performance_drift(y_true, y_pred)
    assert result["evaluated"] is True
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["alert"] is False


def test_performance_drift_detects_a_real_degradation():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 6, size=600)
    y_pred = np.full_like(y_true, 3)  # a model that always predicts "3 stars"
    result = detectors.performance_drift(y_true, y_pred)
    assert result["evaluated"] is True
    assert result["alert"] is True
