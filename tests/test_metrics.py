"""Tests for the evaluation metrics.

The macro-MAE tests matter most: it is a custom metric, so nothing external would catch
it being subtly wrong, and every model-selection decision downstream depends on it.
"""

from __future__ import annotations

import numpy as np

from src.evaluate.metrics import bootstrap_ci, compute_metrics, macro_mae, worst_misclassifications


class TestMacroMae:
    def test_perfect_predictions_score_zero(self):
        y = np.array([1, 2, 3, 4, 5])
        assert macro_mae(y, y) == 0.0

    def test_distance_is_measured_in_stars(self):
        # Every prediction is exactly one star low.
        y_true = np.array([2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4])
        assert macro_mae(y_true, y_pred) == 1.0

    def test_averages_over_classes_not_rows(self):
        """The reason this metric exists rather than plain MAE.

        One rare class is wrong by 4 stars; a large majority class is perfect. Row-wise
        MAE would call this excellent, hiding a total failure on the minority class.
        """
        y_true = np.array([5] * 99 + [1])
        y_pred = np.array([5] * 99 + [5])

        row_mae = np.abs(y_pred - y_true).mean()
        assert row_mae < 0.05  # looks great

        # Macro-MAE refuses to hide it: class 1 is 4 stars off, class 5 is perfect.
        assert macro_mae(y_true, y_pred) == 2.0

    def test_absent_classes_are_skipped(self):
        y_true = np.array([1, 1])
        y_pred = np.array([1, 1])
        assert macro_mae(y_true, y_pred) == 0.0


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3, 4, 5] * 4)
        m = compute_metrics(y, y)
        assert m["macro_f1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["macro_mae"] == 0.0
        assert m["adjacent_accuracy"] == 1.0

    def test_majority_predictor_scores_poor_macro_f1(self):
        """The imbalance trap the metric choice exists to expose."""
        y_true = np.array([5] * 90 + [1] * 10)
        y_pred = np.array([5] * 100)
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.90  # flattering
        assert m["macro_f1"] < 0.25  # honest

    def test_adjacent_accuracy_credits_near_misses(self):
        y_true = np.array([3, 3, 3, 3])
        y_pred = np.array([2, 3, 4, 1])
        # three within one star, one two stars off
        assert compute_metrics(y_true, y_pred)["adjacent_accuracy"] == 0.75

    def test_per_class_keys_present(self):
        y = np.array([1, 2, 3, 4, 5])
        m = compute_metrics(y, y)
        for star in range(1, 6):
            assert f"f1_class_{star}" in m
            assert f"support_class_{star}" in m


class TestBootstrapCi:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, size=400)
        y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(1, 6, size=400))

        point = compute_metrics(y_true, y_pred)["macro_f1"]
        low, high = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=0)
        assert low <= point <= high

    def test_is_deterministic_given_a_seed(self):
        y_true = np.array([1, 2, 3, 4, 5] * 20)
        y_pred = np.array([1, 2, 3, 4, 4] * 20)
        assert bootstrap_ci(y_true, y_pred, n_resamples=50, seed=7) == bootstrap_ci(
            y_true, y_pred, n_resamples=50, seed=7
        )


class TestWorstMisclassifications:
    def test_orders_by_severity_and_excludes_correct_rows(self):
        texts = ["a", "b", "c"]
        y_true = np.array([5, 5, 3])
        y_pred = np.array([1, 4, 3])  # 4 stars off, 1 star off, correct
        worst = worst_misclassifications(texts, y_true, y_pred, np.array([0.9, 0.9, 0.9]), n=5)

        assert len(worst) == 2  # the correct row is dropped
        assert worst[0]["stars_off"] == 4
