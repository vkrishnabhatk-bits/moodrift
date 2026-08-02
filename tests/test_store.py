"""Tests for the feature store.

The store's value is a promise: a given piece of text yields the same vector whether it
arrives through the batch pipeline or an HTTP request. These tests hold that promise to
account - including the float32 round-trip, where a silent dtype change would corrupt
every cached vector at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.store import FeatureStore
from src.provenance import text_key


@pytest.fixture
def store(tmp_path):
    return FeatureStore(path=tmp_path / "features.db", model="test-model", dimension=4)


class TestRoundTrip:
    def test_vector_survives_exactly(self, store):
        vec = np.array([[0.1, -0.2, 0.3, 0.4]], dtype=np.float32)
        store.write_many(["k1"], vec)
        np.testing.assert_array_equal(store.get("k1"), vec[0])

    def test_missing_key_returns_none(self, store):
        assert store.get("absent") is None

    def test_read_many_returns_only_what_exists(self, store):
        store.write_many(["a", "b"], np.zeros((2, 4), dtype=np.float32))
        found = store.read_many(["a", "b", "missing"])
        assert set(found) == {"a", "b"}

    def test_survives_reopening(self, tmp_path):
        """A cache that empties on restart is not a feature store."""
        path = tmp_path / "features.db"
        FeatureStore(path=path, dimension=4).write_many(["k"], np.ones((1, 4), dtype=np.float32))
        assert FeatureStore(path=path, dimension=4).get("k") is not None


class TestUpsert:
    def test_rewriting_a_key_replaces_rather_than_duplicates(self, store):
        store.write_many(["k"], np.zeros((1, 4), dtype=np.float32))
        store.write_many(["k"], np.ones((1, 4), dtype=np.float32))
        assert store.stats()["rows"] == 1
        np.testing.assert_array_equal(store.get("k"), np.ones(4, dtype=np.float32))

    def test_length_mismatch_is_rejected(self, store):
        with pytest.raises(ValueError, match="length mismatch"):
            store.write_many(["a", "b"], np.zeros((1, 4), dtype=np.float32))


class TestMatrix:
    def test_row_order_follows_requested_keys(self, store):
        store.write_many(
            ["a", "b"], np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        )
        matrix, missing = store.matrix(["b", "a"])
        assert not missing
        assert matrix[0][0] == 2.0
        assert matrix[1][0] == 1.0

    def test_missing_keys_are_reported_not_hidden(self, store):
        """Zero-filling a miss without reporting it would train on empty features."""
        store.write_many(["a"], np.ones((1, 4), dtype=np.float32))
        matrix, missing = store.matrix(["a", "ghost"])
        assert missing == ["ghost"]
        assert matrix.shape == (2, 4)

    def test_handles_batches_above_the_sqlite_variable_limit(self, store):
        keys = [f"k{i}" for i in range(2000)]
        store.write_many(keys, np.ones((2000, 4), dtype=np.float32))
        matrix, missing = store.matrix(keys)
        assert not missing
        assert matrix.shape == (2000, 4)


class TestProvenance:
    def test_source_is_tracked_per_row(self, store):
        """Batch vs online writes must stay distinguishable for the Week 3 read path."""
        store.write_many(["a"], np.zeros((1, 4), dtype=np.float32), source="batch")
        store.write_many(["b"], np.zeros((1, 4), dtype=np.float32), source="online")
        assert store.stats()["by_source"] == {"batch": 1, "online": 1}

    def test_metadata_round_trips(self, store):
        store.set_metadata(model="m", dimension=4)
        assert store.stats()["metadata"]["model"] == "m"


class TestKeying:
    def test_identical_text_yields_identical_key(self):
        assert text_key("Great product") == text_key("Great product")

    def test_different_text_yields_different_key(self):
        assert text_key("Great product") != text_key("Terrible product")

    def test_key_is_the_bridge_between_batch_and_online(self, store):
        """The end-to-end promise, in one test."""
        text = "Arrived quickly and tastes great"
        batch_vector = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        store.write_many([text_key(text)], batch_vector, source="batch")

        # Serving path: same text, recomputed key, cache hit.
        np.testing.assert_array_equal(store.get(text_key(text)), batch_vector[0])
