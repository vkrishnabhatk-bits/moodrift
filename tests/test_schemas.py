"""The API contract's edge cases, tested before the API exists.

These are cheap now and load-bearing later: every row of the edge-case table in
docs/api_contract.md that can be decided by the schema alone is asserted here, so Week 3's
implementation has a failing test to satisfy rather than a paragraph to interpret.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.serve import schemas


def _prediction(**overrides):
    base = {
        "stars": 5,
        "confidence": 0.87,
        "probabilities": {1: 0.02, 2: 0.03, 3: 0.03, 4: 0.05, 5: 0.87},
        "feature_source": "store",
        "text_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    }
    return schemas.Prediction(**{**base, **overrides})


class TestPredictRequest:
    def test_text_is_stripped(self):
        assert schemas.PredictRequest(text="  great coffee  ").text == "great coffee"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_blank_text_is_rejected(self, text):
        # Contract: 422, not a fabricated rating. FastAPI turns this into one.
        with pytest.raises(ValidationError):
            schemas.PredictRequest(text=text)

    def test_request_id_is_optional(self):
        assert schemas.PredictRequest(text="fine").request_id is None


class TestBatchPredictRequest:
    def test_items_are_stripped(self):
        request = schemas.BatchPredictRequest(texts=[" a ", "b"])
        assert request.texts == ["a", "b"]

    def test_empty_list_is_rejected(self):
        with pytest.raises(ValidationError):
            schemas.BatchPredictRequest(texts=[])

    def test_one_blank_item_rejects_the_batch(self):
        with pytest.raises(ValidationError):
            schemas.BatchPredictRequest(texts=["fine", "  "])

    def test_oversize_is_not_a_validation_error(self):
        # It is a 413 from the route, not a 422 from pydantic: the request is well-formed,
        # just too large. If this ever starts raising, the contract has drifted.
        texts = ["ok"] * (int(schemas.limits()["max_batch"]) + 1)
        request = schemas.BatchPredictRequest(texts=texts)
        assert schemas.oversized_batch(request.texts)


class TestLimits:
    def test_batch_cap_is_read_from_config(self):
        assert schemas.oversized_batch(["x"] * (int(schemas.limits()["max_batch"]) + 1))
        assert not schemas.oversized_batch(["x"] * int(schemas.limits()["max_batch"]))

    def test_text_cap_is_read_from_config(self):
        cap = int(schemas.limits()["max_input_chars"])
        assert schemas.oversized_text("x" * (cap + 1))
        assert not schemas.oversized_text("x" * cap)


class TestPrediction:
    def test_flags_default_to_false(self):
        flags = _prediction().flags
        assert not (flags.truncated or flags.low_signal or flags.out_of_domain)

    @pytest.mark.parametrize("stars", [0, 6])
    def test_stars_outside_the_label_space_are_rejected(self, stars):
        with pytest.raises(ValidationError):
            _prediction(stars=stars)

    def test_probabilities_must_cover_all_five_classes(self):
        with pytest.raises(ValidationError):
            _prediction(probabilities={1: 0.5, 2: 0.5})

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            _prediction(probabilities={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1})

    def test_feature_source_is_constrained(self):
        assert _prediction(feature_source="live").feature_source == "live"
        with pytest.raises(ValidationError):
            _prediction(feature_source="guessed")


class TestModelRef:
    def test_every_provenance_field_is_required(self):
        # A response that cannot say which model answered is useless for debugging a
        # drift alert weeks later, so none of these may be optional.
        for missing in ("name", "version", "alias", "run_id", "git_sha"):
            fields = {
                "name": "moodrift-classifier",
                "version": "1",
                "alias": "production",
                "run_id": "cdfd6b64221e4244a0e9d0cbca21f76b",
                "git_sha": "542ed28",
            }
            fields.pop(missing)
            with pytest.raises(ValidationError):
                schemas.ModelRef(**fields)
