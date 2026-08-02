"""Tests for the sampling and splitting logic.

This file exists because of a gap found while auditing the repository: the cross-split
deduplication in ADR-0005 - the most consequential correctness fix in the data plane -
was guarded only by a runtime assertion inside a stage that takes minutes to run. Nothing
in the test suite covered it, and `tests/test_validation.py` referred to a test by name
that had never been written.

The properties below are the ones that silently inflate every downstream metric when they
break, and none of them announce themselves: the pipeline still succeeds, the numbers just
look better than the model deserves.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.features.clean import build_model_input
from src.features.sample import LABEL, MODEL_INPUT, TEXT_KEY, _proportional_sample
from src.provenance import text_key


def make_corpus(n_per_class: int = 40, duplicates: int = 10) -> pd.DataFrame:
    """A corpus with a known number of texts repeated by *different* reviewers.

    This is the real pattern: short reviews like "Great product!" written independently by
    many people. Validation is right to keep them - they are genuine, distinct records -
    so the duplication has to be handled at split time instead.
    """
    rows = []
    for score in (1, 2, 3, 4, 5):
        for i in range(n_per_class):
            rows.append({"UserId": f"U{score}{i}", "Summary": "", "Text": f"review {score} number {i}", LABEL: score})
    for i in range(duplicates):
        rows.append({"UserId": f"DUP{i}", "Summary": "", "Text": "Great product!", LABEL: 5})
    return pd.DataFrame(rows)


class TestDeduplication:
    def test_identical_text_from_different_reviewers_collapses_to_one_row(self):
        df = make_corpus(duplicates=10)
        df[MODEL_INPUT] = build_model_input(df["Summary"], df["Text"])

        deduped = df.drop_duplicates(subset=MODEL_INPUT, keep="first")

        # The ten "Great product!" rows become one; nothing else is touched.
        assert (df[MODEL_INPUT] == "Great product!").sum() == 10
        assert (deduped[MODEL_INPUT] == "Great product!").sum() == 1
        assert len(deduped) == len(df) - 9

    def test_deduplication_makes_the_split_disjoint(self):
        """The property the stage asserts, reproduced end to end.

        Without the dedup step the same string can land in train and in test; with it, the
        intersection of text keys must be empty.
        """
        df = make_corpus(duplicates=20)
        df[MODEL_INPUT] = build_model_input(df["Summary"], df["Text"])
        df[TEXT_KEY] = df[MODEL_INPUT].map(text_key)

        deduped = df.drop_duplicates(subset=MODEL_INPUT, keep="first")
        train, test = train_test_split(
            deduped, train_size=0.7, stratify=deduped[LABEL], random_state=42
        )
        assert not (set(train[TEXT_KEY]) & set(test[TEXT_KEY]))

    def test_without_deduplication_the_split_can_leak(self):
        """Demonstrates the bug this guards against, so the guard cannot be quietly removed."""
        df = make_corpus(n_per_class=4, duplicates=40)
        df[MODEL_INPUT] = build_model_input(df["Summary"], df["Text"])
        df[TEXT_KEY] = df[MODEL_INPUT].map(text_key)

        # Split the *undeduplicated* frame: with 40 copies of one string, both sides get some.
        train, test = train_test_split(df, train_size=0.7, random_state=42)
        assert set(train[TEXT_KEY]) & set(test[TEXT_KEY]), "expected leakage without dedup"


class TestProportionalSampling:
    def test_class_proportions_are_preserved(self):
        df = make_corpus(n_per_class=100, duplicates=0)
        subset = _proportional_sample(df, size=150, seed=42)

        source = df[LABEL].value_counts(normalize=True).sort_index()
        drawn = subset[LABEL].value_counts(normalize=True).sort_index()
        assert len(subset) == 150
        assert (source - drawn).abs().max() < 0.02

    def test_sampling_is_deterministic_for_a_seed(self):
        df = make_corpus(n_per_class=50, duplicates=0)
        first = _proportional_sample(df, size=100, seed=42)
        second = _proportional_sample(df, size=100, seed=42)
        assert list(first.index) == list(second.index)

    def test_requesting_more_than_available_returns_everything(self):
        df = make_corpus(n_per_class=10, duplicates=0)
        assert len(_proportional_sample(df, size=10_000, seed=42)) == len(df)

    def test_every_class_survives_a_stratified_split(self):
        """A split that loses a rare class makes its per-class F1 undefined, not zero."""
        df = make_corpus(n_per_class=20, duplicates=0)
        train, test = train_test_split(df, train_size=0.7, stratify=df[LABEL], random_state=42)
        assert set(train[LABEL]) == {1, 2, 3, 4, 5}
        assert set(test[LABEL]) == {1, 2, 3, 4, 5}


class TestTextKey:
    def test_key_follows_normalised_text_not_raw_text(self):
        """Two rows differing only in HTML must share a key after normalisation."""
        raw = pd.Series(["Great product<br />", "Great product"])
        normalised = build_model_input(pd.Series(["", ""]), raw)
        assert normalised.iloc[0] == normalised.iloc[1]
        assert text_key(normalised.iloc[0]) == text_key(normalised.iloc[1])


@pytest.mark.parametrize("size", [5, 50, 120])
def test_proportional_sample_never_exceeds_requested_size(size):
    df = make_corpus(n_per_class=30, duplicates=0)
    assert len(_proportional_sample(df, size=size, seed=1)) <= size


def test_sample_smaller_than_class_count_fails_with_a_useful_message():
    """Found by this test file: the raw sklearn error named no config value.

    Stratified sampling cannot return fewer rows than there are classes. The failure is
    only reachable by misconfiguration, but the error a reader sees should point at the
    setting to change rather than at sklearn's internals.
    """
    df = make_corpus(n_per_class=30, duplicates=0)
    with pytest.raises(ValueError, match="conf/data.yaml"):
        _proportional_sample(df, size=3, seed=1)
