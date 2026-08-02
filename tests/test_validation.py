"""Tests for the validation rules.

Each test pins one rejection reason. The point is not that bad rows disappear - it is
that they disappear *for a stated reason* and land in quarantine, because a rule that
silently widens is how data quietly rots.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest.schema import _rule_masks, build_schema


@pytest.fixture
def cfg():
    return load_config("data")


def make_row(**overrides):
    row = {
        "Id": 1,
        "ProductId": "B001E4KFG0",
        "UserId": "A3SGXH7AUHU8GW",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 1,
        "Score": 5,
        "Time": 1303862400,
        "Summary": "Good",
        "Text": "A perfectly ordinary review body.",
    }
    row.update(overrides)
    return row


def reasons(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Apply the rules the way the stage does: first match wins."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, mask in _rule_masks(df, cfg):
        newly = mask.fillna(False) & out.isna()
        out[newly] = name
    return out


class TestRejectionRules:
    def test_clean_row_is_accepted(self, cfg):
        df = pd.DataFrame([make_row()])
        assert reasons(df, cfg).isna().all()

    def test_empty_text_is_rejected(self, cfg):
        df = pd.DataFrame([make_row(Text="")])
        assert reasons(df, cfg).iloc[0] == "missing_text"

    def test_whitespace_only_text_is_rejected(self, cfg):
        """A row that is blank after stripping has nothing to classify."""
        df = pd.DataFrame([make_row(Text="   \n\t  ")])
        assert reasons(df, cfg).iloc[0] == "missing_text"

    def test_null_text_is_rejected(self, cfg):
        df = pd.DataFrame([make_row(Text=None)])
        assert reasons(df, cfg).iloc[0] == "missing_text"

    @pytest.mark.parametrize("score", [0, 6, -1])
    def test_out_of_range_score_is_rejected(self, cfg, score):
        df = pd.DataFrame([make_row(Score=score)])
        assert reasons(df, cfg).iloc[0] == "invalid_score"

    def test_overlong_text_is_rejected(self, cfg):
        df = pd.DataFrame([make_row(Text="x" * (cfg["validation"]["max_text_chars"] + 1))])
        assert reasons(df, cfg).iloc[0] == "text_too_long"

    def test_duplicate_id_keeps_the_first(self, cfg):
        df = pd.DataFrame([make_row(Id=1), make_row(Id=1, Text="Different body")])
        result = reasons(df, cfg)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "duplicate_id"

    def test_same_user_same_product_same_text_is_a_duplicate(self, cfg):
        df = pd.DataFrame([make_row(Id=1), make_row(Id=2)])
        result = reasons(df, cfg)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "duplicate_review"

    def test_different_users_with_identical_text_are_kept_here(self, cfg):
        """Both are genuine records, so validation keeps them.

        Cross-split leakage from identical text is a *modelling* concern and is handled
        later, in the sample stage - see test_sampling_deduplicates_text.
        """
        df = pd.DataFrame([make_row(Id=1, UserId="U1"), make_row(Id=2, UserId="U2")])
        assert reasons(df, cfg).isna().all()

    def test_helpfulness_exceeding_total_is_rejected(self, cfg):
        df = pd.DataFrame([make_row(HelpfulnessNumerator=5, HelpfulnessDenominator=2)])
        assert reasons(df, cfg).iloc[0] == "helpfulness_exceeds_total"

    def test_first_matching_rule_wins(self, cfg):
        """One reason per row keeps the quarantine file interpretable."""
        df = pd.DataFrame([make_row(Text="", Score=99)])
        assert reasons(df, cfg).iloc[0] == "missing_text"


class TestSchemaContract:
    def test_clean_frame_satisfies_the_declared_schema(self, cfg):
        df = pd.DataFrame([make_row(Id=i) for i in range(1, 4)])
        df["Score"] = df["Score"].astype("Int8")
        df["Time"] = df["Time"].astype("Int64")
        df["HelpfulnessNumerator"] = df["HelpfulnessNumerator"].astype("Int32")
        df["HelpfulnessDenominator"] = df["HelpfulnessDenominator"].astype("Int32")
        build_schema(cfg).validate(df)

    def test_schema_rejects_what_the_rules_would_have_caught(self, cfg):
        """The contract is a backstop: if the rules ever miss, this must still fail."""
        import pandera as pa

        df = pd.DataFrame([make_row(Score=9)])
        df["Score"] = df["Score"].astype("Int8")
        df["Time"] = df["Time"].astype("Int64")
        df["HelpfulnessNumerator"] = df["HelpfulnessNumerator"].astype("Int32")
        df["HelpfulnessDenominator"] = df["HelpfulnessDenominator"].astype("Int32")

        with pytest.raises(pa.errors.SchemaError):
            build_schema(cfg).validate(df)
