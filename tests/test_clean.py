"""Tests for text normalisation.

These lock in the choices that would silently cost accuracy if someone "tidied up" the
cleaner later - negation survival above all.
"""

from __future__ import annotations

import pandas as pd

from src.features.clean import build_model_input, normalise


class TestNegationSurvives:
    """The single most important property: "not good" must not become "good"."""

    def test_not_is_preserved(self):
        assert "not" in normalise("This is not good at all").lower()

    def test_negated_phrase_differs_from_positive(self):
        assert normalise("would not recommend") != normalise("would recommend")


class TestNormalisation:
    def test_html_breaks_removed(self):
        assert "<br" not in normalise("Great product<br />Would buy again")

    def test_urls_become_a_token(self):
        out = normalise("See https://example.com/page for details")
        assert "http" not in out
        assert "<url>" in out

    def test_product_codes_become_a_token(self):
        assert "<product>" in normalise("I ordered B001E4KFG0 last week")

    def test_elongation_is_collapsed_to_exactly_two_characters(self):
        """Pins the exact output.

        An earlier version asserted only that "soo" appeared and "sooo" did not, which was
        loose enough that the module docstring documented the wrong result ("soo goood")
        for some time without any test failing.
        """
        assert normalise("sooooo goooood") == "soo good"
        assert normalise("aaa") == "aa"
        assert normalise("aa") == "aa"  # runs of two are already short enough to keep

    def test_zero_width_characters_removed(self):
        assert normalise("good​product") == "goodproduct"

    def test_replacement_characters_removed(self):
        assert "�" not in normalise("caf� was fine")

    def test_whitespace_collapsed(self):
        assert normalise("too    many\n\nspaces") == "too many spaces"

    def test_newlines_separate_words_rather_than_welding_them(self):
        """Regression: deleting control characters merged the words around them.

        Harmless on the SNAP corpus (single-line rows) but not at serving time, where
        multi-line pasted reviews are routine - "great\\nproduct" must not tokenise as
        the single unknown word "greatproduct".
        """
        assert normalise("great\nproduct") == "great product"
        assert normalise("line one\r\nline two") == "line one line two"
        assert normalise("col1\tcol2") == "col1 col2"

    def test_case_is_preserved(self):
        # Lowercasing is the vectoriser's decision, not the cleaner's.
        assert "GREAT" in normalise("GREAT product")


class TestTotality:
    """normalise() is called on user input at serving time; it must never raise."""

    def test_none_returns_empty(self):
        assert normalise(None) == ""

    def test_non_string_returns_empty(self):
        assert normalise(12345) == ""  # type: ignore[arg-type]

    def test_empty_and_whitespace(self):
        assert normalise("") == ""
        assert normalise("   \n\t ") == ""

    def test_emoji_only_survives(self):
        # Emoji-only input is low-signal but must still round-trip, not vanish or crash.
        assert normalise("🎉🎉") != ""


class TestModelInput:
    def test_summary_is_prepended(self):
        out = build_model_input(pd.Series(["Great buy"]), pd.Series(["Arrived on time"]))
        assert out.iloc[0].startswith("Great buy")
        assert "Arrived on time" in out.iloc[0]

    def test_missing_summary_leaves_no_leading_separator(self):
        out = build_model_input(pd.Series([""]), pd.Series(["Just the body"]))
        assert not out.iloc[0].startswith(".")
