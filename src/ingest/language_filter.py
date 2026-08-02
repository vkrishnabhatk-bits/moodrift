"""Tier 0: language identification, used to keep the corpus English-only.

The plan specifies fastText ``lid.176.ftz``. We use ``py3langid`` instead - the same
langid.py 97-language model, but pure Python with no compilation step, which matters
because fastText has no prebuilt wheel for this platform. Accuracy on review-length
text is equivalent for our purpose (a coarse keep/drop gate), and the swap removes a
build dependency from every developer machine and container.

Rationale recorded in ``docs/decisions/ADR-0002-language-filter.md``.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _identifier():
    """Load the langid model once; it is ~1MB and thread-safe for our read-only use."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    # norm_probs=True turns the raw log-probabilities into a usable 0-1 confidence.
    return LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


def detect(text: str) -> tuple[str, float]:
    """Return ``(language_code, confidence)`` for one string."""
    if not text or not text.strip():
        return "unknown", 0.0
    lang, confidence = _identifier().classify(text)
    return lang, float(confidence)


def detect_series(texts: pd.Series) -> pd.DataFrame:
    """Language-identify a column, returning ``language`` and ``language_confidence``."""
    results = [detect(t if isinstance(t, str) else "") for t in texts]
    return pd.DataFrame(results, columns=["language", "language_confidence"], index=texts.index)


def english_mask(texts: pd.Series, cfg: dict) -> pd.Series:
    """Boolean mask of rows to KEEP under the configured language policy.

    Short texts are kept unconditionally: below roughly 25 characters language ID is
    close to a coin flip, and discarding them would bias the corpus toward long reviews
    rather than toward English ones.
    """
    lf = cfg["language_filter"]
    if not lf.get("enabled", True):
        return pd.Series(True, index=texts.index)

    detected = detect_series(texts)
    too_short = texts.fillna("").str.len() < lf["min_chars_for_detection"]
    confident_target = (detected["language"] == lf["target"]) & (
        detected["language_confidence"] >= lf["min_confidence"]
    )
    return (confident_target | too_short).rename("keep")
