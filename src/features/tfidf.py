"""Tier-1 features: combined word and character TF-IDF.

The vectoriser is **fit on the training split only**. Fitting on all rows (or re-fitting at
evaluation) would leak test vocabulary and inflate every metric downstream - the single
most common silent bug in text pipelines.

It is not persisted separately. The vectoriser is the first step of the tier-1 sklearn
``Pipeline``, so the fitted vocabulary is serialised with the model and travels with it
into the MLflow registry. That is deliberate: a vectoriser saved beside a model is a
vectoriser that can be paired with the wrong one.

Word and char n-grams are unioned rather than chosen between: word n-grams carry phrasing,
char n-grams absorb the typos, elongation and emoji that this corpus is full of. The char
analyser is also the dominant training cost - see the note in ``src/train/tier1.py``.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer(cfg: dict[str, Any]) -> FeatureUnion:
    """Construct the (unfitted) word + char TF-IDF union from config."""
    word_cfg = dict(cfg["vectorizer"]["word"])
    char_cfg = dict(cfg["vectorizer"]["char"])
    for c in (word_cfg, char_cfg):
        if "ngram_range" in c:
            c["ngram_range"] = tuple(c["ngram_range"])

    return FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=True, **word_cfg)),
            ("char", TfidfVectorizer(lowercase=True, **char_cfg)),
        ]
    )
