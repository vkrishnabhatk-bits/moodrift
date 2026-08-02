"""Tier-1 features: combined word and character TF-IDF.

The vectoriser is **fit on the training split only** and persisted as an artifact. Fitting
on all rows (or re-fitting at evaluation) would leak test vocabulary and inflate every
metric downstream - the single most common silent bug in text pipelines.

Word and char n-grams are unioned rather than chosen between: word n-grams carry phrasing,
char n-grams absorb the typos, elongation and emoji that this corpus is full of.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

ARTIFACT_NAME = "tfidf_vectorizer.joblib"


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


def save_vectorizer(vectorizer: FeatureUnion, directory: Path) -> Path:
    """Persist the fitted vectoriser next to the model that depends on it."""
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ARTIFACT_NAME
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(directory: Path) -> FeatureUnion:
    """Load a previously fitted vectoriser."""
    import joblib

    return joblib.load(Path(directory) / ARTIFACT_NAME)
