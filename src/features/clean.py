"""Text normalisation shared by training and serving.

This module is the *single* definition of what model input looks like. Both the batch
feature pipeline and the online ``/predict`` path import ``normalise`` from here, which
is what prevents train/serve skew - the classic failure where the two paths drift apart
and nobody notices until accuracy quietly rots.

Deliberate choices, all of which cost accuracy if reversed on this corpus:

* **Negations are preserved.** "not good" must not become "good". No stopword removal.
* **Elongation is collapsed, not stripped.** "sooooo goooood" -> "soo goood": the
  emphasis is signal for star rating, but the unbounded variants fragment the vocabulary.
* **URLs and HTML are replaced with tokens**, not deleted, so their presence stays
  visible to the model (this corpus contains a lot of raw ``<br />``).
* **Case is preserved.** Lowercasing is left to the vectoriser, so the char n-grams and
  the transformer tokenizer can each make their own decision.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]{1,40}>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Amazon product codes (B00XXXXXXX) carry no sentiment but fragment the vocabulary.
_PRODUCT_CODE = re.compile(r"\bB[0-9A-Z]{9}\b")
_ELONGATION = re.compile(r"(\w)\1{2,}")
_WHITESPACE = re.compile(r"\s+")
# Zero-width and control characters, minus the whitespace we normalise separately.
_INVISIBLE = re.compile(r"[​-‏ -‮﻿]")

URL_TOKEN = " <url> "
EMAIL_TOKEN = " <email> "
PRODUCT_TOKEN = " <product> "


def normalise(text: str | None) -> str:
    """Normalise one string. Total function: never raises, always returns a ``str``."""
    if text is None or not isinstance(text, str):
        return ""

    # NFKC folds compatibility variants (full-width chars, ligatures) onto canonical forms.
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE.sub("", out)
    out = out.replace("�", " ")  # replacement chars from the ingest decode
    out = _HTML_BREAK.sub(" ", out)
    out = _HTML_TAG.sub(" ", out)
    out = _URL.sub(URL_TOKEN, out)
    out = _EMAIL.sub(EMAIL_TOKEN, out)
    out = _PRODUCT_CODE.sub(PRODUCT_TOKEN, out)
    out = _ELONGATION.sub(r"\1\1", out)
    out = _strip_control_characters(out)
    out = _WHITESPACE.sub(" ", out)
    return out.strip()


def _strip_control_characters(text: str) -> str:
    """Remove invisible characters, treating separators and joiners differently.

    Control characters (``Cc``: newline, tab, carriage return) become **spaces**, because
    they separate words - deleting them welds the surrounding tokens together
    ("many\\n\\nspaces" -> "manyspaces"), which silently corrupts the vocabulary. Other
    invisibles (``Cf`` format characters, surrogates, unassigned) are deleted, because
    those appear *within* words and a space would split one word into two.
    """
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cc":
            out.append(" ")
        elif category.startswith("C"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def normalise_series(texts: pd.Series) -> pd.Series:
    """Vectorised wrapper over :func:`normalise`."""
    return texts.map(normalise)


def build_model_input(summary: pd.Series, text: pd.Series) -> pd.Series:
    """Compose the field the model actually sees.

    The summary is the reviewer's own headline and is disproportionately predictive of
    the star rating, so it is prepended rather than discarded.
    """
    return (normalise_series(summary) + ". " + normalise_series(text)).str.strip().str.replace(
        r"^\.\s*", "", regex=True
    )
