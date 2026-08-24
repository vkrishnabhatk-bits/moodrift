"""Drift detectors: input drift (PSI + KS), concept drift (a deliberately weak domain
classifier), and performance drift (rolling macro-F1/macro-MAE against a pinned baseline).

Design and thresholds: `docs/decisions/ADR-0006-drift-detection-approach.md`,
`docs/drift_design.md`. Three unrelated failures hide under "drift" and each has its own
detector - see the table there for which is which and why a statistical shift in the
input does not by itself mean the model got worse.

Nothing here trains a model or writes to the registry: a detector reports a number, and
`src.monitor.trigger` turns numbers into a decision. Keeping the two separate is what
makes the decision auditable - you can read a trigger log and see exactly why, without
re-running the detectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import load_config, resolve
from src.evaluate.metrics import compute_metrics

_WORD_RE = re.compile(r"[A-Za-z']+")


def _words(text: str) -> list[str]:
    """Monitoring's own cheap word tokenisation - not the model's subword tokenizer.

    Good enough for a scalar length/vocabulary feature, and means this offline script
    never has to load the champion's tokenizer just to count words.
    """
    return _WORD_RE.findall(text.lower())


@lru_cache(maxsize=1)
def reference_vocabulary(min_freq: int = 2) -> frozenset[str]:
    """Word vocabulary frozen from the reference window (Week 1, before any model
    existed) - the "corpus vocabulary frozen at Week 1" option from
    `docs/drift_design.md`'s open question, chosen over the tier-1 model's TF-IDF
    vocabulary so monitoring never has to load a specific model version just to compute
    an input-drift feature. `min_freq=2` drops hapax legomena (typos, one-off product
    codes) that would otherwise inflate OOV on both windows for no informative reason.
    """
    cfg = load_config("monitor")
    ref = pd.read_parquet(resolve(cfg["reference"]["path"]), columns=["model_input"])
    counts: dict[str, int] = {}
    for text in ref["model_input"]:
        for word in _words(text):
            counts[word] = counts.get(word, 0) + 1
    return frozenset(word for word, count in counts.items() if count >= min_freq)


def text_features(texts: pd.Series) -> pd.DataFrame:
    """The four scalar input-drift features named in `conf/monitor.yaml`.

    Scalar and interpretable on purpose (`docs/drift_design.md`): if a detector fires on
    one of these, a histogram shows why in about ten seconds - no per-token frequency
    drift over the full vocabulary to untangle.
    """
    vocab = reference_vocabulary()
    rows = []
    for text in texts.fillna(""):
        words = _words(text)
        oov = sum(1 for word in words if word not in vocab)
        rows.append(
            {
                "char_count": len(text),
                "token_count": len(words),
                "oov_rate": oov / len(words) if words else 0.0,
                "mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
            }
        )
    return pd.DataFrame(rows)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index, quantile-binned on the reference distribution.

    Outer edges are pinned to +/-inf so a current-window value outside the reference
    range still lands in a bin instead of being silently dropped from the count - the
    exact case a length-shift or vocabulary-shift scenario is meant to produce.
    """
    reference = pd.Series(reference).astype(float)
    current = pd.Series(current).astype(float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # degenerate reference (near-constant feature) - nothing to compare
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    # Laplace-style floor: an empty bin would make the log-ratio undefined, and a bin
    # that only the current window populated is exactly the signal PSI exists to catch.
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    alert: bool


def input_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, text_col: str = "Text") -> dict[str, Any]:
    """PSI + KS on each configured scalar feature.

    KS never triggers alone (`conf/monitor.yaml` `input_drift.ks.require_psi_agreement`):
    with `windows.size` samples per window a KS test finds statistically significant
    differences that are practically meaningless. PSI measures how much the distribution
    moved; KS measures whether it moved by more than chance. Requiring both means the
    alert says "a real amount, and not by chance".
    """
    cfg = load_config("monitor")["input_drift"]
    ref_features = text_features(reference_df[text_col])
    cur_features = text_features(current_df[text_col])

    results = []
    for feature in cfg["features"]:
        feature_psi = psi(ref_features[feature], cur_features[feature], bins=int(cfg["psi"]["bins"]))
        ks_stat, ks_p = stats.ks_2samp(ref_features[feature], cur_features[feature])
        psi_alert = feature_psi >= float(cfg["psi"]["alert"])
        ks_significant = bool(ks_p < float(cfg["ks"]["alpha"]))
        alert = bool(psi_alert and (ks_significant if cfg["ks"]["require_psi_agreement"] else True))
        results.append(FeatureDrift(feature, feature_psi, float(ks_stat), float(ks_p), alert))

    return {
        "features": [vars(r) for r in results],
        "alert": any(r.alert for r in results),
        "warn": any(r.psi >= float(cfg["psi"]["warn"]) for r in results),
    }


def concept_drift(reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> dict[str, Any]:
    """A deliberately weak, cross-validated classifier separating the two windows.

    Weak on purpose (`C=0.1`, 3-fold CV, per `conf/monitor.yaml`): a strong model on
    384-dimensional embeddings will separate almost any two samples given enough
    capacity, including two random halves of the same distribution - it would fire every
    window and be switched off within a day. AUC near 0.5 means the windows are
    interchangeable; AUC well above it means a weak, regularised model found real
    structure, which is a much higher bar.
    """
    cfg = load_config("monitor")["concept_drift"]
    dc = cfg["domain_classifier"]
    reference_embeddings = np.asarray(reference_embeddings)
    current_embeddings = np.asarray(current_embeddings)

    features = np.vstack([reference_embeddings, current_embeddings])
    labels = np.concatenate(
        [np.zeros(len(reference_embeddings)), np.ones(len(current_embeddings))]
    )
    clf = LogisticRegression(
        C=float(dc["C"]), max_iter=int(dc["max_iter"]), random_state=int(dc["seed"])
    )
    folds = min(int(dc["cv_folds"]), min(np.bincount(labels.astype(int))))
    scores = cross_val_score(clf, features, labels, cv=max(folds, 2), scoring="roc_auc")
    auc = float(np.mean(scores))

    return {
        "auc": auc,
        "alert": auc >= float(cfg["auc_alert"]),
        "warn": auc >= float(cfg["auc_warn"]),
    }


def performance_drift(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Rolling macro-F1 / macro-MAE on a labelled slice, against the pinned baseline.

    The only detector that needs ground truth, and therefore the only one that cannot
    run on live traffic without a label lagging behind it (`docs/drift_design.md`). Below
    `min_labelled_samples`, a 5-class macro-F1 swings on a handful of minority-class rows
    and "drift" would just be sampling noise - reported as not evaluated instead.
    """
    cfg = load_config("monitor")["performance_drift"]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < int(cfg["min_labelled_samples"]):
        return {"evaluated": False, "n_samples": n, "alert": False, "reason": "below min_labelled_samples"}

    metrics = compute_metrics(y_true, y_pred)
    f1_drop = float(cfg["baseline_macro_f1"]) - metrics["macro_f1"]
    mae_rise = metrics["macro_mae"] - float(cfg["baseline_macro_mae"])
    alert = f1_drop >= float(cfg["macro_f1_drop"]) or mae_rise >= float(cfg["macro_mae_rise"])

    return {
        "evaluated": True,
        "n_samples": n,
        "macro_f1": metrics["macro_f1"],
        "macro_mae": metrics["macro_mae"],
        "macro_f1_drop": f1_drop,
        "macro_mae_rise": mae_rise,
        "alert": alert,
    }
