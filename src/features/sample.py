"""Sample stage: sample, deduplicate, filter to English, split, freeze a reference window.

Four decisions worth stating explicitly, because each one is easy to get wrong:

1. **The sample is proportional, not balanced.** The real 1-5 star imbalance is preserved
   so that macro-F1 is measured against the distribution the model will actually meet, and
   so the drift reference window is realistic. Imbalance is corrected at *training* time
   via class weights (ADR-0001).
2. **Splitting is stratified and seeded**, so every class is represented in val/test and
   the split is byte-identical on re-runs.
3. **Identical normalised texts are removed before splitting.** Different reviewers write
   byte-identical short reviews; letting one land in train and another in test scores the
   model on memorised strings. This was ~13% of the test split before it was fixed
   (ADR-0005), and an assertion below now guards it.
4. **The reference window is frozen now**, from the test split, for the drift detectors
   built later. Deriving it after the model exists would let model behaviour contaminate
   the baseline it is measured against.

Run with ``python -m src.features.sample``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ensure_parent, load_config, resolve
from src.features.clean import build_model_input
from src.ingest.language_filter import english_mask
from src.provenance import set_seeds, text_key

LABEL = "Score"
MODEL_INPUT = "model_input"
TEXT_KEY = "text_key"

_OUTPUT_COLUMNS = [
    "Id",
    "ProductId",
    "UserId",
    "Time",
    "Summary",
    "Text",
    MODEL_INPUT,
    TEXT_KEY,
    LABEL,
]


def load_split(name: str) -> pd.DataFrame:
    """Read one of the frozen splits (``train``/``val``/``test``/``reference``).

    Every training and evaluation entry point goes through here, so no stage can
    accidentally read a different file or re-derive a split of its own.
    """
    paths = load_config("data")["paths"]
    if name not in paths:
        raise KeyError(f"unknown split {name!r}; expected one of train/val/test/reference")
    path = resolve(paths[name])
    if not path.exists():
        raise FileNotFoundError(f"{name} split not found at {path} - run `make sample` first")
    return pd.read_parquet(path)


def _proportional_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Stratified subset that preserves the source class proportions.

    Raises a clear error when ``size`` is smaller than the number of classes. Stratified
    sampling cannot satisfy that request, and sklearn's own message ("train_size = 1
    should be greater or equal to the number of classes = 5") gives no hint about which
    config value caused it. Silently falling back to an unstratified draw would be worse:
    it would quietly break the proportionality guarantee this function exists to provide.
    """
    if size >= len(df):
        return df.copy()

    n_classes = df[LABEL].nunique()
    if size < n_classes:
        raise ValueError(
            f"cannot draw a stratified sample of {size} rows across {n_classes} classes - "
            "every class needs at least one row. Raise sample.size (or "
            "sample.reference_window_size) in conf/data.yaml."
        )

    subset, _ = train_test_split(df, train_size=size, stratify=df[LABEL], random_state=seed)
    return subset


def sample() -> dict[str, Any]:
    """Run the sample stage end to end and return its statistics."""
    cfg = load_config("data")
    paths, sample_cfg = cfg["paths"], cfg["sample"]
    seed = sample_cfg["seed"]
    set_seeds(seed)

    df = pd.read_parquet(resolve(paths["validated"]))
    print(f"[sample] validated rows: {len(df):,}")

    # 1. Oversampled candidate pool, so the language filter runs on a bounded number of rows.
    target = int(sample_cfg["size"])
    pool_size = min(len(df), int(target * float(sample_cfg["language_filter_oversample"])))
    pool = _proportional_sample(df, pool_size, seed)
    print(f"[sample] candidate pool: {len(pool):,} rows")

    # 2. Normalise text once, here, and carry it forward. Downstream stages never re-clean.
    pool = pool.copy()
    pool[MODEL_INPUT] = build_model_input(pool["Summary"], pool["Text"])
    # A normalisation that empties a row leaves nothing to classify.
    non_empty = pool[MODEL_INPUT].str.len() > 0
    pool = pool[non_empty]

    # 2b. Deduplicate on the normalised text *before* splitting.
    #
    # The validation stage already removed same-user-same-product duplicates, but this
    # corpus also contains short reviews ("Great product!") posted independently by
    # different reviewers. Those are distinct records, yet after normalisation they are
    # byte-identical strings - and if the same string lands in train and in test, the
    # model is scored partly on text it memorised. Measured on the first run this was
    # ~13% of the test split, which would have inflated every metric below.
    #
    # We deduplicate here rather than in validation because it is a *modelling* concern
    # (leakage across splits), not a data-integrity one: the underlying rows are real.
    before_dedup = len(pool)
    pool = pool.drop_duplicates(subset=MODEL_INPUT, keep="first")
    n_duplicate_texts = before_dedup - len(pool)
    print(f"[sample] dropped {n_duplicate_texts:,} duplicate normalised texts (cross-split leakage)")

    # 3. English-only (tier 0).
    keep = english_mask(pool[MODEL_INPUT], cfg)
    n_dropped = int((~keep).sum())
    pool = pool[keep]
    print(f"[sample] language filter dropped {n_dropped:,} non-English rows -> {len(pool):,} remain")

    # 4. Final proportional sample at the configured size.
    sampled = _proportional_sample(pool, min(target, len(pool)), seed)
    sampled = sampled.copy()
    sampled[TEXT_KEY] = sampled[MODEL_INPUT].map(text_key)
    sampled = sampled[_OUTPUT_COLUMNS]
    print(f"[sample] final sample: {len(sampled):,} rows")

    # 5. Seeded, stratified 70/15/15.
    splits = sample_cfg["splits"]
    train_df, temp_df = train_test_split(
        sampled, train_size=splits["train"], stratify=sampled[LABEL], random_state=seed
    )
    val_share = splits["val"] / (splits["val"] + splits["test"])
    val_df, test_df = train_test_split(
        temp_df, train_size=val_share, stratify=temp_df[LABEL], random_state=seed
    )

    # 6. Freeze the drift reference window from the test split.
    ref_size = min(int(sample_cfg["reference_window_size"]), len(test_df))
    reference_df = _proportional_sample(test_df, ref_size, seed)

    # Check the invariants BEFORE writing anything. These assertions previously ran after
    # the parquet files were written, which meant a failure left corrupt splits on disk for
    # the next stage to pick up - the opposite of what a guard is for.
    #
    # No normalised text may appear in more than one split, or test scores are inflated.
    overlap = set(train_df[TEXT_KEY]) & (set(val_df[TEXT_KEY]) | set(test_df[TEXT_KEY]))
    assert not overlap, f"{len(overlap)} texts leak between train and val/test"

    # Proportions must survive the split; a silent stratification bug is expensive later.
    train_share = train_df[LABEL].value_counts(normalize=True).sort_index()
    test_share = test_df[LABEL].value_counts(normalize=True).sort_index()
    drift = (train_share - test_share).abs().max()
    assert drift < 0.02, f"train/test class proportions diverge by {drift:.3f} - stratification failed"
    print(f"[sample] max train/test class-proportion gap: {drift:.4f}")

    for name, frame in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
        ("reference", reference_df),
    ):
        dest = ensure_parent(resolve(paths[name]))
        frame.to_parquet(dest, index=False, compression="zstd")
        print(f"[sample] {name:<9} {len(frame):>7,} rows -> {paths[name]}")

    stats = {
        "validated_rows": len(df),
        "pool_rows": pool_size,
        "duplicate_texts_dropped": n_duplicate_texts,
        "non_english_dropped": n_dropped,
        "sampled_rows": len(sampled),
        "splits": {k: len(v) for k, v in (("train", train_df), ("val", val_df), ("test", test_df))},
        "reference_rows": len(reference_df),
        "class_distribution": {
            name: {int(k): int(v) for k, v in frame[LABEL].value_counts().sort_index().items()}
            for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))
        },
    }
    return stats


if __name__ == "__main__":
    sample()
