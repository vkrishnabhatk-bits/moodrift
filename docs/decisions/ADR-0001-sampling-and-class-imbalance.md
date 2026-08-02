# ADR-0001: Proportional sampling, with imbalance handled at training time

## Status

Accepted — 2026-08-02 (Week 1)

## Context

Amazon Fine Food Reviews contains 568,454 reviews, and the star distribution measured
during ingest is severely imbalanced:

| Score | Rows | Share |
|---|---|---|
| 1 | 52,268 | 9.2% |
| 2 | 29,769 | 5.2% |
| 3 | 42,640 | 7.5% |
| 4 | 80,655 | 14.2% |
| 5 | 363,122 | 63.9% |

Two separate decisions follow from this: how many rows to train on, and what to do about
the imbalance.

On volume: the full corpus is workable for TF-IDF but not for repeatedly embedding every
row on a laptop during a compressed three-week build, and the fine-tune in tier 3 has to
fit inside a free Colab session.

On imbalance: it is tempting to balance the sample so that every class has equal
representation, which makes training simpler and the numbers look better.

## Decision

**Sample 60,000 rows, stratified proportionally**, preserving the real class distribution.
**Handle imbalance at training time** via `class_weight='balanced'` (tier 1 and tier 2),
never by resampling the dataset.

Splits are 70/15/15, stratified and seeded (`seed: 42`), and the stage asserts that
train/test class proportions agree to within 2 percentage points.

## Alternatives considered

**Balanced (equal-size-per-class) sampling.** Rejected on two grounds. It would cap the
dataset at roughly 5× the smallest class (~149K rows before other filters, but far fewer
after), discarding most 5-star data. More importantly it makes every downstream number a
lie about production: macro-F1 would be measured against a distribution the model will
never meet, and the drift reference window — frozen from this same test split — would
encode a class balance that real traffic does not have, so Week 3's detectors would fire
on the difference between our sample and reality rather than on genuine drift.

**Oversampling the minority classes (SMOTE or duplication).** Rejected: on text, synthetic
interpolation between TF-IDF or embedding vectors produces points that correspond to no
real sentence, and naive duplication mostly teaches the model to memorise the duplicated
rows. Class weighting achieves the same reweighting without inventing data.

**Using the full 568K rows.** Rejected for cost, not correctness. It would make each
embedding pass roughly 10× longer for a modest accuracy gain, and the sample size is a
single config value (`sample.size`) that can be raised later if time allows.

## Consequences

- Reported metrics reflect the real distribution, so macro-F1 is honest but *lower* than a
  balanced-sample number would be. This is a feature; see the Week 1 results discussion.
- Accuracy is a near-useless headline here (predicting 5 unconditionally scores ~0.64), so
  macro-F1 and macro-MAE carry the evaluation instead — see `src/evaluate/metrics.py`.
- The minority classes (2 and 3 stars) will remain the weakest, and per-class F1 is
  reported so that weakness is visible rather than averaged away.
- `sample.size` is config, not code: raising it changes the DVC stage hash and triggers a
  clean rebuild of everything downstream.

## Related

- [ADR-0005: cross-split text deduplication](ADR-0005-cross-split-text-deduplication.md) —
  the other sampling decision, made after the first run revealed leakage.
