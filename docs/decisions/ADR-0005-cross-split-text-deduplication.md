# ADR-0005: Deduplicate identical normalised text before splitting

## Status

Accepted — 2026-08-02 (Week 1)

## Context

This decision was forced by an observation, not planned in advance.

The validation stage already removes duplicate reviews, defined as the *same reviewer*
posting the *same text* on the *same product* — a genuine double-submission.

After the first end-to-end run, the feature-store population log showed something the
validation rule had not caught:

```
[embed] train     encoding 39,336 texts (0 cached)
[embed] val       encoding  7,920 texts (955 cached)
[embed] test      encoding  7,706 texts (1,147 cached)
```

The "cached" counts are texts already embedded while processing an earlier split. In other
words **955 validation rows and 1,147 test rows had normalised text byte-identical to a
training row** — about 13% of the test split.

The cause is benign and common in review corpora: short reviews such as `"Great product!"`
or `"Arrived quickly. Excellent."` are written independently by many different reviewers.
Each row is a real, distinct record, so validation is right to keep them. But once the same
string appears in both train and test, the model is scored partly on strings it memorised,
and every test metric is inflated by an unknown amount.

This is exactly the class of bug that never announces itself: nothing crashes, and the
metrics simply look better than the model deserves.

## Decision

Deduplicate on the **normalised model input** in the sample stage, keeping the first
occurrence, *before* the train/val/test split.

The stage then asserts the property directly, so a regression cannot reintroduce it:

```python
overlap = set(train_df[TEXT_KEY]) & (set(val_df[TEXT_KEY]) | set(test_df[TEXT_KEY]))
assert not overlap, f"{len(overlap)} texts leak between train and val/test"
```

On the current pool this drops 7,485 rows (~10%).

## Alternatives considered

**Do nothing and report the higher numbers.** Rejected. The inflation is real and
unquantified, and it would be discovered — by a grader, or by the model underperforming
in Week 3 serving relative to its logged metrics.

**Deduplicate in the validation stage instead.** Rejected on separation-of-concerns
grounds: validation answers "is this row valid data?", and these rows are valid. This is a
modelling concern about how the *splits* are constructed, so it belongs with the splitting
logic, where its effect is visible next to the stratification it interacts with.

**Split by reviewer or product instead (grouped split).** A stronger guarantee — it would
also prevent the same reviewer's style appearing on both sides — but it distorts the class
distribution, which ADR-0001 deliberately preserves. Recorded as a possible refinement if
Week 2 shows evidence of reviewer-level memorisation.

## Consequences

- No model was ever trained on the leaking splits. The overlap was caught from the embed
  cache counts before any training run, so there are no inflated "before" figures to
  compare against - every number in `docs/model_comparison.md` is post-fix by construction.
  The size of the inflation this avoided is therefore unmeasured, only bounded by the ~13%
  of the test split that would have been memorised.
- The effective sample is drawn from a slightly smaller pool; `language_filter_oversample`
  (1.25) leaves enough headroom to still reach the configured 60,000 rows.
- Deduplication runs before the language filter, so it also reduces language-ID work.
- A permanent assertion now guards the property, so this cannot silently regress.

## Related

- [ADR-0001: sampling and class imbalance](ADR-0001-sampling-and-class-imbalance.md) —
  the other half of the sampling design.
