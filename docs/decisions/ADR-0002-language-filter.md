# ADR-0002: py3langid for the tier-0 language filter, applied to the sampled pool

## Status

Accepted — 2026-08-02 (Week 1)

## Context

The project scope is English-only, enforced at the data layer so that non-English reviews
never reach training or become spurious "drift" in Week 3. PROJECT_PLAN.md specifies
fastText `lid.176.ftz` as the tier-0 filter, and describes it as running "at ingestion".

Two practical problems surfaced during implementation.

**Installation.** The `fasttext` package has no prebuilt wheel for Python 3.11 on macOS
arm64 and requires a C++ toolchain to build from source. Every developer machine and every
container image would inherit that build step.

**Cost of placement.** Running language ID over all 567,140 validated rows costs roughly
10 minutes of CPU per pipeline run, to discard a fraction of a percent of rows — measured
at 0.47% on this corpus.

## Decision

**Use `py3langid`** — a maintained pure-Python port of langid.py, same 97-language model,
no compilation step.

**Apply the filter in the sample stage**, to an oversampled candidate pool
(`sample.language_filter_oversample: 1.25`), rather than to the full validated corpus. The
pipeline draws 75,000 candidates, removes duplicate normalised texts (ADR-0005, 7,485 rows),
filters the remainder to English, and takes the final 60,000 proportional sample from the
survivors.

## Alternatives considered

**fastText as specified.** Rejected on install friction alone. Accuracy is not the
deciding factor: for a coarse keep/drop gate on review-length English text both models
agree overwhelmingly, and the confidence threshold (0.90) matters more than the choice of
model.

**Filtering the full corpus at ingestion, as originally planned.** Rejected on cost. The
end state is identical — no non-English row survives into any split — for roughly 1/8th of
the compute, because only the candidate pool is ever examined.

**`langdetect` / `lingua`.** `lingua` is more accurate but substantially slower;
`langdetect` is non-deterministic without explicit seeding, which conflicts with the
project's reproducibility requirement.

## Consequences

- No compiler is needed to run this project; `pip install` alone is sufficient.
- Language ID cost scales with sample size, not corpus size — so raising `sample.size`
  raises it proportionally rather than being fixed at the full-corpus cost.
- Texts shorter than 25 characters are **kept unconditionally**: below that length
  language ID approaches a coin flip, and filtering on it would bias the corpus toward
  long reviews rather than toward English ones.
- The filter reports how many rows it dropped (317 on the current run), so an unexpected
  jump is visible rather than silent.
- If the team later wants fastText specifically, `src/ingest/language_filter.py` exposes a
  single `english_mask` function and can be swapped without touching the sample stage.
