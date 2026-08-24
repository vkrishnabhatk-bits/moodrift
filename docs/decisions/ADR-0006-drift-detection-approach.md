# ADR-0006: Drift detection approach — PSI+KS, a weak domain classifier, rolling macro-F1/MAE

## Status

Accepted — 2026-08-24 (Week 3)

## Context

`docs/drift_design.md` (Week 2) laid out the reasoning ahead of the implementation:
"drift" covers three unrelated failures — input drift, concept drift, performance drift —
and each needs its own detector, because only performance drift is directly harmful and
only performance drift needs a label. `conf/monitor.yaml` pinned the thresholds. This ADR
records what implementing and running `src/monitor/detectors.py` and
`src/monitor/simulate.py` against real data confirmed, and one thing it corrected.

Three open questions were left for Week 3:

1. Calibrate the domain-classifier AUC floor (0.65) on two random halves of the reference
   window before trusting it.
2. Decide whether `oov_rate` is computed against the tier-1 TF-IDF vocabulary or a corpus
   vocabulary frozen at Week 1.
3. Decide whether the labelled slice for performance drift is simulated or hand-labelled.

## Decision

**Input drift: PSI + KS on `char_count`, `token_count`, `oov_rate`, `mean_word_length`**,
exactly as designed. KS never triggers alone (`require_psi_agreement`). Verified on the
frozen reference window split into two random halves (`tests/test_monitor_detectors.py`,
`test_input_drift_is_quiet_on_a_null_case`): no alert, all four PSI values under 0.02
except `oov_rate` at 0.159 — see the "known asymmetry" note below. On a real shift (the
pool's top-length-quartile reviews vs. the reference), `char_count` PSI reaches ~6.9 and
the detector alerts correctly.

**Question 2, resolved: a corpus vocabulary frozen at Week 1**, built from the reference
window itself (`detectors.reference_vocabulary`, word tokens occurring ≥2 times), not the
tier-1 model's TF-IDF vocabulary. This decouples monitoring from any specific model
version — a monitoring pipeline should not need to reload a specific tier's fitted
pipeline just to compute an input-drift feature, and the reference window is already the
frozen, DVC-versioned artifact that exists precisely to serve as monitoring's baseline.

*Known asymmetry, worth stating rather than hiding*: because the vocabulary is fit on the
reference window itself, reference rows have a structurally lower `oov_rate` against it
than genuinely external rows do, even under no drift — measured at PSI ≈ 0.16 on the null
case above, comfortably under the 0.20 alert line but above the 0.10 warn line. This is a
property of a self-fit vocabulary, not a bug; if it ever needs tightening, the fix is a
larger or separately-frozen vocabulary corpus, not a threshold change.

**Concept drift: a deliberately weak (`C=0.1`, 3-fold CV) logistic regression** on
feature-store MiniLM embeddings, exactly as designed.

**Question 1, resolved: the 0.65 alert floor holds.** Calibrated on two random halves of
the reference window (2,500 rows each): **AUC = 0.490**, essentially chance, 15 points
below the alert line and 11 below the warn line. A strong classifier was not needed to
show this — a *weak, regularised* one already can't separate two halves of the same
distribution, which is exactly the property the design asked for. No config change.

**Performance drift: rolling macro-F1 and macro-MAE against a pinned baseline**, as
designed — but the baseline itself was wrong, and this ADR is where that got caught.

**A real train/serve skew, found by running the detector, not assumed.** The pinned
baseline (0.6001 macro-F1) is tier3's own *training-time* evaluation number, which scores
`model_input` = `Summary + ". " + Text` (`src/features/clean.build_model_input`). The live
`POST /predict` endpoint has no Summary field — `PredictRequest` is a single `text` string
(`docs/api_contract.md`) — and `src/serve/app.py` feeds that string to the model with
nothing prepended. Scored the way `/predict` actually scores (`Text` only) on 4,000 real
held-out rows (the frozen test split, minus whatever rows the reference window already
claimed): **macro-F1 0.569, macro-MAE 0.474** — roughly 4.8 macro-F1 points below the
training-time number. `conf/monitor.yaml`'s `performance_drift.baseline_macro_f1` /
`baseline_macro_mae` are corrected to these measured numbers; `macro_f1_drop` /
`macro_mae_rise` are recomputed so the trigger still fires exactly at
`PROJECT_PLAN.md` §2's absolute floors (0.55 macro-F1, 0.55 macro-MAE) rather than at a
now-meaningless fixed offset from the old baseline. Full reasoning in the config comment.

*This is a serving-plane issue, not a monitoring-config issue*, and is out of scope to fix
here (`/predict`'s contract, tests and Postman collection are owned by the serving plane's
Person B and were reviewed and merged in PR #1). Flagged for that owner: either accept
the accuracy cost of a Summary-less API, or add an optional summary field to
`PredictRequest` and prepend it the way training does, closing the skew instead of just
monitoring around it.

**"Rolling" means rolling, not "fresh per window."** The first implementation evaluated
performance drift on each window's own 500 rows in isolation and found it unusable: five
clean, uninjected 500-row windows produced macro-F1 anywhere from 0.52 to 0.60, three of
them already past the (uncorrected) alert threshold from sampling noise alone — a 5-class
macro-F1 is not stable on 500 rows when several classes are a minority of them.
`src/monitor/simulate.py` now pools the last `ROLLING_WINDOWS` windows before scoring,
matching `docs/drift_design.md`'s own word for this metric. The buffer size was chosen by
testing, not guessing: `ROLLING_WINDOWS=4` (2,000 rows) still produced two false alerts
across 8 clean windows after the baseline correction above; `6` (3,000 rows) produced zero
across 16 consecutive clean windows, minimum observed macro-F1 0.5519 against a 0.55
floor; `8` bought no further stability worth its larger buffer. Settled on `6`.

**Question 3, resolved: simulated, and stated as simulated.** The labelled slice is the
frozen test split (real text, real star ratings) replayed through the same inference path
`/predict` uses, not a live labelled-feedback loop the project doesn't have. Documented
here rather than left implicit.

## Alternatives considered

**Per-token frequency drift over the full vocabulary**, for input drift. Rejected in the
original design: thousands of correlated tests, a multiple-comparisons problem, and a
result nobody can read. `oov_rate` captures the same signal in one interpretable number.

**A strong concept-drift classifier** (gradient boosting on 384-d embeddings). Rejected:
would separate almost any two samples given enough capacity, including two random halves
of the same distribution, and would fire every window.

**Re-baselining performance drift from MLflow at runtime.** Rejected, twice over: it would
have silently hidden the skew this ADR found instead of surfacing it, and a monitoring run
should not be able to quietly rebaseline itself against whatever the champion currently
measures.

## Consequences

- `conf/monitor.yaml`'s `performance_drift` block now documents *why* its numbers are what
  they are, including the correction, rather than presenting them as originally-correct.
- The train/serve skew is real and currently unfixed; `docs/drift_report.md` (generated by
  `python -m src.monitor.simulate`) is the up-to-date evidence for it and for the four
  scenario runs.
- Anyone re-running `python -m src.monitor.simulate` after a future serving-plane change
  (e.g. a summary field added to `/predict`) should re-measure the baseline rather than
  assume this ADR's numbers still hold - the whole point of this detector.
