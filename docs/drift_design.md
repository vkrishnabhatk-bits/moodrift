# Drift detection and retraining trigger — design

**Status:** designed in Week 2, implemented in Week 3. The thresholds live in
`conf/monitor.yaml`; this document is why they have the values they do. Now implemented,
run against real data, and written up as two ADRs -
[`ADR-0006-drift-detection-approach`](decisions/ADR-0006-drift-detection-approach.md)
(what the implementation confirmed, and the one thing it corrected: the performance-drift
baseline was measured on the wrong input format) and
[`ADR-0007-retraining-trigger-design`](decisions/ADR-0007-retraining-trigger-design.md).
Results from the four scenario runs: [`drift_report.md`](drift_report.md). The open
questions below are resolved in ADR-0006; left in place as a record of what Week 2 didn't
yet know, not edited out now that Week 3 does.

## What can actually drift here

Worth being concrete before choosing detectors, because "drift" covers three unrelated
failures and the response to each is different:

| Kind | What moved | Detected by | Can it be seen without labels? |
|---|---|---|---|
| Input drift | The text arriving looks different — length, vocabulary, formatting | PSI + KS on scalar features | Yes |
| Concept drift | The relationship between text and rating moved | Weak domain classifier on embeddings | Yes |
| Performance drift | The model is simply wrong more often | Rolling macro-F1 and macro-MAE | **No** — needs a labelled slice |

The third is the one that matters and the only one that cannot run unsupervised. That
asymmetry is the whole reason the trigger has four tiers rather than one: the cheap
detectors are early warnings, and only the expensive signal justifies retraining.

## Input drift: PSI + KS

**Features:** `char_count`, `token_count`, `oov_rate` (share of tokens missing from the
tier-1 training vocabulary), `mean_word_length`. Scalar and interpretable on purpose — when
a detector fires, a histogram shows why in about ten seconds.

**PSI thresholds: 0.10 warn, 0.20 alert.** These are the conventional credit-risk lines and
are used here because they are conventional: a threshold everyone recognises needs less
defending than a bespoke one, and the simulated scenarios exist to check the convention
holds on this corpus rather than to assume it.

**KS never triggers on its own** (`require_psi_agreement: true`). With 500 samples per
window a KS test finds statistically significant differences that are practically
meaningless — that is what significance testing does at scale. PSI measures *how much* the
distribution moved; KS measures *whether* it moved. Requiring both means the alert says
"a real amount, and not by chance".

**Rejected:** per-token frequency drift over the full vocabulary. Thousands of correlated
tests, a multiple-comparisons problem to correct for, and a result nobody can read. `oov_rate`
captures the same signal in one number.

## Concept drift: a deliberately weak domain classifier

Train a logistic regression to tell reference-window embeddings from current-window
embeddings. If it cannot (AUC ≈ 0.5), the windows are interchangeable. If it can
(AUC > 0.65), they are not.

**Why weak on purpose (`C: 0.1`, 3-fold CV).** A strong model — gradient boosting on 384
dimensions — will separate almost any two samples of 500 given enough capacity, including
two random halves of the same distribution. It would fire every window and be switched off
within a day. The regularisation is not a limitation to apologise for; it is the mechanism
that makes the signal mean something.

**Why embeddings from the feature store.** The same MiniLM vectors training consumed
(`data/feature_store/features.db`), not a fresh encode. A monitoring pipeline computing its
own features is a second feature definition waiting to disagree with the first — precisely
the skew the store exists to prevent.

**AUC 0.65 for alert, 0.60 for warn.** 0.5 is chance; 0.65 is a margin large enough that a
weak, cross-validated model found real structure. Calibrated by running the detector on two
random halves of the reference window: the null-case AUC came in at 0.490, essentially
chance and 15 points below the 0.65 floor, so the floor holds without adjustment. Full
result: [ADR-0006](decisions/ADR-0006-drift-detection-approach.md), "Question 1, resolved."

## Performance drift: rolling macro-F1 and macro-MAE

**Baseline: macro-F1 0.6001, macro-MAE 0.4226** — the champion's measured test scores,
pinned in config rather than read from MLflow at runtime, so a monitoring run cannot
quietly rebaseline itself against a degraded model.

**Trigger at 5 macro-F1 points below baseline**, i.e. 0.55. That is not an arbitrary
5-point rule: 0.55 is the project's minimum acceptable macro-F1, so the trigger fires
exactly when the model stops meeting the bar it was allowed to ship at. The two numbers
agreeing is a coincidence worth keeping, because it makes the threshold explainable in one
sentence.

**Both metrics, not one.** Macro-F1 cannot distinguish "predicted 4 instead of 5" from
"predicted 1 instead of 5"; on an ordinal rating those are different failures. A model that
holds its macro-F1 while macro-MAE climbs is drifting in a way only the second metric sees.

**Minimum 500 labelled samples.** Below that, a 5-class macro-F1 swings on a handful of
minority-class rows and the "drift" is sampling noise.

## Simulation scenarios

Four scenarios, all **ramped** across 8 windows rather than applied as a step change. A step
change is trivially detectable and proves nothing; a ramp answers the question that
actually matters — *at what point* does each detector notice.

| Scenario | Injection | Ramp | Expected to fire | Point of the scenario |
|---|---|---|---|---|
| Vocabulary shift | Modern slang and emoji absent from the training-era vocabulary | 0 → 40% | PSI, KS, domain classifier | The easy case. If this does not fire, the thresholds are too loose. |
| Topic shift | Reviews from a different product category; label space stays 1–5 | 0 → 50% | Domain classifier | Subject matter moves while surface statistics barely do — the embedding detector should catch what PSI misses. |
| Length shift | Short reviews swapped for long ones | 0 → 60% | PSI, KS, **and latency** | A cross-signal: p95 latency moving with input distribution is a reminder that drift is an infrastructure event too. |
| Label noise / sarcasm | Reviews whose surface sentiment contradicts the true rating | 0 → 30% | **Performance drift only** | The important one. The statistical detectors are expected to stay silent — the inputs really are in-distribution. Only rolling macro-F1/MAE catches it. |

The fourth scenario is the one to demo. A monitoring stack that cannot show a case its own
statistical detectors miss does not understand its detectors, and the honest limitation is
worth more marks than a clean sweep.

## The four-tier trigger

| Tier | Condition | Action | Why not just retrain |
|---|---|---|---|
| `WATCH` | PSI > 0.2 **or** domain-classifier AUC > 0.65, sustained 3 windows | Log + alert | Input drift alone does not mean the model got worse. Retraining here burns GPU time to fix nothing. |
| `CANDIDATE` | Rolling macro-F1 ≥ 5 points below baseline on ≥ 500 labelled samples | Queue + open an issue | The model *is* worse. But a queue with a human in it beats an automatic retrain on data nobody has looked at. |
| `FIRE` | `CANDIDATE` holds **and** 7-day cooldown elapsed **and** the new window passes schema validation | Trigger training | The cooldown stops a flapping detector retraining nightly; the validation stops training on a corrupted feed — which is the fastest way to turn a drift incident into a bad model. |
| `PROMOTE` | Challenger beats champion by ≥ 1 macro-F1 point on the frozen holdout with no per-slice regression | Move `@production` | Retraining does not entitle a model to serve. If the challenger is not better, the champion keeps serving and the failed attempt is recorded. |

**Why rule-based and not an ML-driven policy.** A learned trigger would need training data
this project does not have (drift incidents with outcomes), and it would make the one
decision that most needs explaining unexplainable. Four rules with thresholds anyone can
read are the right answer here, and would still be at ten times the scale.

**Why the sustained-3-windows requirement.** A single window crossing PSI 0.2 is routine —
a marketing campaign, a weekend, one bulk import. Three consecutive windows is a trend.
This is the cheapest false-positive suppression available and costs only latency to alert.

**Why the trigger does not train.** It emits an event; the training job consumes it. The
decision and the execution stay separate so the decision is auditable on its own — you can
read the log and see exactly why a retrain was or was not requested, without reading
training logs.

## Open questions for Week 3

- Calibrate the domain-classifier AUC floor on two random halves of the reference window
  before trusting 0.65.
- Decide whether `oov_rate` is computed against the tier-1 TF-IDF vocabulary (cheap, fixed,
  but tied to a model that is not the champion) or a corpus vocabulary frozen at Week 1.
- Whether the labelled slice for performance drift is simulated (the held-out test split
  replayed through the API) or hand-labelled. Simulated is honest if it is *stated* to be
  simulated, which is the plan.
