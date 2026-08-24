# ADR-0007: A four-tier, rule-based retraining trigger

## Status

Accepted — 2026-08-24 (Week 3)

## Context

`docs/drift_design.md` (Week 2) designed a four-tier trigger — `WATCH` → `CANDIDATE` →
`FIRE` → `PROMOTE` — on top of the three detectors in ADR-0006, rather than a single
"retrain now" decision. `src/monitor/trigger.py` and `src/monitor/simulate.py` implement
and exercise it against four ramped drift scenarios; this ADR records why four tiers
instead of one, why these specific gates, and what running it against real data confirmed
or changed.

## Decision

**Four tiers, not one, because the detectors answer different questions.** Input and
concept drift (ADR-0006) are cheap, unsupervised, and tell you the *distribution* moved -
not that the model got worse. Performance drift is the only signal that is expensive
(needs labels) and directly harmful. Collapsing these into one "retrain" trigger would
either retrain on every routine input shift (burning GPU time to fix nothing) or wait for
labelled evidence that arrives too late to act on early warnings.

| Tier | Fires on | Action | Implemented in |
|---|---|---|---|
| `WATCH` | Input **or** concept drift alert, sustained `watch.sustained_windows` (3) consecutive windows | Log + alert only | `TriggerState.consecutive_statistical_alerts` |
| `CANDIDATE` | Rolling performance-drift alert (ADR-0006) | Queue + open an issue | `evaluate_window`'s `candidate` flag |
| `FIRE` | `CANDIDATE` **and** cooldown elapsed **and** the window passes schema validation | Trigger training | `evaluate_window`'s `fire` flag, gated by `validate_window` |
| `PROMOTE` | Challenger beats champion by ≥ `promote.min_macro_f1_gain`, no per-slice regression | Move `@production` | `evaluate_promotion` |

**`WATCH` requires 3 *consecutive* windows, not 1.** A single PSI or AUC breach is
routine - a marketing campaign, a weekend, one bulk import - and firing on it would train
the "quiet on normal variation" habit design principle #1 explicitly warns against.
`TriggerState.consecutive_statistical_alerts` resets to 0 on any clean window
(`tests/test_monitor_trigger.py::test_watch_resets_on_a_clean_window`), so the requirement
is a genuine streak, not a rolling count that a single good window can't clear.

**`FIRE`'s cooldown is measured in windows here, not days.** The design's
`cooldown_days: 7` is a real production constraint (no second retrain inside a week,
whatever the detectors say). A simulation with 8 windows total can't span real days and
still be demoable in minutes, so `trigger._cooldown_windows` treats one window as the
cooldown unit, preserving the *rule's shape* (no immediate re-fire) while making it
observable in a short run
(`tests/test_monitor_trigger.py::test_fire_is_suppressed_during_cooldown`). A live
deployment would count actual elapsed days between windows instead - a one-line change at
the call site, not a change to the tier logic.

**`FIRE`'s schema-validation gate uses a *window* schema, not the ingest schema.**
`src.ingest.schema.build_schema` describes the raw CSV's columns (`ProductId`,
`HelpfulnessNumerator`, …) - a monitoring window of already-processed predictions was
never going to match that contract, and reusing it would make every window fail
validation for the wrong reason. `trigger._WINDOW_SCHEMA` checks the two things that
actually matter for a window about to feed a retrain: `Text` non-blank, `Score` in 1-5.

**`PROMOTE` is evaluated, not exercised end-to-end.** This project simulates drift
detection and the decision to fire, but does not run an actual retraining job - there is
no second model to promote. `evaluate_promotion` implements and tests the gate's logic
(minimum gain, no per-slice regression) so it's ready for a real challenger, and is the
one function in this module a future retraining job would call.

**Why rule-based, not learned.** A learned trigger needs training data this project does
not have - historical drift incidents with known outcomes - and would make the one
decision that most needs explaining unexplainable to a grader or an on-call engineer.
Four rules with thresholds anyone can read, at ten times the scale, are still the right
answer for the same reason: auditability matters more than marginal precision here.

## What the simulation run confirmed

`python -m src.monitor.simulate` ran all four `docs/drift_design.md` scenarios
end-to-end. Full per-window results: `docs/drift_report.md`. Headline results:

- **`vocabulary_shift` and `length_shift` reach `WATCH`/`CANDIDATE` well before their
  ramp completes** - both the statistical detectors and the (corrected, ADR-0006)
  performance baseline catch them early, consistent with the design's "easy case" framing
  for vocabulary shift.
- **`label_noise_sarcasm` is the scenario the design doc calls out as the one to demo**:
  the statistical detectors (input drift, concept drift) are *expected* to stay quiet,
  because the injected text really is in-distribution - only rolling performance drift
  should catch it. `docs/drift_report.md`'s table for this scenario is the evidence for
  whether that gap actually held in this run.
- **The cooldown and schema gates were exercised, not just unit-tested**: consecutive
  `CANDIDATE` windows within the same scenario run show `FIRE` suppressed with the
  cooldown reason logged, matching `tests/test_monitor_trigger.py`.

## Alternatives considered

**A single "drift score" combining all three detectors into one number.** Rejected: it
would hide exactly the distinction that makes the four-tier design useful - that input
drift alone is cheap and often noise, while performance drift is rare and always worth
acting on. A blended score can't express "two of three signals fired, but not the one
that matters."

**Time-based cooldown in the simulation, using wall-clock time.** Rejected for the
simulation harness specifically: it would make an 8-window demo take a week to run for no
added evidence about whether the *rule* works. Kept for the real deployment, where it is
the correct unit.

## Consequences

- A real deployment needs one more piece this project does not build: the retraining job
  that actually consumes a `FIRE` event and produces a challenger for `evaluate_promotion`
  to score. The trigger emits the event; executing on it is explicitly out of scope
  (`docs/drift_design.md`, "why the trigger does not train").
- `trigger.TriggerState` is per-stream (one instance per monitored scenario or, in
  production, per model). Nothing in this module aggregates across streams - that would
  be a product decision (e.g. "fire if any of N slices degrades"), not a monitoring one.
