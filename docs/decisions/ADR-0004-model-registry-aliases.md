# ADR-0004: Registry promotion via aliases, not stages

## Status

Accepted — 2026-08-02 (Week 1)

## Context

PROJECT_PLAN.md §3 specifies a gated promotion path through the MLflow Model Registry
using **stages**, with serving pinned to `models:/moodrift-classifier/Production`:

- `None → Staging` on passing the M3 evaluation thresholds
- `Staging → Production` on passing the Week 3 smoke and load tests
- `Production → Archived` automatically when a newer version is promoted

The installed MLflow is **3.15.0**. MLflow deprecated model-version *stages* in 2.9 and
removed them from the recommended path in 3.x, replacing them with **aliases**: named,
movable pointers to a specific version. `MlflowClient.transition_model_version_stage`
still exists but is deprecated and slated for removal.

A second constraint surfaced at the same time: the Model Registry requires a
*database-backed* tracking store. The default `mlruns/` file store cannot register a model
at all.

## Decision

**Use aliases**, keeping the plan's gating logic byte-for-byte and changing only the
mechanism.

| Plan (stages) | Implemented (aliases) |
|---|---|
| `Staging` | `@candidate` |
| `Production` | `@production` |
| best-by-comparison | `@champion` |
| `Archived` | implicit — an alias moves, old versions remain queryable |

Serving resolves `models:/moodrift-classifier@production` (`registry.model_uri()`), never
`latest`, so the version in production is always an explicit, auditable choice.

**Tracking store**: SQLite at `mlflow.db`, with artifacts in `mlruns/`. Both are
git-ignored; the Week 3 compose stack runs the same configuration as a service.

## Alternatives considered

**Pin MLflow to 2.x to keep stages as written.** Rejected: it freezes the project on a
line that is already deprecated, to preserve a mechanism whose replacement is strictly
better. Aliases can move freely and are not restricted to four hard-coded names.

**Use stages anyway via the deprecated API.** Rejected: it emits deprecation warnings on
every promotion and will break on a future upgrade, for no benefit.

**Keep the file store and skip the registry.** Not possible — the registry is a graded M3
requirement and the file store does not support it.

## Consequences

- Rollback is a one-line operation: point `@production` at the previous version. Nothing is
  deleted, so the previous champion stays queryable — the same guarantee "Archived" gave.
- Aliases carry no implicit ordering, so the *gates* must be enforced in code rather than
  inherited from stage semantics. They are, in `src/train/registry.py`.
- PROJECT_PLAN.md §3 and the Week 3 serving section still say `/Production`; they should be
  read as `@production`. Worth correcting in the plan when the team next reviews it.
- Anyone reproducing this on MLflow 2.x will find aliases available there too (2.9+), so
  the code is not tied to 3.x.
