# ADR-0003: A SQLite key-value feature store, deliberately not a platform

## Status

Accepted — 2026-08-02 (Week 1)

## Context

A feature store is a required component for all project flavors. The instruction was
explicit about the intent: a *very simple* store — key-value, SQLite — to get hands-on
experience with writing features from a batch job and reading them back at inference time.

The engineering risk here runs in the opposite direction from most: not that the component
is too weak, but that it grows. Feast, Redis-backed online stores, TTL and eviction
policies, point-in-time correctness — all defensible in a system with many models and many
teams, all pure cost in a three-week single-model project.

## Decision

A single SQLite file, `data/feature_store/features.db`, wrapped by `src/features/store.py`.

Schema:

```sql
features(text_key PRIMARY KEY, model, dimension, vector BLOB, source, created_at)
metadata(key PRIMARY KEY, value)
```

- **Key**: SHA-256 of the *normalised* model input (`src.provenance.text_key`).
- **Value**: the 384-dimensional MiniLM embedding, stored as raw float32 bytes.
- **`source`**: `batch` for rows written by the Week 1 pipeline, `online` for rows written
  back by the Week 3 serving read-through path.
- **Journal mode**: WAL, so the serving process can read while a batch job writes.

Only the sentence embedding is stored. The tier-1 TF-IDF vector is *not*: it is cheap to
recompute from the persisted vectoriser and would be far larger than the text it came from.

## Alternatives considered

**Feast.** Rejected. It brings a registry, an offline/online split and a materialisation
step — infrastructure that solves feature reuse across many models and teams. This project
has one model family and one pipeline.

**Redis or another server.** Rejected: another service to run, another failure mode in the
Week 3 compose stack, and no benefit at this data size.

**Parquet or npy files on disk.** Simpler to write, but there is no key-based lookup, which
is precisely the online read path the exercise is about. A dictionary in a file is not a
store.

**Storing vectors as JSON or text.** Rejected: larger and lossy at the float boundary. Raw
float32 bytes round-trip exactly, which `tests/test_store.py` asserts.

## Consequences

- Training and serving read features through one code path, which is what makes
  train/serve skew structurally hard rather than merely unlikely.
- Re-running the embed stage is cheap: already-cached keys are skipped. Measured on the
  re-run after ADR-0005, only 10,904 of 65,866 vectors needed recomputation.
- The database is DVC-tracked, so a given data version carries its features with it.
- The store is a **cache, not a source of truth**: a cache miss at serving time must
  recompute using the same `encode()` function the batch path used, never a different one.
  This is stated in the module docstring because it is the one way this design can fail.
- It will not scale to millions of rows or concurrent writers, and it is not meant to.
  The risk register in PROJECT_PLAN.md names scope creep here as a tracked risk.
