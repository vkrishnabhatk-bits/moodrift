# Serving API contract

**Status:** designed in Week 2, implemented in Week 3. Nothing here is live yet.
The schemas exist as code in `src/serve/schemas.py`; this document is the prose version
and the Week 3 test checklist. Where the two disagree, the code is right.

The point of writing this before the implementation is that the expensive parts of an API
are the decisions, not the typing: what an empty string returns, whether oversize is 413 or
422, whether `/health` touches the model. Deciding those under time pressure in Week 3,
with the monitoring work running in parallel, is how contracts end up inconsistent.

## Boundary rules

- The serving container **never trains and never reads the raw dataset**. It loads one
  model artifact plus `conf/serve.yaml`.
- `src/serve/` never imports from `src/train/` or `src/ingest/`.
- The model is resolved from an **explicit alias** — `models:/moodrift-classifier@production`
  — never `latest`. Until Week 3's smoke and load tests move `@production`, that URI does
  not resolve and `/ready` returns 503. That is the intended behaviour: an unproven model
  should not become servable just by being the most recent thing trained.
- Every response carries the model version, run ID and git SHA that produced it.

## Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| POST | `/predict` | Rate one review | 200 |
| POST | `/predict/batch` | Rate up to 64 reviews | 200 |
| GET | `/health` | Liveness — process is up | 200 |
| GET | `/ready` | Readiness — model loaded, feature store reachable | 200 / 503 |
| GET | `/model/info` | What is running: version, alias, data hash, feature-store stats | 200 |
| GET | `/metrics` | Prometheus exposition | 200 |

`/health` deliberately does **not** touch the model. A liveness probe that fails while the
model reloads would have the orchestrator restart a process that was about to be fine;
that distinction is exactly why `/ready` is a separate endpoint.

### POST /predict

```json
{ "text": "Arrived stale and the seal was broken.", "request_id": "optional-client-id" }
```

```json
{
  "request_id": "0f2c…",
  "prediction": {
    "stars": 1,
    "confidence": 0.945,
    "probabilities": {"1": 0.945, "2": 0.031, "3": 0.012, "4": 0.007, "5": 0.005},
    "flags": {"truncated": false, "low_signal": false, "out_of_domain": false},
    "feature_source": "live",
    "text_hash": "9f86d0818…"
  },
  "model": {
    "name": "moodrift-classifier", "version": "1", "alias": "production",
    "run_id": "cdfd6b64221e4244a0e9d0cbca21f76b", "git_sha": "542ed28…"
  },
  "latency_ms": 41.8
}
```

### POST /predict/batch

`{"texts": [...]}` → `{"request_id", "count", "predictions": [...], "model", "latency_ms"}`.
Per-item `flags` are per item; the batch fails as a whole only on validation or oversize.

### GET /model/info

Returns the `model` block above plus `tier`, the DVC `data_hash` of the splits it was
trained on, the running `image_tag` (`moodrift-serve:<git-sha>`), `feature_store`
(`rows`, `by_source`, `last_write`) and `runtime` (`onnx`, `quantised`, `threads`).

Those five identifiers — registry version, run ID, git SHA, data hash, image tag — are
the whole traceability story in one response: from a prediction back to the commit, the
data version and the image that produced it.

## Status codes

| Code | When | Body |
|---|---|---|
| 200 | Success, including flagged-but-served inputs | response schema |
| 413 | Batch > 64 items, or a single text > 20,000 characters | `ErrorResponse` |
| 422 | Empty/whitespace-only text, wrong types, missing fields | FastAPI validation error |
| 503 | `/ready` when the model is not loaded or the feature store is unreachable | `ReadyResponse` |
| 500 | Unhandled error — logged with the request ID, never returned to the client verbatim | `ErrorResponse` |

**Why 413 and not 422 for oversize.** A batch of 65 well-formed texts is not malformed —
it is too big, which is what 413 means. Pydantic can only produce 422, so the caps live in
`conf/serve.yaml` and are enforced in the route via `schemas.oversized_batch` /
`schemas.oversized_text` rather than as field constraints.

## Edge cases

This table is the Week 3 test suite. Every row becomes a test and a Postman request.

| Input | Behaviour | Rationale |
|---|---|---|
| `""` or `"   "` | 422 | There is no honest rating for empty text, and a fabricated one would poison the drift detectors. |
| Text longer than the model's 256-token window | 200, `truncated: true` | The tokeniser truncates; the flag reports what actually happened rather than guessing from a character count. |
| Text > 20,000 characters | 413 | Matches `validation.max_text_chars` in `conf/data.yaml`, so the API and the training pipeline agree on what a review is. Also bounds memory. |
| Batch > 64 items | 413 | Cap from `conf/serve.yaml`. |
| Emoji-only, punctuation-only, bare URL | 200, `low_signal: true` | Served, not refused: a flag surfaces the case to the caller and to monitoring; a refusal would hide it. |
| Non-English text | 200, `out_of_domain: true` | English-only corpus (ADR-0002). Same tier-0 detector the sample stage used, so "out of domain" means the same thing in both places. |
| Control characters, zero-width joiners, invalid UTF-8 sequences | 200, sanitised before inference | Sanitising uses the same normalisation as training — a second, subtly different cleaner is exactly the train/serve skew the feature store exists to prevent. |
| Model not loaded yet | `/ready` 503, `/predict` 503 | Readiness is not liveness. |

**Deviation from the original plan, recorded deliberately.** The plan said "text > 512
chars → truncate". That number is wrong for this model: 256 tokens is roughly 1,000–1,400
characters of English, so truncating at 512 characters would throw away text the model can
actually use, and would report `truncated` on inputs that were never truncated. The rule
here is instead: truncate at the model's real window and set the flag from what the
tokeniser did.

## Latency budget

Measured, not assumed — from `make bench` (200 real test reviews, batch=1, one CPU thread,
arm64 Darwin). Full numbers in [model_comparison.md](model_comparison.md).

| Component | p95 |
|---|---|
| Champion model, end to end (tokenise + forward pass) | 43.7 ms |
| Remaining budget for HTTP, validation, normalisation, feature-store lookup, logging | ~86 ms |
| **Target** | **< 130 ms** |

Two consequences worth stating now, because they change Week 3's priorities:

1. **ONNX + INT8 quantisation is no longer load-bearing for latency.** The Week 1 plan
   assumed it would be, on the reasonable guess that a 320 MB transformer would be slow on
   CPU. Measured, it is not: the champion has ~3x headroom against the target even
   single-threaded. Quantisation stays in scope for **artifact size** (320 MB vs tier 1's
   15 MB, which lands on the serving image), and the accuracy delta still gets measured —
   but if it has to be cut, the latency target survives.
2. **The feature store's read path is worth its complexity on tier 2, not on tier 3.**
   A cache hit takes tier 2 from 25.7 ms to 1.3 ms (20x). The champion is a fine-tuned
   transformer that does not use stored embeddings on its inference path, so on the
   champion the store is read for monitoring features rather than to skip work. Say that
   plainly in the report instead of implying a speed-up the champion does not get.

**Measured (Week 3): the INT8 accuracy delta, on 1,000 real held-out test-split rows**
(`python -m src.serve.export_onnx --quantize`, dynamic quantisation via
`onnxruntime.quantization.quantize_dynamic`, `src.evaluate.metrics.compute_metrics` — the
same function every tier's own evaluation uses). fp32 macro-F1 on this sample (0.6028) is
close to the full 9,000-row test-split number (0.6001), which is what makes it a credible
sample rather than a lucky/unlucky 1,000 rows.

| | fp32 | INT8 | Delta |
|---|---|---|---|
| Macro-F1 | 0.6028 | 0.6039 | +0.0011 |
| Macro-MAE | 0.4124 | 0.4152 | +0.0028 |
| Artifact size | 313.4 MB | 78.7 MB | −75% |
| 1,000-row batched CPU time (batch=32, 1 thread) | 361.5 s | 122.1 s | ~3x faster |

Both deltas are noise-level — nowhere near the 5-point macro-F1 drop that would trip the
`CANDIDATE` drift-trigger tier (§8, M5) — for a 4x smaller artifact and a meaningful CPU
speed-up. **Serving now defaults to the INT8 build** (`app.py` prefers
`model.int8.onnx`, falling back to the fp32 `model.onnx` if a plain export without
`--quantize` was run). Numbers are logged onto the champion's own MLflow run
(`onnx_fp32_macro_f1`, `onnx_int8_macro_f1`, etc.), not a side file, so they can't drift
out of sync with which run was actually measured.

## HTTP load test (Week 3)

`scripts/smoke_test.sh` (curl) and `postman/moodrift.postman_collection.json` cover the
happy path plus every row of the edge-case table above, run against a live
`uvicorn`/`docker compose` process. `hey` measured `/predict` and `/predict/batch`
end to end - real HTTP, not the in-process model-only benchmark the table above uses.

| Load | p50 | p95 | p99 | Notes |
|---|---|---|---|---|
| `/predict`, sequential (`hey -c 1`, n=200) | 120.7 ms | 121.7 ms | 122.4 ms | Comparable methodology to `make bench`'s single-thread number; the ~78 ms over the model's own 43.7 ms is HTTP + normalisation + feature-store lookup + flag detection + logging - within the ~86 ms budget §Latency budget above allocated for exactly that. |
| `/predict`, concurrent (`hey -c 10`, n=500) | 184.4 ms | 225.9 ms | 246.0 ms | **Exceeds the 130 ms target under concurrency.** One `uvicorn` worker, one process: concurrent requests queue for the single-threaded ONNX session and for Python's GIL rather than running in parallel. Scaling target concurrency needs more worker processes (`uvicorn --workers N`) or replicas, not a code fix - out of scope for this PR, recorded here so the number is measured rather than assumed. |
| `/predict/batch`, batch=8, concurrent (`hey -c 5`, n=100) | 984 ms | 1,018 ms | 1,037 ms | **Resolved below, with a caveat** — the quantized model this table otherwise measures still loops (same numbers apply); a real shared forward pass now exists for the fp32 graph but was measured to not actually help. See "Batch endpoint: what shipped and why" below. |

**A concurrency bug found and fixed while running this test, not before it.** The first
`hey -c 10` run threw sporadic `500`s (36 of 500) - `RuntimeError: Already borrowed`
inside `tokenizers`. HuggingFace's fast (Rust) tokenizer mutates shared internal state
on every call to configure truncation/padding, which is not safe for concurrent callers
on one instance; FastAPI runs sync routes in a thread pool, so concurrent `/predict`
calls raced on the single module-level `tokenizer`. Fixed with a `threading.Lock` around
tokenisation only (`src/serve/app.py`, `_tokenize`) - microseconds of critical section,
not the ONNX forward pass, which dominates latency and is safe to run concurrently
(`onnxruntime.InferenceSession.run` is thread-safe). Re-run after the fix: 0 errors
across 500 requests. This is exactly why the load test scope in §9 of `PROJECT_PLAN.md`
exists as its own line item rather than being assumed from the model-only benchmark.

## Batch endpoint: what shipped and why

Closes the "Open for Week 3" note below with a measurement, not the fix that note
assumed. `/predict/batch` now shares one tokenisation call across the batch either way
(`_tokenize_batch`), and shares one ONNX forward pass too - but only on the fp32 graph.

**Why not on the quantized graph, which is what's actually served (`@production` is
INT8):** ONNX Runtime's dynamic quantization recomputes its activation scale from
whatever tensor it's given, so a batch-of-N forward pass is not numerically equivalent to
N batch-of-1 calls the way it is on fp32. Measured directly (bypassing HTTP, same ONNX
session, same 200 held-out test rows): batching flipped the predicted star on **5% of
them** versus the single-item path. That's a real accuracy risk, not a rounding
difference, so `_looped_logits` keeps the quantized path looping - `/predict/batch`
always agrees with `/predict` on identical text, verified by
`tests/test_serve.py::test_predict_batch_matches_single_item_predictions`.

**A second, independent finding: even where batching is numerically safe (fp32), it
doesn't help.** Measured directly against the fp32 ONNX session, same texts, same
process, warm: 8 looped single-item calls average 2744 ms; one real batch-of-8 call
averages 2884 ms — batching is marginally *slower*, not faster. `INTRA_OP_THREADS=1` is
deliberate (see `_TOKENIZER_LOCK`'s comment on concurrent-request safety), so the forward
pass is compute-bound on one thread either way; there's no per-call Python/session
overhead here large enough for sharing one call to recover. The original "Open for Week
3" note assumed batching was primarily an overhead win. Measured, it isn't, for this
model at this thread count.

**Net effect:** the batch endpoint's correctness is now solid and tested (INT8 and fp32
predictions match `/predict` exactly, always). Its latency for the model actually in
production is **unchanged** — still ~120 ms × N, because looping is the only numerically
safe option for that graph, and batching wasn't going to be faster even if it weren't.
Multi-worker `uvicorn` was tried as an alternative lever for concurrency (see the load
test table above) and didn't help on the hardware it was measured on either. The
remaining real levers - static/calibrated INT8 quantization (removes the batch-invariance
problem) or `intra_op_num_threads > 1` (trades single-request latency for batch
throughput) - are both out of scope for this pass and are documented as open, not
attempted-and-hidden.

## Feature store on the request path

1. Normalise the input with the same code the batch pipeline used (`src/features/clean.py`).
2. Hash it (`src.provenance.text_key`) → the feature-store key, returned as `text_hash`.
3. Look it up in `data/feature_store/features.db`. Hit → `feature_source: "store"`.
4. Miss → compute the embedding live with the same transform, write it back with
   `source='online'`, return `feature_source: "live"`. The second identical request hits.

`GET /model/info` reports the store's row count, per-source breakdown and last-write time,
so a grader can watch `by_source.online` increment as they send requests.

## Prediction log

JSONL via structlog, written on the request path from Week 3 — not deferred to the
monitoring work, so the drift detectors have real accumulated data the day they are built.

One line per prediction: `timestamp`, `request_id`, `text_hash`, `token_count`,
`char_count`, `stars`, `confidence`, `probabilities`, `flags`, `feature_source`,
`model_version`, `run_id`, `git_sha`, `latency_ms`.

Raw text is **not** logged by default (`logging.log_raw_text: false`). The hash is enough
to detect repeats and to join back to the feature store, and the log ships with the repo
during evaluation — there is no reason for user text to travel with it. The flag exists
because drift debugging occasionally needs the text, and a flag that is documented is
better than someone adding a `print` later.

## Open for Week 3

- ~~Whether `/predict/batch` shares one tokenisation pass or loops the single path.~~
  **Resolved, with a finding that changed the plan:** it now shares tokenisation
  unconditionally and shares the ONNX forward pass on fp32 — but batching turned out not
  to be a latency win at all (see "Batch endpoint: what shipped and why" above), and is
  actively unsafe on the quantized graph actually served (5% prediction disagreement,
  measured), so that graph still loops. The endpoint's correctness improved; its latency
  for `@production` did not, and there's no cheap further lever within this pass's scope.
- ~~Whether the ONNX export replaces the transformers pipeline outright or sits behind a
  `runtime.onnx` flag with the pipeline as fallback.~~ **Resolved:** ONNX replaces the
  pipeline outright; no fallback needed. The concern was quantisation degrading accuracy
  past a gate, and it measured at noise-level (see the INT8 table above) — a fallback
  code path defending against a risk that didn't materialise is exactly the kind of
  gold-plating §1's cut-order principle warns against.
- **New, opened by this pass:** static/calibrated INT8 quantization would remove the
  batch-invariance problem above and let the served model actually benefit from batching.
  Not attempted here — it needs a calibration dataset, a re-export, and a fresh accuracy-
  delta measurement against the existing "+0.0011 macro-F1, noise-level" dynamic-INT8
  number. Left open rather than rushed.
- **New, opened by this pass:** concurrent-load latency (p95 226 ms vs. the 130 ms target,
  §HTTP load test above) is still open. Multi-worker `uvicorn --workers 4` was tried as
  the obvious fix and measured *no better* (p95 238 ms) on the machine it was tested on —
  each worker's own MPS-backed feature-store encoder warmup is a plausible contention
  source, not confirmed. Scaling this needs real profiling, not another guess.
