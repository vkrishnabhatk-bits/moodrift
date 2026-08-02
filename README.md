# moodrift

Predicts a 1–5 star rating from free-text product reviews, built as a full MLOps pipeline:
ingest → validate → version → train/compare → serve → monitor → retrain-trigger.

> **Status: Weeks 1–2 of 4 complete.** The data plane, all three model tiers, the model
> registry and the reproduce command work end to end. Serving (FastAPI/ONNX) and monitoring
> (drift detection, retraining triggers) are designed — [contract](docs/api_contract.md),
> [drift design](docs/drift_design.md) — but not yet built. This README is deliberately
> brief and gets finalised in Week 4.

## Current results

Held-out test split, 9,000 reviews. Macro-F1 is the headline metric, not accuracy: ~64% of
this corpus is 5-star, so predicting 5 unconditionally already scores 0.64. Latency is
measured on 200 real reviews at batch=1 on a single CPU thread.

| Tier | Model | Macro-F1 | Macro-MAE | Within 1 star | p95 | Size |
|---|---|---|---|---|---|---|
| 1 | TF-IDF + logistic regression | 0.5137 | 0.628 | 92.5% | 1.9 ms | 15 MB |
| 2 | MiniLM embeddings + LightGBM | 0.4292 | 1.000 | 88.4% | 25.7 ms | 10 MB |
| **3** | **Fine-tuned DistilRoBERTa** | **0.6001** | **0.4226** | **96.7%** | **43.7 ms** | **320 MB** |

**Tier 3 is the champion** — registered as `moodrift-classifier` v1 `@champion` — and the
confidence intervals do not overlap, so the margin is real rather than run-to-run noise.
It costs 24× the latency and 22× the size of the baseline for 8.6 macro-F1 points, and
still fits the < 130 ms serving budget with room to spare. Two more results worth reading
twice:

- **Plain TF-IDF beat the sentence-embedding tier by 8.5 points.** "Use a transformer" is
  not automatically right; the ladder is what showed which was which.
- **The fine-tune's gain sits in the minority classes**, which is what macro-F1 exists to
  measure: 2-star F1 goes 0.288 → 0.449 and 3-star 0.386 → 0.498, while 5-star barely
  moves. Accuracy would have hidden this — tiers 1 and 2 score within 0.2 points of each
  other on it.

Full breakdown in [`docs/model_comparison.md`](docs/model_comparison.md).

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). The raw corpus downloads
automatically (Stanford SNAP, no credentials needed).

```bash
make setup      # create .venv and install dependencies
make data       # run the full data pipeline via DVC (~15 min, downloads 122MB)
make train      # train and log tiers 1 and 2 to MLflow
make bench      # measure per-tier latency and artifact size
make compare    # regenerate the model comparison report
make register   # register the champion, behind the promotion gates
make mlflow-ui  # browse experiments at http://127.0.0.1:5000
```

Any logged run can be re-run and checked:

```bash
make reproduce RUN_ID=847d5707134c4119bc39018e354335c4
```

It verifies the commit, the data hashes and the config against what the run recorded,
retrains, and asserts the metric matches within a per-tier tolerance. Both CPU tiers
reproduce **exactly** (delta 0.000000); tier 3 gets a documented tolerance band because
Apple's MPS backend is not bit-deterministic — two runs from the same seed gave 0.6017 and
0.6001, and claiming byte-identical reproduction there would be false.

`make help` lists every target.

`make data` needs no credentials and no DVC remote: it downloads the corpus from a public
URL and rebuilds every split. Verified from a clean clone — the regenerated splits are
byte-identical to the ones the published models were trained on.

Tier 3 needs a GPU and is run separately: `python -m src.train.tier3`. It auto-selects
CUDA, then Apple Silicon (MPS), then CPU — the published run took 52 minutes on an M1 Pro,
so a discrete GPU is not required.

## How it fits together

Four planes, each runnable on its own. Only the first two exist today.

```
Data plane          download → validate → sample → feature store → DVC tag ✅ built
Experiment plane    config → train tiers → evaluate → MLflow → registry    ✅ built
Serving plane       registry → ONNX → FastAPI → Docker → /predict          ⬜ week 3
Observability plane prediction log → drift → trigger → retrain             ⬜ week 3
```

Text normalisation lives in one place (`src/features/clean.py`) and is shared by training and
serving, and features are cached in a SQLite feature store keyed by a hash of the normalised
text — together these are what keep train/serve skew structurally hard rather than merely
unlikely. The store earns its place measurably: a cache hit takes tier 2's p95 from 25.7 ms
to 1.3 ms.

Promotion uses MLflow **aliases**, not the deprecated stages (ADR-0004):
`@candidate` → `@champion` → `@production`. `make register` refuses to promote a model that
fails a gate in `conf/evaluation.yaml`, and writes the passing numbers into the version
description. `@production` stays unset until the Week 3 smoke and load tests pass — serving
always resolves an explicit alias, never "latest".

## Data

[Amazon Fine Food Reviews](https://snap.stanford.edu/data/web-FineFoods.html) — 568,454
reviews. 567,140 pass validation; the rest are quarantined with a stated reason (never
dropped silently) in `data/interim/rejected/`. See
[`docs/data_quality_report.md`](docs/data_quality_report.md).

Training uses a 60,000-row proportional sample that preserves the real class imbalance;
imbalance is handled at training time with class weights rather than by reshaping the data.

## Design decisions

Non-obvious choices are recorded as ADRs in [`docs/decisions/`](docs/decisions/), including
why the sample is proportional rather than balanced, why the feature store is deliberately a
single SQLite file, and the cross-split text deduplication that removed ~13% test-set leakage.

## Known limitations

- English only. The filter runs in the sample stage rather than at ingestion, on an
  oversampled candidate pool - same end result, a fraction of the compute (ADR-0002).
- Trained on food-product reviews from 2012 and earlier; language and products have moved on.
- No sarcasm handling. It shows up clearly among the worst misclassifications: a 5-star
  review opening "made in china... so what?" is predicted 1-star at 93% confidence.
- Label noise is real: some 5-star reviews describe delivery problems rather than the product.
- Tier 2 needs PyTorch and LightGBM in one process, and their bundled OpenMP runtimes do not
  coexist on macOS/arm64: torch-first segfaults, LightGBM-first deadlocks, both silently.
  `OMP_NUM_THREADS=1` plus importing LightGBM first is the working combination, which
  `make bench` sets. The champion (tier 3) does not hit this.

## Roadmap

| Week | Scope | State |
|---|---|---|
| 1 | Data plane, versioning, all three model tiers | ✅ complete (`week1-data`) |
| 2 | Comparison report, model registry, `make reproduce`, serving + drift design | ✅ complete (`week2-experiments`) |
| 3 | FastAPI serving + drift monitoring and retraining triggers | next |
| 4 | Documentation, hardening, demo | planned |
