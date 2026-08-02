# moodrift

Predicts a 1–5 star rating from free-text product reviews, built as a full MLOps pipeline:
ingest → validate → version → train/compare → serve → monitor → retrain-trigger.

> **Status: Week 1 of 4 complete.** The data plane and CPU baselines work end to end.
> Serving (FastAPI/ONNX) and monitoring (drift detection, retraining triggers) are designed
> but not yet built — see the roadmap below. This README is deliberately brief and gets
> finalised in Week 4.

## Current results

Held-out test split, 9,000 reviews. Macro-F1 is the headline metric, not accuracy: ~64% of
this corpus is 5-star, so predicting 5 unconditionally already scores 0.64.

| Tier | Model | Macro-F1 | Macro-MAE | Within 1 star |
|---|---|---|---|---|
| 1 | TF-IDF + logistic regression | **0.5137** | **0.628** | 92.5% |
| 2 | MiniLM embeddings + LightGBM | 0.4292 | 1.000 | 88.4% |
| 3 | Fine-tuned DistilRoBERTa | *pending* | | |

Tier 1 is the current champion — the cheap model is winning. Full breakdown in
[`docs/model_comparison.md`](docs/model_comparison.md).

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). The raw corpus downloads
automatically (Stanford SNAP, no credentials needed).

```bash
make setup      # create .venv and install dependencies
make data       # run the full data pipeline via DVC (~10 min, downloads 122MB)
make train      # train and log tiers 1 and 2 to MLflow
make compare    # regenerate the model comparison report
make mlflow-ui  # browse experiments at http://127.0.0.1:5000
```

`make help` lists every target. Tier 3 needs a GPU: `python -m src.train.tier3`.

## How it fits together

Four planes, each runnable on its own. Only the first two exist today.

```
Data plane (M2)          raw → validate → sample → feature store → DVC tag      ✅ built
Experiment plane (M3)    config → train tiers → evaluate → MLflow → registry    ✅ built
Serving plane (M4)       registry → ONNX → FastAPI → Docker → /predict          ⬜ week 3
Observability plane (M5) prediction log → drift → trigger → retrain             ⬜ week 3
```

Text normalisation lives in one place (`src/features/clean.py`) and is shared by training and
serving, and features are cached in a SQLite feature store keyed by a hash of the normalised
text — together these are what keep train/serve skew structurally hard rather than merely
unlikely.

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

- English only, enforced at ingestion.
- Trained on food-product reviews from 2012 and earlier; language and products have moved on.
- No sarcasm handling — the worst misclassifications are dominated by it.
- Label noise is real: some 5-star reviews describe delivery problems rather than the product.

## Roadmap

| Week | Scope | State |
|---|---|---|
| 1 | Data plane, versioning, CPU baselines | ✅ complete (`week1-data`) |
| 2 | Comparison report, model registry, `make reproduce` | next |
| 3 | FastAPI serving + drift monitoring and retraining triggers | planned |
| 4 | Documentation, hardening, demo | planned |
