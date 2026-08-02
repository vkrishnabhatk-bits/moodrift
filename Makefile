# moodrift - developer entry points.
# Every target is safe to re-run; nothing here depends on manual setup steps.

PY := .venv/bin/python
PIP := uv pip install --python .venv

.DEFAULT_GOAL := help
.PHONY: help setup data ingest validate sample features train tier1 tier2 compare \
        test lint format mlflow-ui freeze clean-artifacts

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	uv venv --python 3.11 .venv
	$(PIP) -e ".[dev]"

# ---------------------------------------------------------------- data plane (M2)

data:  ## Rebuild the whole data plane via DVC (only what changed)
	$(PY) -m dvc repro

ingest:  ## Download and parse the raw corpus
	$(PY) -m src.ingest.loader

validate:  ## Schema-validate and quarantine bad rows
	$(PY) -m src.ingest.schema

sample:  ## Stratified sample, language filter, splits, reference window
	$(PY) -m src.features.sample

features:  ## Embed texts and populate the feature store
	$(PY) -m src.features.embed

# ------------------------------------------------------- experimentation (M3)

train: tier1 tier2  ## Train and log every CPU tier

tier1:  ## Train tier 1 (TF-IDF + logistic regression)
	$(PY) -m src.train.tier1

tier2:  ## Train tier 2 (MiniLM embeddings + LightGBM)
	$(PY) -m src.train.tier2

compare:  ## Regenerate docs/model_comparison.md from logged MLflow runs
	$(PY) -m src.evaluate.compare

mlflow-ui:  ## Browse experiments at http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ------------------------------------------------------------------- quality

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint and type-check
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

format:  ## Auto-format and fix lint errors
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

freeze:  ## Pin the resolved dependency set into requirements.lock.txt
	uv pip freeze --python .venv > requirements.lock.txt

clean-artifacts:  ## Remove local run artifacts (keeps data/ and the feature store)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
