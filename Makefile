# moodrift - developer entry points.
# Every target is safe to re-run; nothing here depends on manual setup steps.

PY := .venv/bin/python
PIP := uv pip install --python .venv

# DVC stage commands in dvc.yaml invoke a bare `python`, which resolves through PATH
# rather than through $(PY). Without this, `make data` runs the stages against whatever
# interpreter happens to be first on PATH - typically a system or conda python with none
# of this project's dependencies installed - and every stage fails on import.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

.DEFAULT_GOAL := help
.PHONY: help setup data ingest validate sample features train tier1 tier2 tier3 compare \
        bench register reproduce export-onnx image serve test lint format mlflow-ui \
        freeze clean-artifacts

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	uv venv --python 3.11 .venv
	$(PIP) -e ".[dev]"

# ------------------------------------------------------------------- data plane

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

# ---------------------------------------------------------- experimentation

train: tier1 tier2  ## Train and log every CPU tier

tier1:  ## Train tier 1 (TF-IDF + logistic regression)
	$(PY) -m src.train.tier1

tier2:  ## Train tier 2 (MiniLM embeddings + LightGBM)
	$(PY) -m src.train.tier2

tier3:  ## Fine-tune tier 3 (DistilRoBERTa) - needs a GPU, ~52 min on an M1 Pro
	$(PY) -m src.train.tier3

bench:  ## Measure per-tier serving latency and artifact size, logged back to MLflow
	$(PY) -m src.evaluate.benchmark

compare:  ## Regenerate docs/model_comparison.md from logged MLflow runs
	$(PY) -m src.evaluate.compare

register:  ## Register the champion in the model registry, behind the promotion gates
	$(PY) -m src.train.promote

reproduce:  ## Re-run a logged experiment and assert the metric matches: make reproduce RUN_ID=<id>
	@test -n "$(RUN_ID)" || { echo "usage: make reproduce RUN_ID=<mlflow run id>"; exit 2; }
	$(PY) -m src.train.reproduce $(RUN_ID)

mlflow-ui:  ## Browse experiments at http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ----------------------------------------------------------------------- serving

export-onnx:  ## Export @production to ONNX + INT8, measuring the quantisation accuracy delta
	$(PY) -m src.serve.export_onnx --quantize

image:  ## Build the serving image, tagged moodrift-serve:<current git SHA>
	GIT_SHA=$$(git rev-parse --short HEAD) docker compose build

serve:  ## Build (if needed) and run the serving stack, tagged with the current git SHA
	GIT_SHA=$$(git rev-parse --short HEAD) docker compose up --build

# ---------------------------------------------------------------- monitoring

simulate-drift:  ## Run the four drift scenarios end to end, write docs/drift_report.{md,html}
	$(PY) -m src.monitor.simulate

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
	# Excludes the "-e file:///..." self-install line `uv pip freeze` adds for this
	# project's own editable install: an absolute path to this machine's checkout, which
	# breaks `pip install -r requirements.lock.txt` on any other machine (Docker's build
	# included) - the file is meant to pin *dependencies*, not reference itself.
	uv pip freeze --python .venv | grep -v '^-e ' > requirements.lock.txt

clean-artifacts:  ## Remove local run artifacts (keeps data/ and the feature store)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
