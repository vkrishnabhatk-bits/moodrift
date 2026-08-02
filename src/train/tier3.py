"""Tier 3: fine-tuned DistilRoBERTa, the only tier that needs a GPU.

Written to run in a free Colab or Kaggle T4 session, not on the laptop that runs the rest
of the pipeline. It reads the same DVC-versioned splits and logs to the same MLflow
experiment, so its run sits directly alongside tiers 1 and 2 in the comparison.

Deliberately plain PyTorch rather than ``Trainer``: the loop is short, it avoids pulling
``accelerate``/``datasets`` into the serving image, and every choice that matters here
(class weighting, truncation length, the label mapping) stays visible instead of hidden
behind arguments.

Two details that are easy to get wrong:

* **Labels are shifted.** Stars are 1-5; the model needs 0-4. The mapping is applied on
  the way in and reversed on the way out, so every metric is reported in stars.
* **Class weights come from the training split only**, matching how tiers 1 and 2 handle
  the same imbalance (ADR-0001). Computing them over the whole dataset would leak.

Run with ``python -m src.train.tier3`` on a GPU box, or paste the equivalent into a
notebook cell after ``dvc pull``.
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import REPO_ROOT, load_config
from src.evaluate.metrics import bootstrap_ci, compute_metrics
from src.features.sample import LABEL, MODEL_INPUT, load_split
from src.provenance import set_seeds
from src.train import registry

MODEL_ARTIFACT = "model"


class ReviewDataset(Dataset):
    """Tokenised reviews with zero-indexed labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        # Stars (1-5) -> class indices (0-4).
        self.labels = torch.tensor([label - 1 for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _predict(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted_stars, max_probability)``."""
    model.eval()
    predictions, confidences = [], []
    for batch in loader:
        labels = batch.pop("labels")  # noqa: F841 - not needed for inference
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)
        top = probabilities.max(dim=-1)
        predictions.append(top.indices.cpu().numpy() + 1)  # back to stars
        confidences.append(top.values.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(confidences)


def train() -> dict[str, Any]:
    """Fine-tune, evaluate and log tier 3."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = load_config("model_tier3")
    data_cfg = load_config("data")
    set_seeds(int(cfg["seed"]))

    device = _select_device()
    if device == "cpu":
        print("[tier3] WARNING: no GPU detected. This will take hours - run on Colab/Kaggle instead.")
    print(f"[tier3] device: {device}")

    train_df, val_df, test_df = (load_split(s) for s in ("train", "val", "test"))
    if cfg.get("subsample"):
        train_df = train_df.sample(n=min(int(cfg["subsample"]), len(train_df)), random_state=cfg["seed"])
        print(f"[tier3] subsampled training set to {len(train_df):,} rows")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    max_length = int(cfg["max_length"])
    loaders = {
        name: DataLoader(
            ReviewDataset(frame[MODEL_INPUT].tolist(), frame[LABEL].tolist(), tokenizer, max_length),
            batch_size=int(cfg["batch_size"]),
            shuffle=(name == "train"),
        )
        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=int(cfg["num_labels"])
    ).to(device)

    # Same imbalance strategy as tiers 1 and 2, expressed as loss weights.
    counts = train_df[LABEL].value_counts().sort_index().to_numpy()
    weights = torch.tensor(len(train_df) / (len(counts) * counts), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    print(f"[tier3] class weights: {weights.cpu().numpy().round(3).tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    epochs = int(cfg["epochs"])
    total_steps = len(loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps
    )

    with registry.start_run("tier3_distilroberta", tags={"tier": "3", "family": "transformer"}) as run:
        registry.log_flat_params(cfg)
        registry.log_data_provenance(data_cfg["paths"])
        mlflow.log_params({"device": device, "train_rows": len(train_df)})

        started = time.time()
        for epoch in range(epochs):
            model.train()
            running = 0.0
            for step, batch in enumerate(loaders["train"], start=1):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = loss_fn(model(**batch).logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                running += loss.item()
                if step % 100 == 0:
                    print(f"[tier3] epoch {epoch + 1} step {step}/{len(loaders['train'])} loss={running / step:.4f}")
            mlflow.log_metric("train_loss", running / len(loaders["train"]), step=epoch)

        mlflow.log_metric("train_seconds", time.time() - started)

        results = {}
        for name, frame in (("val", val_df), ("test", test_df)):
            y_pred, confidence = _predict(model, loaders[name], device)
            y_true = frame[LABEL].to_numpy()

            metrics = compute_metrics(y_true, y_pred)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            results[name] = metrics

            if name == "test":
                low, high = bootstrap_ci(y_true, y_pred, seed=int(cfg["seed"]))
                mlflow.log_metrics({"test_macro_f1_ci_low": low, "test_macro_f1_ci_high": high})

            print(
                f"[tier3] {name}: macro_f1={metrics['macro_f1']:.4f} "
                f"macro_mae={metrics['macro_mae']:.4f} acc={metrics['accuracy']:.4f}"
            )

        # Persist to disk BEFORE touching MLflow. Logging has now failed twice after the
        # expensive work was already done - once on tier 2, once here - and on a 52-minute
        # fine-tune that means losing the weights entirely because they only existed in
        # this process. Saving first turns a logging bug into an inconvenience.
        checkpoint = REPO_ROOT / "artifacts" / "tier3"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"[tier3] checkpoint saved -> {checkpoint}")

        # mlflow.transformers wants a Pipeline, a dict of components, or a path to a
        # checkpoint directory - never a bare model object. We have a directory now.
        registry.log_model(
            str(checkpoint), MODEL_ARTIFACT, flavor="transformers", task="text-classification"
        )
        results["run_id"] = run.info.run_id
        results["checkpoint"] = str(checkpoint)

    print(f"[tier3] logged run {results['run_id']}")
    return results


if __name__ == "__main__":
    train()
