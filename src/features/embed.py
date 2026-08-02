"""Sentence embeddings (tier 2 features) and the feature-store write path.

The encoder is frozen - we never fine-tune it here. Embedding is the expensive part of
the pipeline, so it happens exactly once, in this stage, and the vectors land in the
feature store for both training and serving to reuse.

``encode`` is the shared definition of "turn text into a vector". The batch stage below
and the online read-through path in Week 3 both call it, so a cache miss at serving time
produces a vector identical to the one training would have used.

Run with ``python -m src.features.embed``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.features.store import FeatureStore
from src.provenance import file_digest, git_sha, set_seeds


def _select_device() -> str:
    """Prefer Apple GPU / CUDA when present; embedding on CPU is several times slower."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=2)
def get_encoder(model_name: str, max_seq_length: int, device: str | None = None):
    """Load the sentence-transformer once per process."""
    from sentence_transformers import SentenceTransformer

    device = device or _select_device()
    print(f"[embed] loading {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def encode(texts: list[str], cfg: dict[str, Any] | None = None, show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts. The single definition of tier-2 features."""
    cfg = cfg or load_config("model_tier2")
    emb = cfg["embedding"]
    model = get_encoder(emb["model"], int(emb["max_seq_length"]))
    vectors = model.encode(
        texts,
        batch_size=int(emb["batch_size"]),
        convert_to_numpy=True,
        normalize_embeddings=bool(emb["normalize"]),
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def populate_store() -> dict[str, Any]:
    """Embed every split and write the vectors into the feature store."""
    data_cfg = load_config("data")
    tier2_cfg = load_config("model_tier2")
    emb = tier2_cfg["embedding"]
    set_seeds(int(tier2_cfg["seed"]))

    store = FeatureStore(model=emb["model"], dimension=int(emb["dimension"]))

    total_written = 0
    per_split: dict[str, int] = {}
    for split in ("train", "val", "test", "reference"):
        path = resolve(data_cfg["paths"][split])
        df = pd.read_parquet(path, columns=["text_key", "model_input"])

        # Deduplicate within the split: identical normalised text shares one vector.
        df = df.drop_duplicates(subset="text_key")

        # Skip anything already cached, so re-running the stage is cheap.
        cached = store.read_many(df["text_key"].tolist())
        todo = df[~df["text_key"].isin(cached.keys())]
        if todo.empty:
            print(f"[embed] {split:<9} all {len(df):,} vectors already cached")
            per_split[split] = 0
            continue

        print(f"[embed] {split:<9} encoding {len(todo):,} texts ({len(cached):,} cached)")
        vectors = encode(todo["model_input"].tolist(), tier2_cfg, show_progress=True)
        written = store.write_many(todo["text_key"].tolist(), vectors, source="batch")
        per_split[split] = written
        total_written += written

    store.set_metadata(
        model=emb["model"],
        dimension=emb["dimension"],
        normalize=emb["normalize"],
        max_seq_length=emb["max_seq_length"],
        git_sha=git_sha(),
        train_data_sha256=file_digest(resolve(data_cfg["paths"]["train"])),
    )

    stats = store.stats()
    stats["written_this_run"] = total_written
    stats["per_split"] = per_split
    print(f"[embed] feature store now holds {stats['rows']:,} vectors ({total_written:,} new)")
    return stats


if __name__ == "__main__":
    populate_store()
