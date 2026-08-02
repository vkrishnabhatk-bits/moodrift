"""Provenance helpers: git SHA, content hashes, and seeding.

Every trained model must be traceable back to the exact code and the exact data that
produced it. These three functions are what make that possible, and they are logged
with every MLflow run.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

from src.config import REPO_ROOT


def git_sha(short: bool = False) -> str:
    """Current commit SHA, suffixed with ``-dirty`` when the tree has local changes.

    Returns ``"unknown"`` outside a git checkout rather than raising - provenance is
    best-effort metadata and must never break a training run.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        sha = subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Streaming content hash of a file - the data-version half of provenance."""
    path = Path(path)
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(text: str, algorithm: str = "sha256") -> str:
    """Stable key for a piece of text - the feature store's primary key.

    Both the batch write path and the online read path call this, which is what
    guarantees a cache hit for identical text at serving time.
    """
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


def set_seeds(seed: int) -> None:
    """Seed every RNG the pipeline touches, so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
