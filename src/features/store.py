"""The feature store: a deliberately small SQLite key-value cache for embeddings.

Scope is capped on purpose (single file, no service, no TTL, no eviction). It exists to
demonstrate the offline/online feature pattern with a real read path, not to be
production infrastructure - see ADR-0003 and the risk register in PROJECT_PLAN.md.

The contract that makes it worth having:

* **One writer definition, two readers.** The batch pipeline (Week 1) populates it; both
  training and the online ``/predict`` path (Week 3) read from it.
* **The key is a hash of the *normalised* text** (:func:`src.provenance.text_key`), so an
  identical review hits the cache whether it arrives in a batch job or an HTTP request.
* **On a miss, the caller computes with the same code the batch path used** and writes
  back. Cache misses must never become a second, subtly different feature definition -
  that is exactly the train/serve skew a feature store exists to prevent.

Vectors are stored as raw float32 bytes: compact, exact (no float text rounding), and
trivially convertible back to numpy.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ensure_parent, resolve

DEFAULT_DB = "data/feature_store/features.db"
_DTYPE = np.float32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    text_key   TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    dimension  INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_features_source ON features(source);
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class FeatureStore:
    """Key-value store mapping normalised-text hashes to embedding vectors."""

    def __init__(self, path: str | Path = DEFAULT_DB, model: str = "unknown", dimension: int = 0):
        self.path = resolve(path)
        self.model = model
        self.dimension = dimension
        ensure_parent(self.path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        try:
            # WAL lets the serving process read while a batch job writes.
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes

    def write_many(self, keys: Sequence[str], vectors: np.ndarray, source: str = "batch") -> int:
        """Upsert a batch of vectors. Returns the number of rows written."""
        if len(keys) != len(vectors):
            raise ValueError(f"keys/vectors length mismatch: {len(keys)} vs {len(vectors)}")
        vectors = np.asarray(vectors, dtype=_DTYPE)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        dimension = int(vectors.shape[1]) if vectors.ndim == 2 else 0

        rows = [
            (key, self.model, dimension, vec.tobytes(), source, now)
            for key, vec in zip(keys, vectors, strict=True)
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO features (text_key, model, dimension, vector, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(text_key) DO UPDATE SET "
                "  model=excluded.model, dimension=excluded.dimension, vector=excluded.vector, "
                "  source=excluded.source, created_at=excluded.created_at",
                rows,
            )
        return len(rows)

    def set_metadata(self, **items: Any) -> None:
        """Record provenance about how the store was populated."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(k, str(v)) for k, v in items.items()],
            )

    # ------------------------------------------------------------------- reads

    def read_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """Fetch whatever is cached for ``keys``. Missing keys are simply absent."""
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        with self._connect() as conn:
            # Chunked to stay under SQLite's variable limit on large batches.
            for chunk in _chunks(list(keys), 900):
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT text_key, vector FROM features WHERE text_key IN ({placeholders})",  # noqa: S608
                    chunk,
                )
                for key, blob in cursor:
                    out[key] = np.frombuffer(blob, dtype=_DTYPE)
        return out

    def get(self, key: str) -> np.ndarray | None:
        """Single-key lookup - the online serving read path."""
        return self.read_many([key]).get(key)

    def matrix(self, keys: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """Return a dense matrix for ``keys`` plus the list of keys that were missing.

        Row order matches ``keys``; missing rows are zero-filled, so callers must check
        the returned missing list rather than assuming a full hit.
        """
        cached = self.read_many(keys)
        if not cached:
            return np.zeros((len(keys), self.dimension), dtype=_DTYPE), list(keys)
        dimension = self.dimension or len(next(iter(cached.values())))
        out = np.zeros((len(keys), dimension), dtype=_DTYPE)
        missing: list[str] = []
        for i, key in enumerate(keys):
            vec = cached.get(key)
            if vec is None:
                missing.append(key)
            else:
                out[i] = vec
        return out, missing

    # -------------------------------------------------------------- inspection

    def stats(self) -> dict[str, Any]:
        """Row count, per-source breakdown and last write - surfaced by ``/model/info``."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
            by_source = dict(conn.execute("SELECT source, COUNT(*) FROM features GROUP BY source"))
            last_write = conn.execute("SELECT MAX(created_at) FROM features").fetchone()[0]
            metadata = dict(conn.execute("SELECT key, value FROM metadata"))
        return {
            "rows": total,
            "by_source": by_source,
            "last_write": last_write,
            "metadata": metadata,
            "path": str(self.path),
        }


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
