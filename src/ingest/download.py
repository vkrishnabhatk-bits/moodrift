"""Download stage: fetch the raw corpus.

This is a separate pipeline stage rather than a step inside ``ingest`` for one specific
reason: it lets a fresh clone bootstrap with no DVC remote and no credentials. Because the
archive is this stage's *output* rather than a pre-existing input, ``dvc repro`` on a clean
checkout runs the download itself and then proceeds through the rest of the pipeline. An
evaluator can go from ``git clone`` to a fully rebuilt dataset with ``make data`` and
nothing else.

The archive is DVC-tracked and hashed - it is this stage's output, so any change to it
invalidates ``ingest`` and everything downstream.

**Immutability is enforced by DVC's content hashing, not by file permissions.** An earlier
version chmod-ed the file read-only here, which looked reassuring but did nothing: DVC takes
ownership of stage outputs and re-creates the workspace copy from its cache after the stage
finishes, resetting the mode. The guarantee that actually holds is that ``dvc status``
reports any modification and ``dvc checkout`` restores the original bytes - stronger than a
permission bit, which any user can flip back. Teams wanting the mode enforced as well can
set ``dvc config cache.type hardlink``, which makes workspace files read-only links into
the cache.

**This stage always produces the archive, even when a Kaggle ``Reviews.csv`` is present.**
An earlier version returned early in that case, which would have failed the stage outright:
DVC requires a stage to create every output it declares, and this one declares the archive.
The CSV override is honoured where it belongs, in :mod:`src.ingest.loader`, which prefers
it when parsing.

Run with ``python -m src.ingest.download``.
"""

from __future__ import annotations

from typing import Any

from src.config import load_config, resolve
from src.ingest.loader import download_archive


def main() -> dict[str, Any]:
    """Fetch the archive if it is not already present."""
    source = load_config("data")["source"]

    csv_override = resolve(source["csv_override"])
    if csv_override.exists():
        # Downloaded anyway: the archive is this stage's declared DVC output, and skipping
        # it would break `dvc repro`. The loader will still prefer the CSV when parsing.
        print(f"[download] note: {csv_override.name} present and will be preferred by ingest")

    path = download_archive(source["url"], resolve(source["archive"]))
    print(f"[download] raw archive ready: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "csv_override_present": csv_override.exists(),
    }


if __name__ == "__main__":
    main()
