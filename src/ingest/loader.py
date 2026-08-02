"""Ingest stage: fetch the raw corpus and transcribe it into a tabular parquet file.

This stage is deliberately *faithful*, not clever. It does not clean, filter or repair
anything - malformed scores and empty texts are carried through as nulls so that the
validation stage downstream is the single place where rows get judged and quarantined.
Splitting it this way keeps "what arrived" and "what we accepted" separately auditable.

Two sources are supported and produce an identical schema:

* the original Stanford SNAP release (default, no authentication required)
* a Kaggle-style ``Reviews.csv``, used automatically when present

Run with ``python -m src.ingest.loader``.
"""

from __future__ import annotations

import gzip
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import ensure_parent, load_config, resolve

# SNAP block format: one "key: value" per line, records separated by a blank line.
_FIELD_MAP = {
    "product/productId": "ProductId",
    "review/userId": "UserId",
    "review/helpfulness": "_helpfulness",
    "review/score": "_score",
    "review/time": "_time",
    "review/summary": "Summary",
    "review/text": "Text",
}

# review/profileName is intentionally NOT ingested: it is personal data with no
# modelling value for this task. Privacy by default rather than by later deletion.

ARROW_SCHEMA = pa.schema(
    [
        ("Id", pa.int64()),
        ("ProductId", pa.string()),
        ("UserId", pa.string()),
        ("HelpfulnessNumerator", pa.int32()),
        ("HelpfulnessDenominator", pa.int32()),
        ("Score", pa.int8()),
        ("Time", pa.int64()),
        ("Summary", pa.string()),
        ("Text", pa.string()),
    ]
)

_BATCH_ROWS = 50_000


def download_archive(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it is already there.

    Called by the ``download`` stage (:mod:`src.ingest.download`), which owns the archive
    as its output. Note that DVC clears a stage's outputs before re-running it, so under
    ``dvc repro`` the file is normally absent and this does re-fetch; the existence check
    matters when the function is called directly, outside the pipeline.
    """
    dest = ensure_parent(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] archive already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print(f"[download] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r[download]   {read / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def _parse_helpfulness(raw: str | None) -> tuple[int | None, int | None]:
    """``"3/5"`` -> ``(3, 5)``. Returns nulls when unparseable rather than guessing."""
    if not raw or "/" not in raw:
        return None, None
    num, _, den = raw.partition("/")
    try:
        return int(num), int(den)
    except ValueError:
        return None, None


def _to_int(raw: str | None) -> int | None:
    """Parse ints tolerantly - scores arrive as ``"5.0"``, times as ``"1303862400"``."""
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_snap(archive: Path, encoding: str, errors: str) -> Iterator[dict[str, Any]]:
    """Stream records out of the gzipped SNAP block format.

    The archive is *not* clean UTF-8; it carries stray cp1252 bytes. We decode
    permissively and count the damage instead of letting one bad byte abort the ingest.
    The count is reported on stdout by this stage (1,789 on the current corpus); it is
    deliberately not part of the data-quality report, which covers row-level validation.
    """
    record: dict[str, Any] = {}
    last_key: str | None = None
    row_id = 0

    with gzip.open(archive, "rb") as fh:
        for raw_line in fh:
            line = raw_line.decode(encoding, errors=errors).rstrip("\n").rstrip("\r")

            if not line.strip():
                if record:
                    row_id += 1
                    yield _finalise(record, row_id)
                    record, last_key = {}, None
                continue

            key, sep, value = line.partition(": ")
            if sep and key in _FIELD_MAP:
                last_key = _FIELD_MAP[key]
                record[last_key] = value
            elif last_key is not None:
                # Continuation of a multi-line field: append rather than drop it.
                record[last_key] = f"{record.get(last_key, '')}\n{line}"

    if record:  # final record when the file does not end with a blank line
        row_id += 1
        yield _finalise(record, row_id)


def _finalise(record: dict[str, Any], row_id: int) -> dict[str, Any]:
    """Normalise one raw record into the shared output schema."""
    num, den = _parse_helpfulness(record.get("_helpfulness"))
    return {
        "Id": row_id,
        "ProductId": record.get("ProductId"),
        "UserId": record.get("UserId"),
        "HelpfulnessNumerator": num,
        "HelpfulnessDenominator": den,
        "Score": _to_int(record.get("_score")),
        "Time": _to_int(record.get("_time")),
        "Summary": record.get("Summary"),
        "Text": record.get("Text"),
    }


def _write_batches(records: Iterator[dict[str, Any]], dest: Path) -> dict[str, Any]:
    """Stream records into parquet in batches so the full corpus never sits in RAM."""
    ensure_parent(dest)
    columns = [f.name for f in ARROW_SCHEMA]
    buffer: list[dict[str, Any]] = []
    n_rows = 0
    n_replacement_chars = 0
    writer: pq.ParquetWriter | None = None

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        arrays = {col: [rec.get(col) for rec in buffer] for col in columns}
        table = pa.Table.from_pydict(arrays, schema=ARROW_SCHEMA)
        assert writer is not None
        writer.write_table(table)
        buffer = []

    try:
        writer = pq.ParquetWriter(dest, ARROW_SCHEMA, compression="zstd")
        for record in records:
            text = record.get("Text") or ""
            if "�" in text:
                n_replacement_chars += text.count("�")
            buffer.append(record)
            n_rows += 1
            if len(buffer) >= _BATCH_ROWS:
                flush()
                print(f"\r[ingest]   parsed {n_rows:,} records", end="", flush=True)
        flush()
    finally:
        if writer is not None:
            writer.close()

    print(f"\r[ingest]   parsed {n_rows:,} records")
    return {"rows": n_rows, "replacement_chars": n_replacement_chars}


def _ingest_csv(csv_path: Path, dest: Path) -> dict[str, Any]:
    """Alternative path for a Kaggle-style ``Reviews.csv``."""
    import pandas as pd

    print(f"[ingest] using CSV override: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df = df.reindex(columns=[f.name for f in ARROW_SCHEMA])
    ensure_parent(dest)
    pq.write_table(pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False), dest, compression="zstd")
    return {"rows": len(df), "replacement_chars": int(df["Text"].fillna("").str.count("�").sum())}


def ingest() -> dict[str, Any]:
    """Run the ingest stage end to end and return its statistics."""
    cfg = load_config("data")
    source, paths = cfg["source"], cfg["paths"]
    dest = resolve(paths["parsed"])

    csv_override = resolve(source["csv_override"])
    if csv_override.exists():
        stats = _ingest_csv(csv_override, dest)
    else:
        archive = download_archive(source["url"], resolve(source["archive"]))
        records = parse_snap(archive, source["encoding"], source["encoding_errors"])
        stats = _write_batches(records, dest)

    stats["output"] = str(dest.relative_to(resolve(".")))
    print(
        f"[ingest] wrote {stats['rows']:,} rows -> {stats['output']} "
        f"({stats['replacement_chars']:,} undecodable characters replaced)"
    )
    return stats


if __name__ == "__main__":
    ingest()
