"""Validation stage: judge every ingested row, quarantine the bad ones.

Two mechanisms, deliberately combined:

1. **Per-row rules** produce a quarantine file with a ``reason`` column. Rows are never
   silently dropped - a reviewer can open ``data/interim/rejected/`` and see exactly
   what was thrown away and why.
2. **A Pandera schema** states the contract that surviving rows must satisfy, and is
   asserted after filtering. If it ever fails, the bug is in *our rules*, not in the
   data - which is precisely the kind of error that otherwise ships silently.

Distributional facts (class balance) are *reported, never enforced*: the real 1-5 star
imbalance is a property of the domain we want to preserve, not an error to correct here.

Run with ``python -m src.ingest.schema``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa

from src.config import ensure_parent, load_config, resolve


def build_schema(cfg: dict[str, Any]) -> pa.DataFrameSchema:
    """The contract that validated data must satisfy."""
    v = cfg["validation"]
    return pa.DataFrameSchema(
        {
            "Id": pa.Column(int, unique=True, nullable=False),
            "ProductId": pa.Column(str, nullable=True),
            "UserId": pa.Column(str, nullable=True),
            "HelpfulnessNumerator": pa.Column("Int32", nullable=True),
            "HelpfulnessDenominator": pa.Column("Int32", nullable=True),
            "Score": pa.Column(
                "Int8",
                checks=pa.Check.isin(v["valid_scores"]),
                nullable=False,
            ),
            "Time": pa.Column("Int64", nullable=True),
            "Summary": pa.Column(str, nullable=True),
            "Text": pa.Column(
                str,
                checks=[
                    pa.Check.str_length(min_value=v["min_text_chars"], max_value=v["max_text_chars"]),
                    pa.Check(lambda s: s.str.strip().str.len() > 0, element_wise=False, name="non_blank"),
                ],
                nullable=False,
            ),
        },
        strict=False,
        coerce=True,
    )


def _rule_masks(df: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, pd.Series]]:
    """Per-row rejection rules, in priority order.

    Each row is attributed to the *first* rule it violates, so the reason column stays
    interpretable ("why was this dropped?") rather than listing every incidental failure.
    """
    v = cfg["validation"]
    text = df["Text"]
    text_stripped = text.fillna("").str.strip()

    return [
        ("missing_text", text.isna() | (text_stripped.str.len() == 0)),
        ("invalid_score", df["Score"].isna() | ~df["Score"].isin(v["valid_scores"])),
        ("text_too_long", text.fillna("").str.len() > v["max_text_chars"]),
        ("duplicate_id", df["Id"].duplicated(keep="first")),
        # Same reviewer posting identical text on the same product: a genuine duplicate
        # rather than two independent observations. Keeping both would leak across splits.
        ("duplicate_review", df.duplicated(subset=["UserId", "ProductId", "Text"], keep="first")),
        # Known defect in this corpus: more "helpful" votes than total votes.
        (
            "helpfulness_exceeds_total",
            df["HelpfulnessNumerator"].notna()
            & df["HelpfulnessDenominator"].notna()
            & (df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]),
        ),
    ]


def validate() -> dict[str, Any]:
    """Run the validation stage end to end and return its statistics."""
    cfg = load_config("data")
    paths = cfg["paths"]

    src_path = resolve(paths["parsed"])
    print(f"[validate] reading {src_path}")
    df = pd.read_parquet(src_path)
    n_input = len(df)

    # Normalise dtypes up front so both the rules and the schema see the same thing.
    df["Summary"] = df["Summary"].fillna("")
    for col, dtype in (
        ("Score", "Int8"),
        ("Time", "Int64"),
        ("HelpfulnessNumerator", "Int32"),
        ("HelpfulnessDenominator", "Int32"),
    ):
        df[col] = df[col].astype(dtype)

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    counts: dict[str, int] = {}
    for name, mask in _rule_masks(df, cfg):
        newly_failed = mask.fillna(False) & reason.isna()
        counts[name] = int(newly_failed.sum())
        reason[newly_failed] = name

    rejected = df[reason.notna()].copy()
    rejected["reason"] = reason[reason.notna()]
    clean = df[reason.isna()].copy()

    # Assert the contract holds on what survived. A failure here is a bug in the rules.
    schema = build_schema(cfg)
    try:
        clean = schema.validate(clean, lazy=True)
    except pa.errors.SchemaErrors as exc:  # pragma: no cover - defensive
        raise AssertionError(
            "validated data violates the declared schema - the rejection rules above are "
            f"incomplete:\n{exc.failure_cases.head(20)}"
        ) from exc

    ensure_parent(resolve(paths["validated"]))
    ensure_parent(resolve(paths["rejected"]))
    clean.to_parquet(resolve(paths["validated"]), index=False, compression="zstd")
    rejected.to_parquet(resolve(paths["rejected"]), index=False, compression="zstd")

    distribution = clean["Score"].value_counts().sort_index()
    stats = {
        "rows_in": n_input,
        "rows_accepted": len(clean),
        "rows_rejected": len(rejected),
        "rejection_reasons": counts,
        "class_distribution": {int(k): int(v) for k, v in distribution.items()},
    }
    _write_report(stats, resolve(paths["quality_report"]))

    print(
        f"[validate] accepted {len(clean):,} / {n_input:,} rows "
        f"({len(rejected):,} quarantined -> {paths['rejected']})"
    )
    for name, count in counts.items():
        if count:
            print(f"[validate]   {name}: {count:,}")
    return stats


def _write_report(stats: dict[str, Any], dest: Path) -> None:
    """Emit the human-readable data-quality report (a Week 1 deliverable)."""
    total = stats["rows_in"]
    accepted = stats["rows_accepted"]
    dist = stats["class_distribution"]
    dist_total = sum(dist.values()) or 1

    lines = [
        "# Data quality report",
        "",
        "Generated by `python -m src.ingest.schema`. Do not edit by hand.",
        "",
        "## Volume",
        "",
        "| Metric | Rows |",
        "|---|---|",
        f"| Ingested | {total:,} |",
        f"| Accepted | {accepted:,} ({accepted / max(total, 1):.2%}) |",
        f"| Quarantined | {stats['rows_rejected']:,} ({stats['rows_rejected'] / max(total, 1):.2%}) |",
        "",
        "## Rejection reasons",
        "",
        "Rows are attributed to the first rule they violate. Quarantined rows are written",
        "to `data/interim/rejected/` with this reason attached - nothing is dropped silently.",
        "",
        "| Reason | Rows |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {count:,} |" for name, count in stats["rejection_reasons"].items()]
    lines += [
        "",
        "## Class distribution (accepted rows)",
        "",
        "Reported, not enforced. The imbalance below is a real property of the corpus and is",
        "preserved by proportional sampling; it is handled at training time via class weighting",
        "(see `docs/decisions/ADR-0001-sampling-and-class-imbalance.md`).",
        "",
        "| Score | Rows | Share |",
        "|---|---|---|",
    ]
    lines += [f"| {score} | {count:,} | {count / dist_total:.2%} |" for score, count in sorted(dist.items())]
    lines.append("")

    ensure_parent(dest).write_text("\n".join(lines), encoding="utf-8")
    print(f"[validate] report -> {dest}")


if __name__ == "__main__":
    validate()
