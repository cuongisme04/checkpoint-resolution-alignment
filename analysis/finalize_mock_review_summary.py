#!/usr/bin/env python3
"""Attach validated bootstrap intervals to the expanded 54-row summary."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


class FinalizeError(RuntimeError):
    pass


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FinalizeError(f"input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise FinalizeError(f"input is empty: {path}")
    return fields, rows


def interval_index(path: Path, expected_count: int, expected_method: str) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    if len(rows) != expected_count:
        raise FinalizeError(
            f"expected {expected_count} interval rows in {path}, found {len(rows)}"
        )
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        config_id = row.get("configuration_id", "")
        if not config_id or config_id in indexed:
            raise FinalizeError(f"missing or duplicate configuration_id in {path}")
        if row.get("ci_method") != expected_method:
            raise FinalizeError(
                f"unexpected interval method for {config_id}: {row.get('ci_method')}"
            )
        indexed[config_id] = row
    return indexed


def finalize(
    summary_path: Path,
    classification_intervals_path: Path,
    retrieval_intervals_path: Path,
    output_path: Path,
) -> list[dict[str, str]]:
    fields, rows = read_csv(summary_path)
    if len(rows) != 54:
        raise FinalizeError(f"expanded summary must contain 54 rows, found {len(rows)}")
    classification = interval_index(
        classification_intervals_path, 6, "sample_bootstrap"
    )
    retrieval = interval_index(
        retrieval_intervals_path, 24, "image_grouped_bootstrap"
    )
    used_classification: set[str] = set()
    used_retrieval: set[str] = set()
    for row in rows:
        config_id = row.get("configuration_id", "")
        interval = None
        if row["task"] == "image_text_retrieval":
            interval = retrieval.get(config_id)
            if interval is None:
                raise FinalizeError(f"missing grouped retrieval interval for {config_id}")
            used_retrieval.add(config_id)
        elif config_id in classification:
            interval = classification[config_id]
            used_classification.add(config_id)
        if interval is None:
            if not row.get("ci_method") or not row.get("ci_lower_95") or not row.get("ci_upper_95"):
                raise FinalizeError(
                    f"retained classification row lacks an interval: {row.get('model_id')}/{row.get('resolution')}"
                )
            continue
        if not math.isclose(
            float(row["performance_value"]),
            float(interval["performance_value"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FinalizeError(f"point estimate mismatch for {config_id}")
        lower = float(interval["ci_lower_95"])
        upper = float(interval["ci_upper_95"])
        row["ci_lower_95"] = interval["ci_lower_95"]
        row["ci_upper_95"] = interval["ci_upper_95"]
        row["ci_half_width_95"] = f"{0.5 * (upper - lower):.17g}"
        row["ci_method"] = interval["ci_method"]
    if used_classification != set(classification) or used_retrieval != set(retrieval):
        raise FinalizeError("one or more interval rows were not matched to the summary")
    for field in ("ci_lower_95", "ci_upper_95", "ci_half_width_95", "ci_method"):
        if field not in fields:
            fields.append(field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--classification-intervals", type=Path, required=True)
    parser.add_argument("--retrieval-intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = finalize(
            args.summary,
            args.classification_intervals,
            args.retrieval_intervals,
            args.output,
        )
    except (FinalizeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Finalization failed: {exc}") from exc
    print(f"Wrote {len(rows)} rows with validated bootstrap intervals to {args.output}")


if __name__ == "__main__":
    main()
