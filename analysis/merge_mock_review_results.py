#!/usr/bin/env python3
"""Merge validated mock-review summaries without editing the locked baseline.

The focused rerun replaces the four historical BLIP Pet anchor rows and all
22 historical Flickr30k rows, then adds the four missing BLIP-384 task rows.
The result therefore contains 54 rows: 30 Pet and 24 Flickr30k.  This script
does not compute confidence intervals or Pareto membership; those are derived
only after the grouped outcomes pass ``mock_review_analysis.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "results" / "historical" / "summary_50.csv"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT / "reproducibility" / "benchmark" / "configs" / "mock_review_protocol.json"
)
LOCKED_BASELINE_SHA256 = "2d2fa4103fb557776d9ad5d78f9d0ed030242ba34cde581f8051409bc1e17ea1"
EXPECTED_GPU = "NVIDIA RTX 6000 Ada Generation"
BLIP_MODELS = {"blip_itm_base_coco", "blip_itm_large_coco"}


class MergeError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-summary", type=Path, required=True)
    parser.add_argument("--retrieval-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise MergeError(f"input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise MergeError(f"input is empty: {path}")
    return fields, rows


def expected_rows(protocol_path: Path) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict]]:
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read protocol {protocol_path}: {exc}") from exc
    expected: dict[str, dict[tuple[str, int], dict]] = {"classification": {}, "retrieval": {}}
    for task in expected:
        for group in protocol[task]:
            for model in group["models"]:
                for resolution in group["resolutions"]:
                    key = (model, int(resolution))
                    if key in expected[task]:
                        raise MergeError(f"duplicate protocol configuration: {task}/{key}")
                    expected[task][key] = group
    return expected["classification"], expected["retrieval"]


def validate_new_rows(
    rows: list[dict[str, str]],
    *,
    task: str,
    dataset: str,
    expected: dict[tuple[str, int], dict],
) -> None:
    seen: set[tuple[str, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        required = (
            "model_id", "task", "dataset", "resolution", "performance_metric",
            "performance_value", "batch_size", "precision", "gpu_name",
            "configuration_id", "throughput_examples_s", "peak_vram_gb",
        )
        missing = [field for field in required if not (row.get(field) or "").strip()]
        if missing:
            raise MergeError(f"{task} row {row_number} is missing {', '.join(missing)}")
        key = (row["model_id"], int(row["resolution"]))
        if key in seen:
            raise MergeError(f"duplicate {task} summary configuration: {key}")
        seen.add(key)
        if row["task"] != ("zeroshot_classification" if task == "classification" else "image_text_retrieval"):
            raise MergeError(f"unexpected task in {task} row {row_number}: {row['task']}")
        if row["dataset"] != dataset:
            raise MergeError(f"unexpected dataset in {task} row {row_number}: {row['dataset']}")
        if row["gpu_name"] != EXPECTED_GPU:
            raise MergeError(f"unexpected GPU in {task} row {row_number}: {row['gpu_name']}")
        if float(row["performance_value"]) < 0 or float(row["throughput_examples_s"]) <= 0:
            raise MergeError(f"invalid performance/throughput in {task} row {row_number}")
        group = expected.get(key)
        if group is None:
            raise MergeError(f"unexpected {task} configuration: {key}")
        if int(row["batch_size"]) != int(group["image_batch_size"]):
            raise MergeError(f"image batch mismatch for {task}/{key}")
        if row["precision"] != group["precision"]:
            raise MergeError(f"precision mismatch for {task}/{key}")
        if task == "retrieval":
            if not (row.get("text_batch_size") or "").strip():
                raise MergeError(f"missing text batch size for retrieval/{key}")
            if int(row["text_batch_size"]) != int(group["text_batch_size"]):
                raise MergeError(f"text batch mismatch for retrieval/{key}")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        extra = sorted(seen - set(expected))
        raise MergeError(f"{task} summary configuration mismatch; missing={missing}, extra={extra}")


def merge_results(
    baseline_path: Path,
    classification_path: Path,
    retrieval_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    expected_baseline_sha256: str = LOCKED_BASELINE_SHA256,
) -> list[dict[str, str]]:
    if sha256(baseline_path) != expected_baseline_sha256:
        raise MergeError("locked baseline checksum changed")
    baseline_fields, baseline = read_csv(baseline_path)
    classification_fields, classification = read_csv(classification_path)
    retrieval_fields, retrieval = read_csv(retrieval_path)
    if len(baseline) != 50:
        raise MergeError(f"expected 50 baseline rows, found {len(baseline)}")

    expected_classification, expected_retrieval = expected_rows(protocol_path)
    validate_new_rows(
        classification,
        task="classification",
        dataset="oxford_iiit_pet",
        expected=expected_classification,
    )
    validate_new_rows(
        retrieval,
        task="retrieval",
        dataset="flickr30k_test1k",
        expected=expected_retrieval,
    )

    retained = [
        row for row in baseline
        if row["task"] == "zeroshot_classification"
        and not (row["model_id"] in BLIP_MODELS and int(row["resolution"]) in {336, 448})
    ]
    if len(retained) != 24:
        raise MergeError(f"expected 24 retained baseline classification rows, found {len(retained)}")

    for row in classification:
        row["manifest"] = "datasets/manifests/pet_test.csv"
        row["source_csv"] = "outputs/mock_review/classification/benchmark_rows.csv"
    for row in retrieval:
        row["manifest"] = "datasets/manifests/flickr30k_test1k.csv"
        row["source_csv"] = "outputs/mock_review/retrieval/benchmark_rows.csv"

    merged = retained + classification + retrieval
    keys = [(row["model_id"], row["task"], int(row["resolution"])) for row in merged]
    if len(merged) != 54 or len(set(keys)) != 54:
        raise MergeError(f"merged summary must contain 54 unique configurations, found {len(merged)}/{len(set(keys))}")
    merged.sort(key=lambda row: (row["task"], row["model_id"], int(row["resolution"])))

    fields = list(baseline_fields)
    for field in classification_fields + retrieval_fields:
        if field not in fields:
            fields.append(field)
    for field in ("manifest", "source_csv"):
        if field not in fields:
            fields.append(field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    temporary.replace(output_path)
    return merged


def main() -> None:
    args = parse_args()
    try:
        rows = merge_results(
            args.baseline,
            args.classification_summary,
            args.retrieval_summary,
            args.protocol,
            args.output,
        )
    except (MergeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Mock-review merge failed: {exc}") from exc
    print(f"Wrote {len(rows)} validated rows to {args.output}")


if __name__ == "__main__":
    main()
