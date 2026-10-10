#!/usr/bin/env python3
"""Summarise the fixed batch-one versus batch-32 microbenchmark."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


class BatchAnalysisError(RuntimeError):
    pass


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BatchAnalysisError("cannot take a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BatchAnalysisError(f"input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BatchAnalysisError("batch timing input is empty")
    return rows


def analyse(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    required = {
        "configuration_id", "model_id", "resolution", "batch_size", "precision",
        "block", "iteration", "batch_latency_ms", "latency_ms_per_image",
        "throughput_images_s", "timing_scope", "gpu_name",
        "peak_allocated_vram_gb", "peak_reserved_vram_gb",
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for number, row in enumerate(rows, start=2):
        missing = [field for field in required if not (row.get(field) or "").strip()]
        if missing:
            raise BatchAnalysisError(f"row {number} is missing {', '.join(sorted(missing))}")
        grouped[row["configuration_id"]].append(row)
    if len(rows) != 6000 or len(grouped) != 24:
        raise BatchAnalysisError(
            f"expected 6,000 rows across 24 configurations, found {len(rows)}/{len(grouped)}"
        )

    summaries = []
    pair_lookup: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for config_id, config_rows in sorted(grouped.items()):
        if len(config_rows) != 250:
            raise BatchAnalysisError(f"{config_id} has {len(config_rows)} rows, expected 250")
        blocks = {int(row["block"]) for row in config_rows}
        iterations = {(int(row["block"]), int(row["iteration"])) for row in config_rows}
        if blocks != set(range(1, 6)) or len(iterations) != 250:
            raise BatchAnalysisError(f"invalid block/iteration structure for {config_id}")
        invariant_fields = (
            "model_id", "resolution", "batch_size", "precision", "timing_scope",
            "gpu_name", "peak_allocated_vram_gb", "peak_reserved_vram_gb",
        )
        invariants = {field: {row[field] for row in config_rows} for field in invariant_fields}
        changed = [field for field, values in invariants.items() if len(values) != 1]
        if changed:
            raise BatchAnalysisError(f"configuration fields changed within {config_id}: {changed}")
        latency = [float(row["latency_ms_per_image"]) for row in config_rows]
        throughput = [float(row["throughput_images_s"]) for row in config_rows]
        batch_latency = [float(row["batch_latency_ms"]) for row in config_rows]
        if any(value <= 0 for value in latency + throughput + batch_latency):
            raise BatchAnalysisError(f"non-positive timing in {config_id}")
        first = config_rows[0]
        summary = {
            "configuration_id": config_id,
            "model_id": first["model_id"],
            "resolution": int(first["resolution"]),
            "batch_size": int(first["batch_size"]),
            "precision": first["precision"],
            "timing_scope": first["timing_scope"],
            "gpu_name": first["gpu_name"],
            "observation_count": len(config_rows),
            "block_count": len(blocks),
            "median_batch_latency_ms": statistics.median(batch_latency),
            "median_latency_ms_per_image": statistics.median(latency),
            "p05_latency_ms_per_image": percentile(latency, 0.05),
            "p95_latency_ms_per_image": percentile(latency, 0.95),
            "median_throughput_images_s": statistics.median(throughput),
            "peak_allocated_vram_gb": float(first["peak_allocated_vram_gb"]),
            "peak_reserved_vram_gb": float(first["peak_reserved_vram_gb"]),
        }
        summaries.append(summary)
        key = (summary["model_id"], summary["resolution"])
        if summary["batch_size"] in pair_lookup[key]:
            raise BatchAnalysisError(f"duplicate batch size for {key}")
        pair_lookup[key][summary["batch_size"]] = summary

    if len(pair_lookup) != 12:
        raise BatchAnalysisError(f"expected 12 model-resolution pairs, found {len(pair_lookup)}")
    comparisons = []
    for (model_id, resolution), batches in sorted(pair_lookup.items()):
        if set(batches) != {1, 32}:
            raise BatchAnalysisError(f"missing batch-one or batch-32 result for {(model_id, resolution)}")
        one, thirty_two = batches[1], batches[32]
        comparisons.append({
            "model_id": model_id,
            "resolution": resolution,
            "batch1_latency_ms_per_image": one["median_latency_ms_per_image"],
            "batch32_latency_ms_per_image": thirty_two["median_latency_ms_per_image"],
            "batch1_throughput_images_s": one["median_throughput_images_s"],
            "batch32_throughput_images_s": thirty_two["median_throughput_images_s"],
            "batch32_throughput_speedup": thirty_two["median_throughput_images_s"] / one["median_throughput_images_s"],
            "batch32_per_image_latency_ratio": thirty_two["median_latency_ms_per_image"] / one["median_latency_ms_per_image"],
            "batch1_peak_allocated_vram_gb": one["peak_allocated_vram_gb"],
            "batch32_peak_allocated_vram_gb": thirty_two["peak_allocated_vram_gb"],
            "batch32_allocated_vram_delta_gb": thirty_two["peak_allocated_vram_gb"] - one["peak_allocated_vram_gb"],
        })
    return summaries, comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summaries, comparisons = analyse(read_rows(args.input))
    except (BatchAnalysisError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Batch-sensitivity analysis failed: {exc}") from exc
    write_csv(args.output_dir / "batch_configuration_summary.csv", summaries)
    write_csv(args.output_dir / "batch_pair_comparisons.csv", comparisons)
    print(f"Wrote {len(summaries)} configuration summaries and {len(comparisons)} paired comparisons.")


if __name__ == "__main__":
    main()
