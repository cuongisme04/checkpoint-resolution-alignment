#!/usr/bin/env python3
"""Validate a canonical 50- or 54-row summary and regenerate derived results.

The canonical CSV is read-only. All derived CSV, LaTeX, vector figures, trace
records, and manifest artifacts are written to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


getcontext().prec = 40

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "canonical"
    / "dicta_mock_review_54_with_ci.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "processed"

EXPECTED_INPUT_SHA256 = (
    "4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7"
)
EXPECTED_GPU = "NVIDIA RTX 6000 Ada Generation"
BASELINE_TASK_COUNTS = {
    "image_text_retrieval": 22,
    "zeroshot_classification": 28,
}
EXPANDED_TASK_COUNTS = {
    "image_text_retrieval": 24,
    "zeroshot_classification": 30,
}
BASELINE_PROTOCOL_COUNTS = {
    ("image_text_retrieval", "BLIP", None, 32, 128, "fp16", 2, 3): 8,
    ("image_text_retrieval", "OpenCLIP", None, 64, 256, "fp16", 2, 3): 14,
    ("zeroshot_classification", "BLIP", None, 32, 0, "fp16", 2, 3): 8,
    ("zeroshot_classification", "OpenCLIP", None, 64, 0, "fp16", 2, 3): 18,
    ("zeroshot_classification", "PaliGemma", 224, 1, 0, "bf16", 2, 1): 1,
    ("zeroshot_classification", "PaliGemma", 448, 1, 0, "bf16", 5, 1): 1,
}
EXPANDED_PROTOCOL_COUNTS = dict(BASELINE_PROTOCOL_COUNTS)
EXPANDED_PROTOCOL_COUNTS[("image_text_retrieval", "BLIP", None, 32, 128, "fp16", 2, 3)] = 10
EXPANDED_PROTOCOL_COUNTS[("zeroshot_classification", "BLIP", None, 32, 0, "fp16", 2, 3)] = 10
EXPECTED_CROSS_TASK_INTERSECTION = {
    ("blip_itm_base_coco", 160),
    ("clip_rn50_openai", 160),
    ("clip_rn50_openai", 224),
    ("clip_vit_b16_openai", 224),
    ("clip_vit_b32_openai", 224),
    ("siglip_b16_webli", 224),
}

REQUIRED_FIELDS = {
    "model_id",
    "model_name",
    "task",
    "dataset",
    "resolution",
    "batch_size",
    "text_batch_size",
    "precision",
    "warmup_batches",
    "timed_repeats",
    "throughput_examples_s",
    "latency_mean_ms",
    "latency_p50_ms",
    "peak_vram_gb",
    "parameter_count",
    "gpu_name",
    "performance_metric",
    "performance_value",
    "ci_lower_95",
    "ci_upper_95",
    "pareto_latency",
    "pareto_vram",
    "pareto_parameters",
    "pareto_latency_vram",
}

POSITIVE_NUMERIC_FIELDS = (
    "resolution",
    "batch_size",
    "warmup_batches",
    "timed_repeats",
    "throughput_examples_s",
    "latency_mean_ms",
    "latency_p50_ms",
    "peak_vram_gb",
    "parameter_count",
)

FRONTIER_SPECS = (
    ("pareto_latency", ("tau_img_ms",)),
    ("pareto_vram", ("peak_vram_gb",)),
    ("pareto_parameters", ("parameter_count",)),
    ("pareto_latency_vram", ("tau_img_ms", "peak_vram_gb")),
)

MODEL_ORDER = (
    "clip_rn50_openai",
    "clip_vit_b32_openai",
    "clip_vit_b16_openai",
    "clip_vit_l14_openai",
    "siglip_b16_webli",
    "mobileclip_s1_datacompdr",
    "blip_itm_base_coco",
    "blip_itm_large_coco",
    "paligemma_3b_mix_224",
    "paligemma_3b_mix_448",
)

READABLE_NAMES = {
    "clip_rn50_openai": "CLIP RN50",
    "clip_vit_b32_openai": "CLIP ViT-B/32",
    "clip_vit_b16_openai": "CLIP ViT-B/16",
    "clip_vit_l14_openai": "CLIP ViT-L/14",
    "siglip_b16_webli": "SigLIP-B/16",
    "mobileclip_s1_datacompdr": "MobileCLIP-S1",
    "blip_itm_base_coco": "BLIP-1 base (ITC)",
    "blip_itm_large_coco": "BLIP-1 large (ITC)",
    "paligemma_3b_mix_224": "PaliGemma-3B-mix-224",
    "paligemma_3b_mix_448": "PaliGemma-3B-mix-448",
}

EVALUATED_SCORING_PATHS = {
    "clip_rn50_openai": "contrastive_dual_embedding",
    "clip_vit_b32_openai": "contrastive_dual_embedding",
    "clip_vit_b16_openai": "contrastive_dual_embedding",
    "clip_vit_l14_openai": "contrastive_dual_embedding",
    "siglip_b16_webli": "sigmoid_dual_embedding",
    "mobileclip_s1_datacompdr": "contrastive_dual_embedding",
    "blip_itm_base_coco": "blip_itc_dual_embedding",
    "blip_itm_large_coco": "blip_itc_dual_embedding",
    "paligemma_3b_mix_224": "paligemma_adapted_total_likelihood",
    "paligemma_3b_mix_448": "paligemma_adapted_total_likelihood",
}


class ValidationError(RuntimeError):
    """Raised when canonical or derived data violate the locked schema."""


@dataclass
class Record:
    index: int
    raw: Dict[str, str]
    numeric: Dict[str, Decimal]
    key: Tuple[str, str, str, int, int, int, str]
    flags: Dict[str, bool] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.raw["model_id"]

    @property
    def task(self) -> str:
        return self.raw["task"]

    @property
    def resolution(self) -> int:
        return int(self.numeric["resolution"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_decimal(value: str, field_name: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(
            f"row {row_number}: {field_name} is not numeric: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise ValidationError(
            f"row {row_number}: {field_name} must be finite, got {value!r}"
        )
    return parsed


def parse_bool(value: str, field_name: str, row_number: int) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValidationError(
        f"row {row_number}: {field_name} must be True or False, got {value!r}"
    )


def decimal_string(value: Decimal) -> str:
    return format(value, ".30g")


def load_and_validate(path: Path) -> Tuple[List[Record], List[str], str]:
    if not path.is_file():
        raise ValidationError(f"canonical input not found: {path}")

    input_checksum = sha256_file(path)
    if path.resolve() == DEFAULT_INPUT.resolve() and input_checksum != EXPECTED_INPUT_SHA256:
        raise ValidationError(
            "canonical input checksum changed: "
            f"expected {EXPECTED_INPUT_SHA256}, got {input_checksum}"
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_FIELDS - set(fieldnames))
        if missing_columns:
            raise ValidationError(
                "canonical input is missing required fields: "
                + ", ".join(missing_columns)
            )
        raw_rows = list(reader)

    if len(raw_rows) not in {50, 54}:
        raise ValidationError(f"expected exactly 50 or 54 rows, found {len(raw_rows)}")
    expanded = len(raw_rows) == 54
    expected_task_counts = EXPANDED_TASK_COUNTS if expanded else BASELINE_TASK_COUNTS
    expected_protocol_counts = (
        EXPANDED_PROTOCOL_COUNTS if expanded else BASELINE_PROTOCOL_COUNTS
    )

    records: List[Record] = []
    seen_keys = set()
    gpu_names = set()
    task_counts: Counter[str] = Counter()
    protocol_counts: Counter[Tuple[object, ...]] = Counter()

    for zero_index, raw in enumerate(raw_rows):
        row_number = zero_index + 2
        nullable_fields = {"text_batch_size"}
        if expanded:
            nullable_fields.update(flag_name for flag_name, _ in FRONTIER_SPECS)
        for field_name in REQUIRED_FIELDS - nullable_fields:
            if not (raw.get(field_name) or "").strip():
                raise ValidationError(
                    f"row {row_number}: required value {field_name} is missing"
                )

        if raw["precision"] not in {"fp16", "bf16"}:
            raise ValidationError(
                f"row {row_number}: invalid precision {raw['precision']!r}"
            )

        numeric = {
            name: parse_decimal(raw[name], name, row_number)
            for name in POSITIVE_NUMERIC_FIELDS
        }
        numeric["performance_value"] = parse_decimal(
            raw["performance_value"], "performance_value", row_number
        )
        numeric["ci_lower_95"] = parse_decimal(
            raw["ci_lower_95"], "ci_lower_95", row_number
        )
        numeric["ci_upper_95"] = parse_decimal(
            raw["ci_upper_95"], "ci_upper_95", row_number
        )

        for field_name in POSITIVE_NUMERIC_FIELDS:
            if numeric[field_name] <= 0:
                raise ValidationError(
                    f"row {row_number}: {field_name} must be positive"
                )
        for field_name in (
            "resolution",
            "batch_size",
            "warmup_batches",
            "timed_repeats",
        ):
            if numeric[field_name] != numeric[field_name].to_integral_value():
                raise ValidationError(
                    f"row {row_number}: {field_name} must be an integer"
                )
        if not Decimal("0") <= numeric["performance_value"] <= Decimal("1"):
            raise ValidationError(
                f"row {row_number}: performance_value must be within [0, 1]"
            )
        if not (
            Decimal("0")
            <= numeric["ci_lower_95"]
            <= numeric["performance_value"]
            <= numeric["ci_upper_95"]
            <= Decimal("1")
        ):
            raise ValidationError(
                f"row {row_number}: confidence interval is inconsistent"
            )

        text_batch_raw = (raw.get("text_batch_size") or "").strip()
        if raw["task"] == "image_text_retrieval":
            if not text_batch_raw:
                raise ValidationError(
                    f"row {row_number}: retrieval text_batch_size is missing"
                )
            text_batch = parse_decimal(
                text_batch_raw, "text_batch_size", row_number
            )
            if text_batch <= 0:
                raise ValidationError(
                    f"row {row_number}: text_batch_size must be positive"
                )
            if text_batch != text_batch.to_integral_value():
                raise ValidationError(
                    f"row {row_number}: text_batch_size must be an integer"
                )
        elif raw["task"] == "zeroshot_classification":
            if text_batch_raw:
                raise ValidationError(
                    f"row {row_number}: classification text_batch_size must be blank"
                )
            text_batch = Decimal("0")
        else:
            raise ValidationError(
                f"row {row_number}: unsupported task {raw['task']!r}"
            )

        tau_img_ms = Decimal("1000") / numeric["throughput_examples_s"]
        numeric["tau_img_ms"] = tau_img_ms
        if "tau_img_ms" in raw and (raw.get("tau_img_ms") or "").strip():
            stored_tau = parse_decimal(raw["tau_img_ms"], "tau_img_ms", row_number)
            if abs(stored_tau - tau_img_ms) > Decimal("1e-12"):
                raise ValidationError(
                    f"row {row_number}: stored tau_img_ms is stale or mismatched"
                )

        key = (
            raw["model_id"],
            raw["task"],
            raw["dataset"],
            int(numeric["resolution"]),
            int(numeric["batch_size"]),
            int(text_batch),
            raw["precision"],
        )
        if key in seen_keys:
            raise ValidationError(f"row {row_number}: duplicate configuration {key}")
        seen_keys.add(key)

        gpu_names.add(raw["gpu_name"])
        task_counts[raw["task"]] += 1
        if raw["model_id"].startswith("paligemma"):
            protocol_family = "PaliGemma"
            protocol_resolution = int(numeric["resolution"])
        elif raw["model_id"].startswith("blip"):
            protocol_family = "BLIP"
            protocol_resolution = None
        else:
            protocol_family = "OpenCLIP"
            protocol_resolution = None
        protocol_counts[
            (
                raw["task"],
                protocol_family,
                protocol_resolution,
                int(numeric["batch_size"]),
                int(text_batch),
                raw["precision"],
                int(numeric["warmup_batches"]),
                int(numeric["timed_repeats"]),
            )
        ] += 1
        records.append(Record(zero_index, raw, numeric, key))

    if gpu_names != {EXPECTED_GPU}:
        raise ValidationError(
            f"inconsistent GPU names: expected only {EXPECTED_GPU!r}, got {gpu_names}"
        )
    if dict(task_counts) != expected_task_counts:
        raise ValidationError(
            f"unexpected task counts: expected {expected_task_counts}, got {dict(task_counts)}"
        )
    if dict(protocol_counts) != expected_protocol_counts:
        raise ValidationError(
            "recorded batch/precision/warmup/repeat protocol changed: "
            f"expected {expected_protocol_counts}, got {dict(protocol_counts)}"
        )
    return records, fieldnames, input_checksum


def recompute_frontiers(records: Sequence[Record]) -> None:
    buckets: Dict[Tuple[str, str, str], List[Record]] = defaultdict(list)
    for record in records:
        bucket = (
            record.raw["task"],
            record.raw["dataset"],
            record.raw["performance_metric"],
        )
        buckets[bucket].append(record)

    for _, bucket_records in sorted(buckets.items()):
        for flag_name, cost_fields in FRONTIER_SPECS:
            for candidate in bucket_records:
                dominated = False
                for other in bucket_records:
                    if other is candidate:
                        continue
                    no_more_costly = all(
                        other.numeric[field_name] <= candidate.numeric[field_name]
                        for field_name in cost_fields
                    )
                    no_worse_performance = (
                        other.numeric["performance_value"]
                        >= candidate.numeric["performance_value"]
                    )
                    at_least_one_strict = (
                        any(
                            other.numeric[field_name]
                            < candidate.numeric[field_name]
                            for field_name in cost_fields
                        )
                        or other.numeric["performance_value"]
                        > candidate.numeric["performance_value"]
                    )
                    if no_more_costly and no_worse_performance and at_least_one_strict:
                        dominated = True
                        break
                candidate.flags[flag_name] = not dominated


def validate_derived_values(records: Sequence[Record]) -> Dict[str, object]:
    expanded = len(records) == 54
    if not expanded:
        for record in records:
            for stable_flag in ("pareto_vram", "pareto_parameters"):
                legacy = parse_bool(
                    record.raw[stable_flag], stable_flag, record.index + 2
                )
                if legacy != record.flags[stable_flag]:
                    raise ValidationError(
                        f"row {record.index + 2}: unchanged-cost frontier {stable_flag} "
                        "does not reproduce"
                    )

        changes: Dict[str, List[Tuple[str, int, bool, bool]]] = defaultdict(list)
        for record in records:
            legacy = parse_bool(
                record.raw["pareto_latency_vram"],
                "pareto_latency_vram",
                record.index + 2,
            )
            regenerated = record.flags["pareto_latency_vram"]
            if legacy != regenerated:
                changes[record.task].append(
                    (record.model_id, record.resolution, legacy, regenerated)
                )

        flickr_changes = changes.get("image_text_retrieval", [])
        expected_pet_change = [("clip_vit_b16_openai", 224, False, True)]
        pet_changes = sorted(changes.get("zeroshot_classification", []))
        if flickr_changes:
            raise ValidationError(
                "Flickr30k joint tau_img_ms-and-VRAM frontier unexpectedly changed: "
                f"{flickr_changes}"
            )
        if pet_changes != expected_pet_change:
            raise ValidationError(
                "Pet joint tau_img_ms-and-VRAM frontier did not show the expected "
                f"single addition: {pet_changes}"
            )

    task_joint_sets: Dict[str, set] = defaultdict(set)
    for record in records:
        if record.flags["pareto_latency_vram"]:
            task_joint_sets[record.task].add((record.model_id, record.resolution))
    intersection = set.intersection(*task_joint_sets.values())
    if not expanded and intersection != EXPECTED_CROSS_TASK_INTERSECTION:
        raise ValidationError(
            "cross-task joint-frontier intersection mismatch: "
            f"expected {sorted(EXPECTED_CROSS_TASK_INTERSECTION)}, "
            f"got {sorted(intersection)}"
        )

    frontier_counts: Dict[str, Dict[str, int]] = {}
    for flag_name, _ in FRONTIER_SPECS:
        frontier_counts[flag_name] = {
            task: sum(
                record.flags[flag_name]
                for record in records
                if record.task == task
            )
            for task in sorted({record.task for record in records})
        }

    return {
        "frontier_counts": frontier_counts,
        "flickr_joint_frontier_changed": False if not expanded else "not_compared",
        "pet_joint_frontier_additions": (
            [{"model_id": "clip_vit_b16_openai", "resolution": 224}]
            if not expanded else []
        ),
        "legacy_frontier_comparison": (
            "not_applicable_expanded_rerun" if expanded else "validated_baseline_recomputation"
        ),
        "cross_task_joint_frontier_intersection": [
            {"model_id": model_id, "resolution": resolution}
            for model_id, resolution in sorted(intersection)
        ],
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    temporary.replace(path)


def csv_text(fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def processed_rows(
    records: Sequence[Record], input_fieldnames: Sequence[str]
) -> Tuple[List[str], List[Dict[str, str]]]:
    output_fields = list(input_fieldnames)
    if "evaluated_scoring_path" not in output_fields:
        architecture_index = output_fields.index("architecture_family")
        output_fields.insert(architecture_index + 1, "evaluated_scoring_path")
    if "tau_img_ms" not in output_fields:
        throughput_index = output_fields.index("throughput_examples_s")
        output_fields.insert(throughput_index + 1, "tau_img_ms")
    legacy_fields = [
        "legacy_pareto_latency_batch_p50",
        "legacy_pareto_latency_vram_batch_p50",
    ]
    for legacy_field in legacy_fields:
        if legacy_field not in output_fields:
            output_fields.append(legacy_field)

    rows: List[Dict[str, str]] = []
    for record in records:
        row = dict(record.raw)
        row["evaluated_scoring_path"] = EVALUATED_SCORING_PATHS[record.model_id]
        row["tau_img_ms"] = decimal_string(record.numeric["tau_img_ms"])
        row["legacy_pareto_latency_batch_p50"] = record.raw["pareto_latency"]
        row["legacy_pareto_latency_vram_batch_p50"] = record.raw[
            "pareto_latency_vram"
        ]
        for flag_name, _ in FRONTIER_SPECS:
            row[flag_name] = "True" if record.flags[flag_name] else "False"
        rows.append(row)
    return output_fields, rows


def frontier_rows(records: Sequence[Record], flag_name: str) -> List[Dict[str, str]]:
    fields = (
        "model_id",
        "task",
        "dataset",
        "resolution",
        "batch_size",
        "text_batch_size",
        "precision",
        "performance_metric",
        "performance_value",
        "tau_img_ms",
        "peak_vram_gb",
        "parameter_count",
    )
    rows = []
    for record in records:
        if not record.flags[flag_name]:
            continue
        row = {field_name: record.raw.get(field_name, "") for field_name in fields}
        row["tau_img_ms"] = decimal_string(record.numeric["tau_img_ms"])
        rows.append(row)
    return rows


def intersection_rows(records: Sequence[Record]) -> List[Dict[str, str]]:
    by_config: Dict[Tuple[str, int], Dict[str, Record]] = defaultdict(dict)
    for record in records:
        if record.flags["pareto_latency_vram"]:
            by_config[(record.model_id, record.resolution)][record.task] = record

    task_names = {record.task for record in records}
    intersection = {
        key for key, task_records in by_config.items()
        if set(task_records) == task_names
    }
    rows = []
    for model_id, resolution in sorted(intersection):
        task_records = by_config[(model_id, resolution)]
        classification = task_records["zeroshot_classification"]
        retrieval = task_records["image_text_retrieval"]
        rows.append(
            {
                "model_id": model_id,
                "resolution": str(resolution),
                "precision": classification.raw["precision"],
                "image_batch_size": classification.raw["batch_size"],
                "retrieval_text_batch_size": retrieval.raw["text_batch_size"],
                "classification_performance_value": classification.raw[
                    "performance_value"
                ],
                "classification_tau_img_ms": decimal_string(
                    classification.numeric["tau_img_ms"]
                ),
                "classification_peak_vram_gb": classification.raw[
                    "peak_vram_gb"
                ],
                "retrieval_performance_value": retrieval.raw["performance_value"],
                "retrieval_tau_img_ms": decimal_string(
                    retrieval.numeric["tau_img_ms"]
                ),
                "retrieval_peak_vram_gb": retrieval.raw["peak_vram_gb"],
            }
        )
    return rows


def record_sort_key(record: Record) -> Tuple[int, int]:
    try:
        model_position = MODEL_ORDER.index(record.model_id)
    except ValueError as exc:
        raise ValidationError(f"model order missing {record.model_id!r}") from exc
    return model_position, record.resolution


def render_latex_table(records: Sequence[Record], task: str) -> str:
    selected = sorted(
        (record for record in records if record.task == task),
        key=record_sort_key,
    )
    lines = [
        "% Generated by analysis/regenerate_results.py; do not edit manually.",
        r"\noindent\resizebox{0.98\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        (
            r"Model & Res. & $B_{\rm img}$ & Perf. (\%) & 95\% CI & "
            r"Amortised latency (ms/image) & Batch $p50$ (ms) & VRAM (GB) & Params (M) \\"
        ),
        r"\hline",
    ]
    for record in selected:
        performance = record.numeric["performance_value"] * Decimal("100")
        ci_lower = record.numeric["ci_lower_95"] * Decimal("100")
        ci_upper = record.numeric["ci_upper_95"] * Decimal("100")
        parameters_m = record.numeric["parameter_count"] / Decimal("1000000")
        values = [
            READABLE_NAMES[record.model_id],
            str(record.resolution),
            record.raw["batch_size"],
            f"{performance:.2f}",
            f"[{ci_lower:.2f}, {ci_upper:.2f}]",
            f"{record.numeric['tau_img_ms']:.3f}",
            f"{record.numeric['latency_p50_ms']:.1f}",
            f"{record.numeric['peak_vram_gb']:.2f}",
            f"{parameters_m:.1f}",
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}", ""])
    return "\n".join(lines)


def plot_family(record: Record) -> str:
    if record.model_id.startswith("blip"):
        return "BLIP"
    if record.model_id.startswith("paligemma"):
        return "PaliGemma"
    if record.model_id.startswith("siglip"):
        return "SigLIP"
    if record.model_id.startswith("mobileclip"):
        return "MobileCLIP"
    return "CLIP"


SCORING_PATH_LABELS = {
    "CLIP": "CLIP dual embedding",
    "SigLIP": "SigLIP dual embedding",
    "MobileCLIP": "MobileCLIP dual embedding",
    "BLIP": "BLIP ITC dual embedding",
    "PaliGemma": "PaliGemma adapted likelihood",
}

RESOLUTION_COLOURS = {
    160: "#0072B2",
    224: "#009E73",
    336: "#E69F00",
    384: "#56B4E9",
    448: "#CC79A7",
}

FAMILY_MARKERS = {
    "CLIP": "o",
    "SigLIP": "s",
    "MobileCLIP": "D",
    "BLIP": "^",
    "PaliGemma": "P",
}

PUBLICATION_PATH_LABELS = {
    "CLIP": "CLIP",
    "SigLIP": "SigLIP",
    "MobileCLIP": "MobileCLIP",
    "BLIP": "BLIP ITC",
    "PaliGemma": "PaliGemma diagnostic",
}

TASK_TITLES = {
    "zeroshot_classification": "(a) Oxford-IIIT Pet classification",
    "image_text_retrieval": "(b) Flickr30k retrieval",
}

TASK_METRICS = {
    "zeroshot_classification": "Top-1 accuracy (%)",
    "image_text_retrieval": "Mean Recall@1 (%)",
}


def parameter_token_proxy_value(record: Record) -> Decimal:
    """Return the recorded proxy, or the documented parameter-only fallback."""
    raw_proxy = (record.raw.get("parameter_token_proxy") or "").strip()
    if raw_proxy:
        return Decimal(raw_proxy)
    if record.model_id == "mobileclip_s1_datacompdr":
        return record.numeric["parameter_count"]
    raise ValidationError(
        f"row {record.index + 2}: parameter_token_proxy is missing for "
        f"{record.model_id}"
    )


def save_pdf_figure(fig, path: Path) -> None:
    """Write a deterministic vector figure without wall-clock metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format="pdf",
        metadata={
            "Creator": "analysis/regenerate_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
        bbox_inches="tight",
    )
    temporary.replace(path)


def configure_publication_matplotlib() -> None:
    """Apply deterministic typography suitable for a two-column paper."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dicta-mpl-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "dicta-2026-analysis-v2",
        }
    )


def publication_legend_handles(records: Sequence[Record], include_frontier: bool):
    """Return colour- and shape-redundant legend handles."""
    from matplotlib.lines import Line2D

    resolution_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colour,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5.5,
            label=f"{resolution} px",
        )
        for resolution, colour in RESOLUTION_COLOURS.items()
    ]
    present_families = {plot_family(record) for record in records}
    path_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="black",
            markerfacecolor="white",
            markeredgewidth=0.6,
            markersize=5.5,
            label=PUBLICATION_PATH_LABELS[family],
        )
        for family, marker in FAMILY_MARKERS.items()
        if family in present_families
    ]
    if include_frontier:
        path_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="black",
                markerfacecolor="white",
                markeredgewidth=1.5,
                markersize=6,
                label="Joint latency--VRAM frontier",
            )
        )
    return resolution_handles, path_handles


def render_operational_tradeoff_figure(
    records: Sequence[Record], path: Path
) -> None:
    """Render the primary dual-embedding performance-versus-latency figure."""
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    primary_records = [
        record for record in records if not record.model_id.startswith("paligemma")
    ]
    tasks = ("zeroshot_classification", "image_text_retrieval")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.25))
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.93,
        bottom=0.325,
        wspace=0.27,
    )
    annotation_styles = {
        ("siglip_b16_webli", "zeroshot_classification", 224): {
            "label": "SigLIP-B/16",
            "xytext": (-12, -12),
            "ha": "right",
            "va": "top",
        },
        ("mobileclip_s1_datacompdr", "zeroshot_classification", 224): {
            "label": "MobileCLIP-S1",
            "xytext": (12, -12),
            "ha": "left",
            "va": "top",
        },
        ("blip_itm_large_coco", "image_text_retrieval", 384): {
            "label": "BLIP-1 large",
            "xytext": (-8, 6),
            "ha": "right",
            "va": "bottom",
        },
        ("siglip_b16_webli", "image_text_retrieval", 224): {
            "label": "SigLIP-B/16",
            "xytext": (-12, -10),
            "ha": "right",
            "va": "top",
        },
    }

    for ax, task in zip(axes, tasks):
        selected = sorted(
            (record for record in primary_records if record.task == task),
            key=record_sort_key,
        )
        for record in selected:
            x_value = float(record.numeric["tau_img_ms"])
            y_value = float(record.numeric["performance_value"] * Decimal("100"))
            lower = float(
                (record.numeric["performance_value"] - record.numeric["ci_lower_95"])
                * Decimal("100")
            )
            upper = float(
                (record.numeric["ci_upper_95"] - record.numeric["performance_value"])
                * Decimal("100")
            )
            family = plot_family(record)
            ax.errorbar(
                x_value,
                y_value,
                yerr=([lower], [upper]),
                fmt=FAMILY_MARKERS[family],
                color=RESOLUTION_COLOURS[record.resolution],
                markerfacecolor=RESOLUTION_COLOURS[record.resolution],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=5.8,
                ecolor=RESOLUTION_COLOURS[record.resolution],
                elinewidth=0.65,
                capsize=1.7,
                alpha=0.95,
                zorder=3,
            )
            key = (record.model_id, task, record.resolution)
            if key in annotation_styles:
                annotation = annotation_styles[key]
                ax.annotate(
                    annotation["label"],
                    (x_value, y_value),
                    xytext=annotation["xytext"],
                    textcoords="offset points",
                    fontsize=8,
                    ha=annotation["ha"],
                    va=annotation["va"],
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#555555",
                        "linewidth": 0.45,
                    },
                    zorder=5,
                )

        ax.set_title(TASK_TITLES[task])
        ax.set_xlabel("Amortised latency (ms/image; log scale)")
        ax.set_ylabel(TASK_METRICS[task])
        ax.set_xscale("log")
        if task == "image_text_retrieval":
            ax.set_ylim(top=94.5)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)

    resolution_handles, path_handles = publication_legend_handles(
        primary_records, include_frontier=False
    )
    fig.legend(
        handles=resolution_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.155),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    fig.legend(
        handles=path_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.05,
    )
    save_pdf_figure(fig, path)
    plt.close(fig)


def render_combined_resolution_figure(
    records: Sequence[Record], path: Path
) -> None:
    """Render directly labelled checkpoint sweeps with resolution-coloured points."""
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    tasks = ("zeroshot_classification", "image_text_retrieval")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.0))
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.92,
        bottom=0.31,
        wspace=0.27,
    )
    line_styles = (
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    )
    label_offsets = {
        ("zeroshot_classification", "clip_rn50_openai"): -3,
        ("zeroshot_classification", "clip_vit_b32_openai"): 0,
        ("zeroshot_classification", "clip_vit_b16_openai"): 0,
        ("zeroshot_classification", "clip_vit_l14_openai"): 4,
        ("zeroshot_classification", "siglip_b16_webli"): -4,
        ("zeroshot_classification", "blip_itm_base_coco"): -4,
        ("zeroshot_classification", "blip_itm_large_coco"): 4,
        ("image_text_retrieval", "clip_rn50_openai"): 5,
        ("image_text_retrieval", "clip_vit_b32_openai"): -6,
        ("image_text_retrieval", "clip_vit_b16_openai"): 0,
        ("image_text_retrieval", "clip_vit_l14_openai"): 0,
        ("image_text_retrieval", "siglip_b16_webli"): 0,
        ("image_text_retrieval", "blip_itm_base_coco"): -4,
        ("image_text_retrieval", "blip_itm_large_coco"): 4,
    }

    plotted_records: List[Record] = []
    for ax, task in zip(axes, tasks):
        by_model: Dict[str, List[Record]] = defaultdict(list)
        for record in records:
            if record.task == task and not record.model_id.startswith("paligemma"):
                by_model[record.model_id].append(record)
        sweep_ids = [
            model_id
            for model_id in MODEL_ORDER
            if len(by_model.get(model_id, [])) >= 2
        ]
        for index, model_id in enumerate(sweep_ids):
            model_records = sorted(
                by_model[model_id], key=lambda item: item.resolution
            )
            plotted_records.extend(model_records)
            x_values = [record.resolution for record in model_records]
            y_values = [
                float(record.numeric["performance_value"] * Decimal("100"))
                for record in model_records
            ]
            ax.plot(
                x_values,
                y_values,
                color="#666666",
                linestyle=line_styles[index % len(line_styles)],
                linewidth=0.9,
                alpha=0.8,
                zorder=1,
            )
            for record, y_value in zip(model_records, y_values):
                ax.plot(
                    record.resolution,
                    y_value,
                    marker=FAMILY_MARKERS[plot_family(record)],
                    linestyle="none",
                    markerfacecolor=RESOLUTION_COLOURS[record.resolution],
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    markersize=5.6,
                    zorder=3,
                )
            final_record = model_records[-1]
            ax.annotate(
                READABLE_NAMES[model_id],
                (final_record.resolution, y_values[-1]),
                xytext=(5, label_offsets[(task, model_id)]),
                textcoords="offset points",
                fontsize=8,
                va="center",
                zorder=4,
            )

        ax.axvline(224, color="#555555", linestyle="--", linewidth=0.75)
        ax.axvline(384, color="#555555", linestyle=":", linewidth=0.75)
        ax.set_title(TASK_TITLES[task])
        ax.set_xlabel("Evaluation resolution (pixels)")
        ax.set_ylabel(TASK_METRICS[task])
        ax.set_xticks((160, 224, 336, 384, 448))
        ax.set_xlim(145, 610)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)

    resolution_handles, path_handles = publication_legend_handles(
        plotted_records, include_frontier=False
    )
    fig.legend(
        handles=resolution_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    fig.legend(
        handles=path_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    save_pdf_figure(fig, path)
    plt.close(fig)


def render_figure_table_trace(
    input_path: Path,
    input_checksum: str,
    records: Sequence[Record],
) -> str:
    """Return a machine-readable claim trace for generated visual artifacts."""
    try:
        input_display = input_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        input_display = input_path.name
    transformation = {
        "script": "analysis/regenerate_results.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    common_source = {
        "path": input_display,
        "sha256": input_checksum,
        "row_count": len(records),
    }
    classification_count = sum(
        record.task == "zeroshot_classification" for record in records
    )
    retrieval_count = sum(
        record.task == "image_text_retrieval" for record in records
    )
    retrieval_ci_methods = {
        record.raw.get("ci_method", "")
        for record in records
        if record.task == "image_text_retrieval"
    }
    if retrieval_ci_methods == {"image_grouped_bootstrap"}:
        retrieval_interval_summary = (
            "retrieval intervals are image-grouped bootstrap summaries"
        )
        retrieval_interval_limitation = (
            "retrieval intervals preserve each image and its five captions"
        )
    elif retrieval_ci_methods == {"wilson"}:
        retrieval_interval_summary = (
            "retrieval intervals are approximate marginal Wilson summaries"
        )
        retrieval_interval_limitation = (
            "retrieval Wilson intervals are approximate marginal summaries"
        )
    else:
        retrieval_interval_summary = (
            "retrieval interval methods are recorded in the processed summary"
        )
        retrieval_interval_limitation = (
            "retrieval interval interpretation depends on the recorded method"
        )
    trace = {
        "schema_version": 1,
        "source": common_source,
        "transformation": transformation,
        "artifacts": [
            {
                "id": "fig:operational-tradeoffs",
                "path": "figures/performance_vs_tau_img_both_tasks.pdf",
                "role": "primary manuscript figure",
                "claim": (
                    "Performance-cost distributions under the recorded batched "
                    "protocol, with deterministic joint latency--VRAM membership."
                ),
                "limitations": [
                    "tau_img_ms is amortised processing time, not batch-one response latency",
                    "classification intervals are descriptive bootstrap summaries",
                    retrieval_interval_summary,
                    "frontier membership uses point estimates only",
                ],
            },
            {
                "id": "fig:resolution-sweeps",
                "path": "figures/performance_vs_resolution_both_tasks.pdf",
                "role": "primary manuscript figure",
                "claim": (
                    "Within-checkpoint point estimates across the recorded resolution grid."
                ),
                "limitations": [
                    "PaliGemma checkpoints are excluded because they are distinct transfer checkpoints",
                    "MobileCLIP is excluded because only one resolution was recorded",
                    "connecting lines are descriptive and do not establish causality",
                ],
            },
            {
                "id": "fig:parameter-token-proxy-classification",
                "path": "figures/headline_performance_vs_proxy_classification.pdf",
                "role": "secondary generated figure",
                "claim": "Classification performance against a rough descriptive proxy.",
                "limitations": [
                    "not a FLOP count and not used for latency-frontier membership",
                    "MobileCLIP uses parameter count alone because no compatible visual-token count was retained",
                ],
            },
            {
                "id": "fig:parameter-token-proxy-retrieval",
                "path": "figures/headline_performance_vs_proxy_retrieval.pdf",
                "role": "secondary generated figure",
                "claim": "Retrieval performance against a rough descriptive proxy.",
                "limitations": [
                    "not a FLOP count and not used for latency-frontier membership",
                    "MobileCLIP uses parameter count alone because no compatible visual-token count was retained",
                ],
            },
            {
                "id": "tab:classification-results",
                "path": "tables/classification_results.tex",
                "role": "full generated appendix table",
                "claim": (
                    f"All {classification_count} recorded classification "
                    "configurations and derived frontiers."
                ),
                "limitations": ["intervals and frontier flags have distinct interpretations"],
            },
            {
                "id": "tab:retrieval-results",
                "path": "tables/retrieval_results.tex",
                "role": "full generated appendix table",
                "claim": (
                    f"All {retrieval_count} recorded retrieval configurations "
                    "and derived frontiers."
                ),
                "limitations": [retrieval_interval_limitation],
            },
        ],
    }
    return json.dumps(trace, indent=2, sort_keys=True) + "\n"


def render_headline_proxy_plot(
    records: Sequence[Record], task: str, path: Path
) -> None:
    """Plot performance against the parameter-token proxy with path-safe labels."""
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    annotation_keys = {
        "zeroshot_classification": {
            ("siglip_b16_webli", 224),
            ("mobileclip_s1_datacompdr", 224),
            ("paligemma_3b_mix_224", 224),
            ("paligemma_3b_mix_448", 448),
        },
        "image_text_retrieval": {
            ("siglip_b16_webli", 224),
            ("blip_itm_base_coco", 448),
            ("blip_itm_large_coco", 336),
        },
    }

    selected = sorted(
        (record for record in records if record.task == task),
        key=record_sort_key,
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.25), constrained_layout=True)
    for record in selected:
        x_value = float(parameter_token_proxy_value(record))
        y_value = float(record.numeric["performance_value"] * Decimal("100"))
        ax.scatter(
            x_value,
            y_value,
            color=RESOLUTION_COLOURS[record.resolution],
            marker=FAMILY_MARKERS[plot_family(record)],
            edgecolor="black",
            linewidth=0.35,
            s=32,
            zorder=3,
        )
        if (record.model_id, record.resolution) in annotation_keys[task]:
            ax.annotate(
                f"{READABLE_NAMES[record.model_id]} @ {record.resolution}",
                (x_value, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

    title = (
        "Oxford-IIIT Pet classification"
        if task == "zeroshot_classification"
        else "Flickr30k retrieval"
    )
    metric = (
        "Top-1 accuracy (%)"
        if task == "zeroshot_classification"
        else "Mean Recall@1 (%)"
    )
    ax.set_title(title)
    ax.set_xlabel("Parameter-token inference proxy (log scale)")
    ax.set_ylabel(metric)
    ax.set_xscale("log")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)

    resolution_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colour,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=f"{resolution} px",
        )
        for resolution, colour in RESOLUTION_COLOURS.items()
    ]
    path_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="black",
            markerfacecolor="white",
            label=SCORING_PATH_LABELS[family],
        )
        for family, marker in FAMILY_MARKERS.items()
        if any(plot_family(record) == family for record in selected)
    ]
    first_legend = ax.legend(
        handles=resolution_handles,
        title="Resolution",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=path_handles,
        title="Evaluated scoring path",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0,
    )
    save_pdf_figure(fig, path)
    plt.close(fig)


def validate_written_summary(path: Path, records: Sequence[Record]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(records):
        raise ValidationError(
            f"processed summary row count mismatch: {len(rows)} != {len(records)}"
        )
    for record, row in zip(records, rows):
        stored_tau = parse_decimal(
            row["tau_img_ms"], "tau_img_ms", record.index + 2
        )
        if abs(stored_tau - record.numeric["tau_img_ms"]) > Decimal("1e-12"):
            raise ValidationError(
                f"processed row {record.index + 2}: tau_img_ms is mismatched"
            )
        for flag_name, _ in FRONTIER_SPECS:
            if parse_bool(row[flag_name], flag_name, record.index + 2) != record.flags[
                flag_name
            ]:
                raise ValidationError(
                    f"processed row {record.index + 2}: {flag_name} is stale"
                )


def write_outputs(
    records: Sequence[Record],
    input_fieldnames: Sequence[str],
    input_path: Path,
    input_checksum: str,
    output_dir: Path,
    diagnostics: Dict[str, object],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields, summary_rows = processed_rows(records, input_fieldnames)
    summary_path = output_dir / "summary_with_tau_and_pareto.csv"
    atomic_write_text(summary_path, csv_text(summary_fields, summary_rows))

    frontier_fields = (
        "model_id",
        "task",
        "dataset",
        "resolution",
        "batch_size",
        "text_batch_size",
        "precision",
        "performance_metric",
        "performance_value",
        "tau_img_ms",
        "peak_vram_gb",
        "parameter_count",
    )
    for flag_name, _ in FRONTIER_SPECS:
        atomic_write_text(
            output_dir / f"{flag_name}.csv",
            csv_text(frontier_fields, frontier_rows(records, flag_name)),
        )

    intersection_fields = (
        "model_id",
        "resolution",
        "precision",
        "image_batch_size",
        "retrieval_text_batch_size",
        "classification_performance_value",
        "classification_tau_img_ms",
        "classification_peak_vram_gb",
        "retrieval_performance_value",
        "retrieval_tau_img_ms",
        "retrieval_peak_vram_gb",
    )
    atomic_write_text(
        output_dir / "cross_task_joint_frontier_intersection.csv",
        csv_text(intersection_fields, intersection_rows(records)),
    )

    atomic_write_text(
        output_dir / "tables" / "classification_results.tex",
        render_latex_table(records, "zeroshot_classification"),
    )
    atomic_write_text(
        output_dir / "tables" / "retrieval_results.tex",
        render_latex_table(records, "image_text_retrieval"),
    )
    render_operational_tradeoff_figure(
        records,
        output_dir / "figures" / "performance_vs_tau_img_both_tasks.pdf",
    )
    render_headline_proxy_plot(
        records,
        "zeroshot_classification",
        output_dir / "figures" / "headline_performance_vs_proxy_classification.pdf",
    )
    render_headline_proxy_plot(
        records,
        "image_text_retrieval",
        output_dir / "figures" / "headline_performance_vs_proxy_retrieval.pdf",
    )
    render_combined_resolution_figure(
        records,
        output_dir / "figures" / "performance_vs_resolution_both_tasks.pdf",
    )
    atomic_write_text(
        output_dir / "figure_table_trace.json",
        render_figure_table_trace(input_path, input_checksum, records),
    )

    validate_written_summary(summary_path, records)
    if sha256_file(input_path) != input_checksum:
        raise ValidationError("canonical input changed during regeneration")

    generated_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    output_checksums = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in generated_paths
    }
    try:
        input_display = input_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        input_display = input_path.name
    manifest = {
        "schema_version": 2,
        "analysis": "throughput-normalised operational Pareto frontiers",
        "input": {
            "path": input_display,
            "sha256": input_checksum,
            "row_count": len(records),
            "gpu_name": EXPECTED_GPU,
        },
        "formula": "tau_img_ms = 1000 / throughput_examples_s",
        "frontier_rule": (
            "point-estimate performance maximised; listed costs minimised; "
            "at least one inequality strict"
        ),
        "diagnostics": diagnostics,
        "artifact_trace": "figure_table_trace.json",
        "output_sha256": output_checksums,
    }
    atomic_write_text(
        output_dir / "analysis_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return output_checksums


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="canonical 50- or 54-row CSV (default: authoritative 54-row summary)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for derived artifacts (default: analysis/processed)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if input_path == output_dir or output_dir in input_path.parents:
        raise ValidationError("output directory must not contain the canonical input")

    records, fieldnames, input_checksum = load_and_validate(input_path)
    recompute_frontiers(records)
    diagnostics = validate_derived_values(records)
    output_checksums = write_outputs(
        records,
        fieldnames,
        input_path,
        input_checksum,
        output_dir,
        diagnostics,
    )

    print(f"validated_rows={len(records)}")
    print(f"input_sha256={input_checksum}")
    print(
        "joint_frontier_counts="
        + json.dumps(
            diagnostics["frontier_counts"]["pareto_latency_vram"],
            sort_keys=True,
        )
    )
    print(
        "cross_task_joint_intersection="
        + str(len(diagnostics["cross_task_joint_frontier_intersection"]))
    )
    print(f"generated_files={len(output_checksums) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
