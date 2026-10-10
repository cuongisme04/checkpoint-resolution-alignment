#!/usr/bin/env python3
"""Focused post-review analysis for grouped uncertainty and resolution mismatch.

This script is intentionally separate from ``regenerate_results.py`` so the
locked 50-row release-candidate analysis remains reproducible.  It accepts the
expanded RunPod outputs when available and can already run in preflight mode on
the current summary without inventing BLIP-384 outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "results" / "historical" / "summary_50.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "processed" / "mock_review"
DEFAULT_SEED = 20260714
DEFAULT_REPETITIONS = 10_000
BLIP_MODELS = ("blip_itm_base_coco", "blip_itm_large_coco")
REFERENCE_BASIS = {
    "blip_itm_base_coco": "downstream config.",
    "blip_itm_large_coco": "downstream config.",
    "clip_rn50_openai": "release config.",
    "clip_vit_b16_openai": "release config.",
    "clip_vit_b32_openai": "release config.",
    "clip_vit_l14_openai": "release config.",
    "mobileclip_s1_datacompdr": "release config.",
    "paligemma_3b_mix_224": "resolution-specific ckpt.",
    "paligemma_3b_mix_448": "resolution-specific ckpt.",
    "siglip_b16_webli": "release config.",
}
DISPLAY_MODEL = {
    "blip_itm_base_coco": "BLIP-1 base",
    "blip_itm_large_coco": "BLIP-1 large",
    "clip_rn50_openai": "CLIP RN50",
    "clip_vit_b16_openai": "CLIP B/16",
    "clip_vit_b32_openai": "CLIP B/32",
    "clip_vit_l14_openai": "CLIP L/14",
    "mobileclip_s1_datacompdr": "MobileCLIP-S1",
    "paligemma_3b_mix_224": "PaliGemma-224",
    "paligemma_3b_mix_448": "PaliGemma-448",
    "siglip_b16_webli": "SigLIP B/16",
}


class AnalysisError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--classification-outcomes", type=Path)
    parser.add_argument("--retrieval-outcomes", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all BLIP 336/384/448 rows and paired outcomes.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise AnalysisError(f"input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AnalysisError(f"input has no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = ((position + 1) + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average
        position = end
    return ranks


def spearman_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_ranks = average_ranks(x_values)
    y_ranks = average_ranks(y_values)
    x_mean = statistics.fmean(x_ranks)
    y_mean = statistics.fmean(y_ranks)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in x_ranks))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in y_ranks))
    if x_scale == 0 or y_scale == 0:
        return None
    return numerator / (x_scale * y_scale)


def mismatch_and_regret_rows(summary_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row_number, row in enumerate(summary_rows, start=2):
        required = ("model_id", "task", "dataset", "performance_metric", "performance_value", "resolution", "native_resolution")
        missing = [field for field in required if not (row.get(field) or "").strip()]
        if missing:
            raise AnalysisError(f"summary row {row_number} is missing {', '.join(missing)}")
        key = (row["model_id"], row["task"], row["dataset"], row["performance_metric"])
        grouped[key].append(row)

    detail = []
    sweep_rows = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["resolution"]))
        resolutions = [int(row["resolution"]) for row in rows]
        if len(set(resolutions)) != len(resolutions):
            raise AnalysisError(f"duplicate resolution within sweep {key}: {resolutions}")
        references = {int(row["native_resolution"]) for row in rows}
        if len(references) != 1:
            raise AnalysisError(f"reference resolution changed within sweep {key}: {references}")
        reference = references.pop()
        if key[0] not in REFERENCE_BASIS:
            raise AnalysisError(f"missing reference provenance for {key[0]}")
        values = [float(row["performance_value"]) for row in rows]
        best = max(values)
        if best <= 0:
            raise AnalysisError(f"non-positive best performance in sweep {key}")
        mismatches = [abs(math.log2(int(row["resolution"]) / reference)) for row in rows]
        regrets = [100.0 * (best - value) / best for value in values]
        correlation = spearman_correlation(mismatches, regrets) if len(rows) >= 3 else None
        minimum_mismatch = min(mismatches)
        closest_is_best = any(
            mismatch == minimum_mismatch and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-15)
            for mismatch, value in zip(mismatches, values)
        )
        status = "included" if len(rows) >= 3 and correlation is not None else (
            "excluded_constant" if len(rows) >= 3 else "excluded_fewer_than_three"
        )
        peak_resolutions = [
            resolution
            for resolution, value in zip(resolutions, values)
            if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-15)
        ]
        for row, mismatch, regret in zip(rows, mismatches, regrets):
            detail.append({
                "model_id": key[0],
                "task": key[1],
                "dataset": key[2],
                "performance_metric": key[3],
                "reference_resolution": reference,
                "evaluation_resolution": int(row["resolution"]),
                "resolution_mismatch_log2": f"{mismatch:.12g}",
                "performance_value": f"{float(row['performance_value']):.17g}",
                "normalized_regret_percent": f"{regret:.12g}",
                "is_best_point_estimate": int(math.isclose(float(row["performance_value"]), best, rel_tol=0.0, abs_tol=1e-15)),
                "is_closest_to_reference": int(mismatch == minimum_mismatch),
            })
        sweep_rows.append({
            "model_id": key[0],
            "task": key[1],
            "dataset": key[2],
            "performance_metric": key[3],
            "resolution_count": len(rows),
            "reference_resolution": reference,
            "reference_basis": REFERENCE_BASIS[key[0]],
            "peak_resolution": "/".join(str(value) for value in peak_resolutions),
            "spearman_mismatch_regret": "" if correlation is None else f"{correlation:.12g}",
            "aggregation_status": status,
            "closest_to_reference_has_best_point_estimate": int(closest_is_best),
        })
    return detail, sweep_rows


def permutation_median_correlation_null(
    detail_rows: list[dict],
    sweep_rows: list[dict],
    *,
    seed: int,
    repetitions: int,
) -> dict:
    """Test the fixed-reference association by permuting regret within sweeps."""
    eligible_keys = {
        (row["model_id"], row["task"], row["dataset"], row["performance_metric"])
        for row in sweep_rows
        if row["aggregation_status"] == "included"
    }
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in detail_rows:
        key = (row["model_id"], row["task"], row["dataset"], row["performance_metric"])
        if key in eligible_keys:
            grouped[key].append(row)
    if set(grouped) != eligible_keys:
        raise AnalysisError("eligible mismatch sweeps do not match detailed regret rows")

    ordered = []
    observed_correlations = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["evaluation_resolution"]))
        mismatches = [float(row["resolution_mismatch_log2"]) for row in rows]
        regrets = [float(row["normalized_regret_percent"]) for row in rows]
        observed = spearman_correlation(mismatches, regrets)
        if observed is None:
            raise AnalysisError(f"eligible sweep has undefined correlation: {key}")
        ordered.append((mismatches, regrets))
        observed_correlations.append(observed)

    observed_median = statistics.median(observed_correlations)
    rng = np.random.default_rng(seed)
    null_medians = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        correlations = []
        for mismatches, regrets in ordered:
            permuted = rng.permutation(regrets).tolist()
            correlation = spearman_correlation(mismatches, permuted)
            if correlation is None:
                raise AnalysisError("permutation unexpectedly produced undefined correlation")
            correlations.append(correlation)
        null_medians[index] = statistics.median(correlations)
    lower, upper = np.quantile(null_medians, [0.025, 0.975])
    exceedances = int(np.count_nonzero(null_medians >= observed_median - 1e-15))
    return {
        "permutation_null": "within-sweep regret-label permutation with fixed references",
        "permutation_observed_median_spearman": observed_median,
        "permutation_null_median": float(np.median(null_medians)),
        "permutation_null_ci_lower_95": float(lower),
        "permutation_null_ci_upper_95": float(upper),
        "permutation_one_sided_p": (exceedances + 1) / (repetitions + 1),
        "permutation_seed": seed,
        "permutation_repetitions": repetitions,
    }


def render_alignment_sweep_table(sweep_rows: list[dict]) -> str:
    """Render every visible sweep, including explicit short-sweep exclusions."""
    lines = [
        r"\begin{tabular}{@{}llrrrrcl@{}}",
        r"\hline",
        r"Checkpoint & Task & $n_R$ & $R_{\mathrm{ref}}$ & $R_{\mathrm{peak}}$ & $\rho$ & Closest best & Reference basis \\",
        r"\hline",
    ]
    for row in sweep_rows:
        model_id = row["model_id"]
        model = DISPLAY_MODEL.get(model_id, model_id.replace("_", r"\_"))
        task = "Pet" if row["task"] == "zeroshot_classification" else "Flickr"
        correlation = row["spearman_mismatch_regret"] or r"--"
        if correlation not in (r"--", ""):
            correlation = f"{float(correlation):.2f}"
        closest = "yes" if int(row["closest_to_reference_has_best_point_estimate"]) else "no"
        basis = row["reference_basis"]
        lines.append(
            f"{model} & {task} & {row['resolution_count']} & "
            f"{row['reference_resolution']} & {row['peak_resolution']} & "
            f"{correlation} & {closest} & {basis} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def clustered_median_correlation_interval(
    sweep_rows: list[dict], *, seed: int, repetitions: int
) -> dict:
    eligible = [row for row in sweep_rows if row["aggregation_status"] == "included"]
    if not eligible:
        return {
            "eligible_sweep_count": 0,
            "checkpoint_count": 0,
            "median_spearman": None,
            "bootstrap_ci_lower_95": None,
            "bootstrap_ci_upper_95": None,
            "seed": seed,
            "repetitions": repetitions,
        }
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in eligible:
        by_model[row["model_id"]].append(float(row["spearman_mismatch_regret"]))
    model_ids = sorted(by_model)
    observed = statistics.median(value for values in by_model.values() for value in values)
    rng = np.random.default_rng(seed)
    replicates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(model_ids), size=len(model_ids))
        values = [value for model_index in selected for value in by_model[model_ids[int(model_index)]]]
        replicates[index] = statistics.median(values)
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "eligible_sweep_count": len(eligible),
        "checkpoint_count": len(model_ids),
        "median_spearman": observed,
        "bootstrap_ci_lower_95": float(lower),
        "bootstrap_ci_upper_95": float(upper),
        "seed": seed,
        "repetitions": repetitions,
        "bootstrap_unit": "checkpoint/model_id",
    }


def group_outcomes(rows: list[dict]) -> dict[str, list[dict]]:
    required = {"configuration_id", "sample_id"}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        missing = required - set(row)
        if missing or any(not row.get(field) for field in required):
            raise AnalysisError(f"outcome row {row_number} lacks configuration_id/sample_id")
        grouped[row["configuration_id"]].append(row)
    for config_id, config_rows in grouped.items():
        ids = [row["sample_id"] for row in config_rows]
        if len(set(ids)) != len(ids):
            raise AnalysisError(f"duplicate sample_id within {config_id}")
        config_rows.sort(key=lambda row: row["sample_id"])
    return dict(grouped)


def retrieval_point_estimate(rows: list[dict]) -> float:
    image_hits = sum(int(row["image_to_text_hit_at_1"]) for row in rows)
    caption_hits = sum(int(row["text_to_image_hits_at_1"]) for row in rows)
    caption_count = sum(int(row["caption_count"]) for row in rows)
    if not rows or caption_count <= 0:
        raise AnalysisError("retrieval outcomes have an empty denominator")
    return 0.5 * (image_hits / len(rows) + caption_hits / caption_count)


def classification_bootstrap(
    rows: list[dict], *, seed: int, repetitions: int, chunk_size: int = 250
) -> dict:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    correct = np.asarray(
        [int(row["correct_top1"]) for row in ordered], dtype=np.float64
    )
    if len(ordered) == 0 or np.any((correct < 0) | (correct > 1)):
        raise AnalysisError("invalid compact classification outcomes")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        current = min(chunk_size, repetitions - offset)
        sampled = rng.integers(0, len(ordered), size=(current, len(ordered)))
        values[offset : offset + current] = correct[sampled].mean(axis=1)
        offset += current
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "performance_value": float(correct.mean()),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "ci_method": "sample_bootstrap",
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
        "effective_replicates": repetitions,
        "sample_count": len(ordered),
    }


def grouped_retrieval_bootstrap(
    rows: list[dict], *, seed: int, repetitions: int, chunk_size: int = 250
) -> dict:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    i2t = np.asarray([int(row["image_to_text_hit_at_1"]) for row in ordered], dtype=np.float64)
    t2i = np.asarray([int(row["text_to_image_hits_at_1"]) for row in ordered], dtype=np.float64)
    captions = np.asarray([int(row["caption_count"]) for row in ordered], dtype=np.float64)
    if len(ordered) == 0 or np.any(captions <= 0):
        raise AnalysisError("invalid grouped retrieval outcomes")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        current = min(chunk_size, repetitions - offset)
        sampled = rng.integers(0, len(ordered), size=(current, len(ordered)))
        i2t_recall = i2t[sampled].mean(axis=1)
        t2i_recall = t2i[sampled].sum(axis=1) / captions[sampled].sum(axis=1)
        values[offset : offset + current] = 0.5 * (i2t_recall + t2i_recall)
        offset += current
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "performance_value": retrieval_point_estimate(ordered),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "ci_method": "image_grouped_bootstrap",
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
        "effective_replicates": repetitions,
        "group_count": len(ordered),
        "caption_count": int(captions.sum()),
    }


def require_identical_sample_ids(config_rows: list[list[dict]]) -> list[str]:
    ordered_sets = [sorted(row["sample_id"] for row in rows) for rows in config_rows]
    if not ordered_sets or any(values != ordered_sets[0] for values in ordered_sets[1:]):
        raise AnalysisError("paired configurations do not have identical sample-ID sets")
    return ordered_sets[0]


def paired_blip_contrast(
    rows_336: list[dict],
    rows_384: list[dict],
    rows_448: list[dict],
    *,
    task: str,
    seed: int,
    repetitions: int,
    chunk_size: int = 250,
) -> dict:
    sample_ids = require_identical_sample_ids([rows_336, rows_384, rows_448])
    by_resolution = []
    for rows in (rows_336, rows_384, rows_448):
        index = {row["sample_id"]: row for row in rows}
        by_resolution.append([index[sample_id] for sample_id in sample_ids])

    if task == "zeroshot_classification":
        arrays = [np.asarray([int(row["correct_top1"]) for row in rows], dtype=np.float64) for rows in by_resolution]

        def sampled_metric(array, sampled):
            return array[sampled].mean(axis=1)

        observed = [float(array.mean()) for array in arrays]
    elif task == "image_text_retrieval":
        triples = []
        for rows in by_resolution:
            triples.append((
                np.asarray([int(row["image_to_text_hit_at_1"]) for row in rows], dtype=np.float64),
                np.asarray([int(row["text_to_image_hits_at_1"]) for row in rows], dtype=np.float64),
                np.asarray([int(row["caption_count"]) for row in rows], dtype=np.float64),
            ))

        def sampled_metric(triple, sampled):
            i2t, t2i, captions = triple
            return 0.5 * (i2t[sampled].mean(axis=1) + t2i[sampled].sum(axis=1) / captions[sampled].sum(axis=1))

        arrays = triples
        observed = [retrieval_point_estimate(rows) for rows in by_resolution]
    else:
        raise AnalysisError(f"unsupported paired task: {task}")

    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        current = min(chunk_size, repetitions - offset)
        sampled = rng.integers(0, len(sample_ids), size=(current, len(sample_ids)))
        metrics = [sampled_metric(array, sampled) for array in arrays]
        differences[offset : offset + current] = metrics[1] - 0.5 * (metrics[0] + metrics[2])
        offset += current
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "contrast_384_vs_336_448_mean": observed[1] - 0.5 * (observed[0] + observed[2]),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "ci_method": "paired_sample_bootstrap" if task == "zeroshot_classification" else "paired_image_grouped_bootstrap",
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
        "sample_group_count": len(sample_ids),
    }


def blip_contrast_status(
    summary_rows: list[dict],
    classification_groups: dict[str, list[dict]],
    retrieval_groups: dict[str, list[dict]],
    *,
    seed: int,
    repetitions: int,
    require_complete: bool,
) -> list[dict]:
    lookup = {}
    for row in summary_rows:
        if row.get("model_id") not in BLIP_MODELS:
            continue
        key = (row["model_id"], row["task"], int(row["resolution"]))
        if key in lookup:
            raise AnalysisError(f"duplicate BLIP summary configuration: {key}")
        lookup[key] = row
    output = []
    for model_id in BLIP_MODELS:
        for task in ("zeroshot_classification", "image_text_retrieval"):
            keys = [(model_id, task, resolution) for resolution in (336, 384, 448)]
            missing_rows = [resolution for (_, _, resolution), key in zip(keys, keys) if key not in lookup]
            base = {"model_id": model_id, "task": task}
            if missing_rows:
                if require_complete:
                    raise AnalysisError(f"missing BLIP rows for {model_id}/{task}: {missing_rows}")
                output.append({**base, "status": "pending_summary_rows", "missing_resolutions": ",".join(map(str, missing_rows))})
                continue
            groups = classification_groups if task == "zeroshot_classification" else retrieval_groups
            config_ids = [lookup[key].get("configuration_id", "") for key in keys]
            missing_outcomes = [resolution for resolution, config_id in zip((336, 384, 448), config_ids) if not config_id or config_id not in groups]
            if missing_outcomes:
                if require_complete:
                    raise AnalysisError(f"missing paired outcomes for {model_id}/{task}: {missing_outcomes}")
                point_values = [float(lookup[key]["performance_value"]) for key in keys]
                output.append({
                    **base,
                    "status": "pending_paired_outcomes",
                    "missing_resolutions": ",".join(map(str, missing_outcomes)),
                    "contrast_384_vs_336_448_mean": point_values[1] - 0.5 * (point_values[0] + point_values[2]),
                })
                continue
            contrast = paired_blip_contrast(
                groups[config_ids[0]], groups[config_ids[1]], groups[config_ids[2]],
                task=task, seed=seed, repetitions=repetitions
            )
            output.append({**base, "status": "complete", "missing_resolutions": "", **contrast})
    return output


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    try:
        summary = read_csv(args.summary)
        detail, sweeps = mismatch_and_regret_rows(summary)
        mismatch_summary = clustered_median_correlation_interval(
            sweeps, seed=args.seed, repetitions=args.repetitions
        )
        mismatch_summary.update(
            permutation_median_correlation_null(
                detail,
                sweeps,
                seed=args.seed + 1,
                repetitions=args.repetitions,
            )
        )
        mismatch_summary["visible_sweep_count"] = len(sweeps)
        eligible = [row for row in sweeps if row["aggregation_status"] == "included"]
        multi_resolution = [row for row in sweeps if int(row["resolution_count"]) >= 2]
        mismatch_summary["eligible_closest_to_reference_success_count"] = sum(
            int(row["closest_to_reference_has_best_point_estimate"]) for row in eligible
        )
        mismatch_summary["multi_resolution_sweep_count"] = len(multi_resolution)
        mismatch_summary["multi_resolution_closest_to_reference_success_count"] = sum(
            int(row["closest_to_reference_has_best_point_estimate"])
            for row in multi_resolution
        )
        mismatch_summary["excluded_singleton_sweep_count"] = sum(
            int(row["resolution_count"] == 1) for row in sweeps
        )
        mismatch_summary["excluded_two_resolution_sweep_count"] = sum(
            int(row["resolution_count"] == 2) for row in sweeps
        )

        classification_groups = group_outcomes(read_csv(args.classification_outcomes)) if args.classification_outcomes else {}
        retrieval_groups = group_outcomes(read_csv(args.retrieval_outcomes)) if args.retrieval_outcomes else {}
        if args.require_complete and classification_groups:
            if len(classification_groups) != 6:
                raise AnalysisError(
                    f"expected 6 compact Pet configurations, found {len(classification_groups)}"
                )
            require_identical_sample_ids(list(classification_groups.values()))
            if any(len(rows) != 3669 for rows in classification_groups.values()):
                raise AnalysisError("final Pet outcomes must retain 3,669 samples per configuration")
        if args.require_complete and retrieval_groups:
            if len(retrieval_groups) != 24:
                raise AnalysisError(
                    f"expected 24 grouped retrieval configurations, found {len(retrieval_groups)}"
                )
            require_identical_sample_ids(list(retrieval_groups.values()))
            if any(
                int(row["caption_count"]) != 5
                for rows in retrieval_groups.values()
                for row in rows
            ):
                raise AnalysisError("final Flickr30k outcomes must retain five captions per image")

        classification_bootstrap_rows = []
        bootstrap_rows = []
        summary_by_config = {row.get("configuration_id", ""): row for row in summary if row.get("configuration_id")}
        for config_id, rows in sorted(classification_groups.items()):
            result = classification_bootstrap(
                rows, seed=args.seed, repetitions=args.repetitions
            )
            if args.require_complete and result["sample_count"] != 3669:
                raise AnalysisError(
                    f"{config_id} has {result['sample_count']} classification samples"
                )
            summary_row = summary_by_config.get(config_id)
            if summary_row and not math.isclose(
                result["performance_value"],
                float(summary_row["performance_value"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AnalysisError(
                    f"classification outcomes do not reconstruct summary metric for {config_id}"
                )
            classification_bootstrap_rows.append({"configuration_id": config_id, **result})

        for config_id, rows in sorted(retrieval_groups.items()):
            result = grouped_retrieval_bootstrap(rows, seed=args.seed, repetitions=args.repetitions)
            if args.require_complete and (result["group_count"], result["caption_count"]) != (1000, 5000):
                raise AnalysisError(f"{config_id} has {result['group_count']} groups/{result['caption_count']} captions")
            summary_row = summary_by_config.get(config_id)
            if summary_row and not math.isclose(result["performance_value"], float(summary_row["performance_value"]), rel_tol=0.0, abs_tol=1e-12):
                raise AnalysisError(f"outcomes do not reconstruct summary metric for {config_id}")
            bootstrap_rows.append({"configuration_id": config_id, **result})

        contrasts = blip_contrast_status(
            summary,
            classification_groups,
            retrieval_groups,
            seed=args.seed,
            repetitions=args.repetitions,
            require_complete=args.require_complete,
        )
    except AnalysisError as exc:
        raise SystemExit(f"Mock-review analysis failed: {exc}") from exc

    write_csv(
        args.output_dir / "normalized_regret_rows.csv",
        detail,
        ["model_id", "task", "dataset", "performance_metric", "reference_resolution", "evaluation_resolution", "resolution_mismatch_log2", "performance_value", "normalized_regret_percent", "is_best_point_estimate", "is_closest_to_reference"],
    )
    write_csv(
        args.output_dir / "mismatch_sweep_correlations.csv",
        sweeps,
        ["model_id", "task", "dataset", "performance_metric", "resolution_count", "reference_resolution", "reference_basis", "peak_resolution", "spearman_mismatch_regret", "aggregation_status", "closest_to_reference_has_best_point_estimate"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mismatch_summary.json").write_text(json.dumps(mismatch_summary, indent=2, sort_keys=True) + "\n")
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "alignment_sweeps.tex").write_text(
        render_alignment_sweep_table(sweeps), encoding="utf-8"
    )
    write_csv(
        args.output_dir / "classification_bootstrap.csv",
        classification_bootstrap_rows,
        ["configuration_id", "performance_value", "ci_lower_95", "ci_upper_95", "ci_method", "bootstrap_seed", "bootstrap_repetitions", "effective_replicates", "sample_count"],
    )
    write_csv(
        args.output_dir / "retrieval_grouped_bootstrap.csv",
        bootstrap_rows,
        ["configuration_id", "performance_value", "ci_lower_95", "ci_upper_95", "ci_method", "bootstrap_seed", "bootstrap_repetitions", "effective_replicates", "group_count", "caption_count"],
    )
    contrast_fields = ["model_id", "task", "status", "missing_resolutions", "contrast_384_vs_336_448_mean", "ci_lower_95", "ci_upper_95", "ci_method", "bootstrap_seed", "bootstrap_repetitions", "sample_group_count"]
    write_csv(args.output_dir / "blip_384_contrasts.csv", contrasts, contrast_fields)
    print(
        f"Wrote mismatch analysis for {len(sweeps)} sweeps; "
        f"{len(classification_bootstrap_rows)} classification bootstrap configurations; "
        f"{len(bootstrap_rows)} grouped retrieval configurations; "
        f"{sum(row['status'] == 'complete' for row in contrasts)}/4 BLIP contrasts complete."
    )


if __name__ == "__main__":
    main()
