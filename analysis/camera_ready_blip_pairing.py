#!/usr/bin/env python3
"""Exploratory post-review BLIP Pet 336-minus-384 paired accuracy analysis.

Reads immutable saved outcomes; does not load models, datasets, or the network.
Run from any directory with Python >=3.10 and NumPy >=1.22. All outputs stay
inside this camera-ready copy. Existing canonical results are never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = "results/supporting/focused_run"
INPUTS = {
    "outcomes": (f"{RUN}/classification/compact_outcomes.csv", "d3ead7ffa981a1d2f61d03a459444a34878f4c7b725d4e2dbd7a77150a78ca75"),
    "configurations": (f"{RUN}/classification/configuration_index.csv", "535f24f787fc1ba787d1bf7c9e77703c431ea61b756c7e67a8a86f7e31d40cc6"),
    "canonical": ("results/canonical/dicta_mock_review_54_with_ci.csv", "4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7"),
    "registry": ("reproducibility/benchmark/configs/hf_models.json", "f399d295930a131228ed95730023f23864a2d4c44de074282e4282dc358cedc1"),
}
MODELS = ("blip_itm_base_coco", "blip_itm_large_coco")
SEED = 20260915
REPLICATES = 10000
EXPECTED_N = 3669
STATUS = "exploratory_post_review_not_preregistered"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_correctness(rows_a, rows_b):
    """Join by stable image ID, never input row position; reject lossy joins."""
    a = {row["sample_id"]: int(row["correct_top1"]) for row in rows_a}
    b = {row["sample_id"]: int(row["correct_top1"]) for row in rows_b}
    require(len(a) == len(rows_a) and len(b) == len(rows_b), "Duplicate image ID")
    require(a.keys() == b.keys() and bool(a), "Paired image-ID sets differ or are empty")
    ids = sorted(a)
    array_a = np.asarray([a[sample] for sample in ids], dtype=np.int8)
    array_b = np.asarray([b[sample] for sample in ids], dtype=np.int8)
    require(bool(np.isin(array_a, (0, 1)).all() and np.isin(array_b, (0, 1)).all()), "Nonbinary correctness")
    return ids, array_a, array_b


def bootstrap_sums(differences, *, seed=SEED, repetitions=REPLICATES):
    """Ordinary paired nonparametric bootstrap using direct image resampling.

    Each draw selects n image pairs with replacement, then sums the paired
    correctness differences. Integer sums avoid roundoff from two accuracy means.
    A fresh PCG64 generator with the same declared seed is used for each model;
    these are marginal intervals, not a joint cross-model bootstrap.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(differences)
    sums = np.empty(repetitions, dtype=np.int64)
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        sampled = rng.integers(0, n, size=(count, n))
        sums[start:start + count] = differences[sampled].sum(axis=1, dtype=np.int64)
    return sums


def interval_pp(sums, n):
    values = 100.0 * sums / n
    return np.quantile(values, (0.025, 0.975), method="linear")


def verify_pairing(rows336, rows384, a, b, original_sums):
    """Check image-ID pairing and signed arithmetic independently of row order."""
    shuffled336 = list(reversed(rows336))
    shuffled384 = rows384[31:] + rows384[:31]
    _, reordered_a, reordered_b = paired_correctness(shuffled336, shuffled384)
    require(np.array_equal(a, reordered_a) and np.array_equal(b, reordered_b), "Pair-order invariance failed")
    repeated_sums = bootstrap_sums(reordered_a - reordered_b)
    require(np.array_equal(original_sums, repeated_sums), "Deterministic bootstrap replay failed")

    _, swapped_a, swapped_b = paired_correctness(rows384, rows336)
    swapped_sums = bootstrap_sums(swapped_a - swapped_b, repetitions=256)
    require(np.array_equal(swapped_sums, -original_sums[:256]), "Signed contrast check failed")

    for changed in (rows384[:-1], rows384 + rows384[:1]):
        try:
            paired_correctness(rows336, changed)
        except ValueError:
            continue
        raise ValueError("Pairing accepted an incomplete or duplicated image set")
    return {
        "independent_input_row_reordering": "passed",
        "fixed_seed_bootstrap_exact_replay": "passed",
        "contrast_reversal_negates_bootstrap_samples": "passed",
        "missing_and_duplicate_image_ids_rejected": "passed",
    }


def main():
    paths = {}
    provenance = {}
    for label, (relative, expected) in INPUTS.items():
        path = ROOT / relative
        actual = sha256(path)
        require(actual == expected, f"Input checksum mismatch: {relative}")
        paths[label] = path
        provenance[label] = {"path": relative, "sha256": actual}

    configuration_rows = read_csv(paths["configurations"])
    configs = {row["configuration_id"]: row for row in configuration_rows}
    require(len(configs) == len(configuration_rows) == 6, "Expected six unique BLIP Pet configurations")
    require({(r["model_id"], int(r["resolution"])) for r in configuration_rows}
            == {(model, resolution) for model in MODELS for resolution in (336, 384, 448)},
            "Unexpected configuration inventory")
    registry = {row["id"]: row for row in json.loads(paths["registry"].read_text())["models"]}
    canonical = read_csv(paths["canonical"])
    require(len(canonical) == 54, "Expected locked 54-row canonical summary")
    grouped = defaultdict(list)
    for row in read_csv(paths["outcomes"]):
        require(row["configuration_id"] in configs, "Unknown outcome configuration")
        config = configs[row["configuration_id"]]
        for field in ("schema_version", "run_id", "environment_id", "dataset"):
            require(row[field] == config[field], f"Outcome/index mismatch in {field}")
        require(row["correct_top1"] in ("0", "1") and row["correct_top5"] in ("0", "1"), "Nonbinary outcome")
        grouped[row["configuration_id"]].append(row)
    for cid, rows in grouped.items():
        require(len(rows) == EXPECTED_N, f"Wrong sample count for {cid}")
        require(len({r["sample_id"] for r in rows}) == EXPECTED_N, f"Duplicate samples for {cid}")
    require(set(grouped) == set(configs), "Missing outcomes")

    results = []
    checks = {}
    for model in MODELS:
        indexed = {int(c["resolution"]): c for c in configuration_rows if c["model_id"] == model}
        c336, c384 = indexed[336], indexed[384]
        # Except resolution/configuration identity, the entire recorded protocol
        # must match. This includes checkpoint, manifest, prompt, scoring and run.
        for field in c336:
            if field not in {"configuration_id", "resolution"}:
                require(c336[field] == c384[field], f"Incomparable {model} configurations: {field}")
        require(c336["checkpoint_id"] == registry[model]["hub_id"], "Checkpoint registry mismatch")
        expected_fields = {
            "task": "zeroshot_classification", "dataset": "oxford_iiit_pet",
            "adapter": "blip_itc", "image_batch_size": "32", "text_batch_size": "0",
            "precision": "fp16", "scoring_path": "blip_itc_dual_embedding",
        }
        for field, expected in expected_fields.items():
            require(c336[field] == expected, f"Unexpected analysis population/protocol: {field}")

        rows336 = grouped[c336["configuration_id"]]
        rows384 = grouped[c384["configuration_id"]]
        ids, a, b = paired_correctness(rows336, rows384)
        n = len(ids)
        accuracy = {336: float(a.mean()), 384: float(b.mean())}
        for resolution, config in ((336, c336), (384, c384)):
            matching = [r for r in canonical if r["model_id"] == model
                        and r["task"] == c336["task"] and r["dataset"] == c336["dataset"]
                        and int(r["resolution"]) == resolution]
            require(len(matching) == 1, "Canonical row is missing or ambiguous")
            saved = matching[0]
            require(saved["configuration_id"] == config["configuration_id"], "Canonical configuration mismatch")
            require(int(saved["sample_count"]) == n, "Canonical count mismatch")
            for metric in ("performance_value", "top1_accuracy"):
                require(math.isclose(accuracy[resolution], float(saved[metric]), rel_tol=0, abs_tol=1e-12), "Canonical accuracy mismatch")

        difference = a - b
        sums = bootstrap_sums(difference)
        lower, upper = interval_pp(sums, n)
        only336 = int(((a == 1) & (b == 0)).sum())
        only384 = int(((a == 0) & (b == 1)).sum())
        both = int(((a == 1) & (b == 1)).sum())
        neither = int(((a == 0) & (b == 0)).sum())
        require(only336 + only384 + both + neither == n, "Paired contingency count mismatch")
        require(int(difference.sum()) == only336 - only384, "Signed discordance identity failed")
        results.append({
            "analysis_status": STATUS,
            "model_id": model,
            "checkpoint_id": c336["checkpoint_id"],
            "dataset": c336["dataset"],
            "paired_image_count": n,
            "accuracy_336_percent": 100 * accuracy[336],
            "accuracy_384_percent": 100 * accuracy[384],
            "difference_336_minus_384_pp": 100.0 * int(difference.sum()) / n,
            "ci_lower_95_pp": float(lower),
            "ci_upper_95_pp": float(upper),
            "correct_only_336": only336,
            "correct_only_384": only384,
            "correct_both": both,
            "incorrect_both": neither,
            "bootstrap_seed": SEED,
            "bootstrap_repetitions": REPLICATES,
            "ci_method": "paired_image_percentile_bootstrap",
        })
        checks[model] = {
            "canonical_accuracy_reproduction": "passed (absolute tolerance 1e-12)",
            "full_recorded_protocol_comparability": "passed",
            "paired_contingency_and_signed_difference_identity": "passed",
            **verify_pairing(rows336, rows384, a, b, sums),
        }

    payload = {
        "schema_version": "dicta-camera-ready-blip-pairing-v1",
        "analysis_date": "2026-09-15",
        "analysis_status": STATUS,
        "estimand": "100 * (Pet top-1 accuracy at 336 - Pet top-1 accuracy at 384), separately for each fixed BLIP checkpoint",
        "units": "percentage points",
        "resampling_unit": "matched Pet test image, retaining correctness at both resolutions",
        "sampling": "n images with replacement per replicate; direct empirical paired bootstrap",
        "rng": "NumPy Generator(PCG64(20260915)), reset independently for each model",
        "percentile_method": "numpy.quantile(method='linear'), quantiles 0.025 and 0.975",
        "coverage": "marginal, pointwise 95% intervals; no selection or multiple-comparison adjustment",
        "limitations": [
            "Exploratory post-review comparison selected after observing the resolution sweeps; not a preregistered confirmatory test.",
            "Intervals describe test-image sampling uncertainty conditional on these checkpoints, dataset and scoring protocol; they do not capture training, checkpoint, hardware or dataset-shift uncertainty.",
            "Bootstrap treats test-image pairs as exchangeable; no class or source-cluster resampling is possible from the compact fields alone.",
            "An interval containing zero does not establish equivalence or absence of a difference.",
            "This direct 336-minus-384 estimand differs from the original 384-minus-mean(336,448) anchor contrast.",
            "No causal explanation of the BLIP Pet resolution ordering is identified by this comparison.",
        ],
        "input_provenance": provenance,
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "verification": checks,
        "results": results,
    }
    output = ROOT / "analysis/processed/camera_ready"
    output.mkdir(parents=True, exist_ok=True)
    (output / "blip_pet_336_vs_384.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "blip_pet_336_vs_384.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    table = [
        "| Checkpoint | 336 accuracy (%) | 384 accuracy (%) | 336 − 384 (pp) | Paired 95% CI (pp) | Correct only at 336 / only at 384 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        table.append(f"| {row['model_id']} | {row['accuracy_336_percent']:.2f} | {row['accuracy_384_percent']:.2f} | {row['difference_336_minus_384_pp']:+.2f} | [{row['ci_lower_95_pp']:+.2f}, {row['ci_upper_95_pp']:+.2f}] | {row['correct_only_336']} / {row['correct_only_384']} |")
    note = "\n".join([
        "# Exploratory BLIP Pet paired comparison",
        "",
        "",
        "This post-review analysis compares 336 with 384 using saved per-image correctness. It was not preregistered. It does not replace or reinterpret the original 384-minus-mean(336,448) contrast.",
        "",
        "For each BLIP checkpoint, the estimand is 100 × (accuracy at 336 − accuracy at 384), in percentage points. Positive values favour 336. All 3,669 official Pet test-image IDs match exactly between resolutions, with identical recorded checkpoint, manifest, prompt, batch, precision, scoring path and run. There are no duplicate or missing image IDs. The four accuracies reproduce the unchanged canonical summary to absolute tolerance 1e-12.",
        "",
        *table,
        "",
        "The ordinary paired bootstrap resamples 3,669 image pairs with replacement 10,000 times. The same sampled image index selects correctness at both resolutions. A fresh NumPy PCG64 generator seeded with 20260915 is used for each model. Interval endpoints are the 2.5th and 97.5th percentiles with linear interpolation. The script uses direct image-index resampling, not an asymptotic or accelerated approximation. No p-values were calculated.",
        "",
        "The point estimates favour 336. The base interval includes zero; the large interval lies above zero in this exploratory pointwise analysis, providing evidence of higher 336 accuracy for the large checkpoint under the saved Pet protocol. The base result does not demonstrate equivalence. These intervals quantify image-sampling uncertainty conditional on the saved protocol, treating image pairs as exchangeable. They are pointwise intervals without adjustment for the post-review choice of comparison or for multiple comparisons. They omit training/checkpoint variability, dataset shifts and within-class/source clustering. The outputs identify no causal mechanism or general resolution optimum.",
        "",
        "Verification passed: fixed input SHA-256 locks; configuration/registry consistency; sample-ID uniqueness and equality; canonical accuracy reproduction; contingency-count/sign identities; exact output invariance when the two input files are reordered independently; fixed-seed bootstrap replay; sign reversal when resolution order is swapped; rejection of missing or duplicate image IDs. No model was executed and no existing results were modified.",
        "",
        "",
        "Reproduce from the camera-ready copy root:",
        "",
        "```sh",
        "python3 analysis/camera_ready_blip_pairing.py",
        "```",
        "",
        f"Runtime: Python {platform.python_version()}, NumPy {np.__version__}. The JSON contains exact results, input paths/hashes, script hash, methods, checks and limitations. The CSV contains the two comparison rows. The script writes only these new analysis artifacts under `analysis/processed/camera_ready/`.",
        "",
    ])
    (output / "blip_pet_336_vs_384.md").write_text(note, encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "results": results, "verification": "passed"}, indent=2))


if __name__ == "__main__":
    main()
