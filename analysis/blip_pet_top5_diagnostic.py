#!/usr/bin/env python3
"""Replay the final paper's exploratory BLIP Pet top-one/top-five diagnostic.

Uses only retained correctness outcomes; no dependencies, GPU, or downloads.
The diagnostic describes ranking changes and does not identify their cause.
Run from any working directory. Outputs go to ignored analysis/processed/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/processed/camera_ready/blip_pet_top5_diagnostic.json"
INPUTS = {
    "outcomes": ("results/supporting/focused_run/classification/compact_outcomes.csv", "d3ead7ffa981a1d2f61d03a459444a34878f4c7b725d4e2dbd7a77150a78ca75"),
    "configurations": ("results/supporting/focused_run/classification/configuration_index.csv", "535f24f787fc1ba787d1bf7c9e77703c431ea61b756c7e67a8a86f7e31d40cc6"),
    "canonical": ("results/canonical/dicta_mock_review_54_with_ci.csv", "4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7"),
}
MODELS = ("blip_itm_base_coco", "blip_itm_large_coco")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_outcomes(rows):
    """Validate binary correctness and join by image ID, not file row order."""
    indexed = {}
    for row in rows:
        sample = row["sample_id"]
        require(sample and sample not in indexed, "Empty or duplicate sample ID")
        require(all(row[f"correct_top{k}"] in ("0", "1") for k in (1, 5)), "Nonbinary correctness")
        bits = tuple(int(row[f"correct_top{k}"]) for k in (1, 5))
        require(bits[0] <= bits[1], "Top-one correctness must imply top-five correctness")
        indexed[sample] = bits
    return indexed


def summarize_pair(rows336, rows384):
    a, b = index_outcomes(rows336), index_outcomes(rows384)
    require(a and a.keys() == b.keys(), "Empty or mismatched paired image IDs")
    result = {"paired_images": len(a), "metrics": {}}
    for index, k in enumerate((1, 5)):
        gains = [sid for sid in a if a[sid][index] > b[sid][index]]
        losses = [sid for sid in a if a[sid][index] < b[sid][index]]
        result["metrics"][f"top{k}"] = {
            "correct_336": sum(v[index] for v in a.values()),
            "correct_384": sum(v[index] for v in b.values()),
            "gains_at_336": len(gains),
            "losses_at_336": len(losses),
            "difference_percentage_points": 100 * (len(gains) - len(losses)) / len(a),
        }
        if k == 1:
            result["top1_changes_true_label_in_top5_at_both"] = sum(a[sid][1] == b[sid][1] == 1 for sid in gains + losses)
            result["top1_changes_total"] = len(gains) + len(losses)
    return result


def main():
    paths, hashes = {}, {}
    for name, (relative, expected_hash) in INPUTS.items():
        paths[name] = ROOT / relative
        hashes[name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        require(hashes[name] == expected_hash, f"Input checksum mismatch: {relative}")

    config_rows = read_rows(paths["configurations"])
    configs = {r["configuration_id"]: r for r in config_rows}
    require(len(configs) == len(config_rows) == 6, "Expected six unique configurations")
    require({(r["model_id"], int(r["resolution"])) for r in config_rows}
            == {(m, res) for m in MODELS for res in (336, 384, 448)}, "Unexpected configurations")
    canonical_rows = [r for r in read_rows(paths["canonical"])
                      if r["task"] == "zeroshot_classification" and r["model_id"] in MODELS
                      and int(r["resolution"]) in (336, 384, 448)]
    canonical = {(r["model_id"], int(r["resolution"])): r for r in canonical_rows}
    require(len(canonical) == len(canonical_rows) == 6, "Expected six unique canonical BLIP Pet rows")
    grouped = defaultdict(list)
    for row in read_rows(paths["outcomes"]):
        require(row["configuration_id"] in configs, "Unknown outcome configuration")
        c = configs[row["configuration_id"]]
        require(all(row[field] == c[field] for field in
                    ("schema_version", "run_id", "environment_id", "dataset")), "Outcome/index mismatch")
        grouped[(c["model_id"], int(c["resolution"]))].append(row)
    require(grouped.keys() == canonical.keys(), "Incomplete configuration outcomes")
    for key, rows in grouped.items():
        values = index_outcomes(rows)
        require(len(values) == 3669, "Expected 3,669 images per configuration")
        for index, k in enumerate((1, 5)):
            require(math.isclose(sum(v[index] for v in values.values()) / len(values),
                                float(canonical[key][f"top{k}_accuracy"]), rel_tol=0, abs_tol=1e-12),
                    f"Canonical top-{k} accuracy mismatch for {key}")

    results = []
    for model in MODELS:
        pair = [next(c for c in config_rows if c["model_id"] == model and int(c["resolution"]) == res)
                for res in (336, 384)]
        for field in pair[0]:
            if field not in ("configuration_id", "resolution"):
                require(pair[0][field] == pair[1][field], f"Paired protocols differ: {field}")
        results.append({"model_id": model, **summarize_pair(grouped[(model, 336)], grouped[(model, 384)])})

    payload = {
        "status": "Exploratory post-review diagnostic; descriptive counts, not a causal test",
        "input_sha256": hashes,
        "checks": "Unique matched IDs; same recorded protocol except resolution; top-1/top-5 logical consistency; exact canonical accuracy reproduction",
        "results": results,
        "interpretation": "For every image whose top-1 correctness changes, the true breed remains in the top five at both resolutions. These cases change between rank one and a non-winning top-five position. This does not identify which wrong breed won, score margins, or the cause of the ranking change.",
        "limitations": [
            "The analysis was chosen after seeing the main results and is descriptive.",
            "Top-five aggregate stability does not establish statistical equivalence.",
            "Retained outcomes do not contain logits, competing labels, or image attributes.",
            "Input resampling, token count and positional-grid interpolation vary jointly; saved outcomes cannot isolate their contributions.",
            "Source inspection describes the retained adapter and documented Transformers 4.44.2 behavior; historical model-hub revisions were not pinned.",
        ],
        "implementation_sources": [
            "https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/blip/modeling_blip.py#L208-L250",
            "https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/blip/image_processing_blip.py",
            "https://github.com/salesforce/BLIP/blob/main/configs/retrieval_coco.yaml",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
