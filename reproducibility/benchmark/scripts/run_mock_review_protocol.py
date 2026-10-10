#!/usr/bin/env python3
"""Validate, dry-run, or execute the focused DICTA mock-review protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "mock_review_protocol.json"
VALID_PRECISIONS = {"fp16", "bf16", "fp32"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pet-manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "pet_test.csv")
    parser.add_argument("--flickr-manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "flickr30k_test1k.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "mock_review")
    parser.add_argument("--run-id", default="mock-review-dry-run")
    parser.add_argument("--environment-id", default="runpod-rtx6000ada48-unassigned")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--stage", choices=("all", "outcomes", "batch"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load protocol {path}: {exc}") from exc


def validate_protocol(protocol: dict) -> dict[str, int]:
    expected = protocol["expected_cells"]
    counts = {"pet_blip_validation": 0, "flickr_grouped_outcomes": 0, "batch_sensitivity": 0}
    seen = {"classification": set(), "retrieval": set(), "batch": set()}

    for task, output_key in (("classification", "pet_blip_validation"), ("retrieval", "flickr_grouped_outcomes")):
        for group in protocol[task]:
            if group["precision"] not in VALID_PRECISIONS:
                raise ValueError(f"invalid {task} precision")
            for field in ("image_batch_size", "warmup_batches", "timed_repeats"):
                if int(group[field]) <= 0:
                    raise ValueError(f"invalid {task} {field}")
            if task == "retrieval" and int(group["text_batch_size"]) <= 0:
                raise ValueError("invalid retrieval text_batch_size")
            for model in group["models"]:
                for resolution in group["resolutions"]:
                    key = (model, int(resolution))
                    if key in seen[task]:
                        raise ValueError(f"duplicate {task} configuration {key}")
                    seen[task].add(key)
                    counts[output_key] += 1

    batch = protocol["batch_sensitivity"]
    if batch["batch_sizes"] != [1, 32]:
        raise ValueError("batch sizes must remain [1, 32]")
    if (batch["blocks"], batch["warmup_iterations_per_block"], batch["timed_iterations_per_block"]) != (5, 10, 50):
        raise ValueError("batch block settings changed")
    for pair in batch["pairs"]:
        for resolution in pair["resolutions"]:
            for batch_size in batch["batch_sizes"]:
                key = (pair["model"], int(resolution), int(batch_size))
                if key in seen["batch"]:
                    raise ValueError(f"duplicate batch configuration {key}")
                seen["batch"].add(key)
                counts["batch_sensitivity"] += 1

    required = {key: int(expected[key]) for key in counts}
    if counts != required:
        raise ValueError(f"cell counts changed: expected {required}, got {counts}")
    required_blip = {
        (model, resolution)
        for model in ("blip_itm_base_coco", "blip_itm_large_coco")
        for resolution in (336, 384, 448)
    }
    if seen["classification"] != required_blip:
        raise ValueError("Pet validation must be BLIP base/large at 336/384/448")
    return counts


def command_for_group(task: str, group: dict, args: argparse.Namespace) -> list[str]:
    adapter = group["adapter"]
    if task == "classification" and adapter == "blip_itc":
        script = "bench_hf_blip_classification.py"
    elif task == "retrieval" and adapter == "openclip":
        script = "bench_openclip_retrieval.py"
    elif task == "retrieval" and adapter == "blip_itc":
        script = "bench_hf_blip_retrieval.py"
    else:
        raise ValueError(f"unsupported {task}/{adapter}")
    manifest = args.pet_manifest if task == "classification" else args.flickr_manifest
    output = args.output_root / task
    command = [
        args.python_executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--manifest", str(manifest),
        "--dataset-name", "oxford_iiit_pet" if task == "classification" else "flickr30k_test1k",
        "--output-dir", str(output),
        "--models", *group["models"],
        "--resolutions", *(str(value) for value in group["resolutions"]),
    ]
    if task == "classification":
        command.extend(["--batch-size", str(group["image_batch_size"]), "--save-compact-outcomes"])
        command.extend(["--itm-rerank-topk", str(group.get("itm_rerank_topk", 0))])
    else:
        command.extend([
            "--image-batch-size", str(group["image_batch_size"]),
            "--text-batch-size", str(group["text_batch_size"]),
            "--max-images", "1000",
            "--save-grouped-outcomes",
        ])
    command.extend([
        "--precision", group["precision"],
        "--warmup-batches", str(group["warmup_batches"]),
        "--timed-repeats", str(group["timed_repeats"]),
        "--run-id", args.run_id,
        "--environment-id", args.environment_id,
        "--fail-fast",
    ])
    return command


def build_commands(protocol: dict, args: argparse.Namespace) -> list[list[str]]:
    commands = []
    if args.stage in {"all", "outcomes"}:
        for task in ("classification", "retrieval"):
            commands.extend(command_for_group(task, group, args) for group in protocol[task])
    if args.stage in {"all", "batch"}:
        commands.append([
            args.python_executable,
            str(PROJECT_ROOT / "scripts" / "bench_batch_sensitivity.py"),
            "--protocol", str(args.protocol),
            "--output-dir", str(args.output_root / "batch_sensitivity"),
            "--run-id", args.run_id,
            "--environment-id", args.environment_id,
            "--fail-fast",
        ])
    return commands


def protected_outputs(args: argparse.Namespace) -> list[Path]:
    names = []
    if args.stage in {"all", "outcomes"}:
        for task in ("classification", "retrieval"):
            for filename in ("benchmark_rows.csv", "configuration_index.csv", "compact_outcomes.csv", "grouped_outcomes.csv", "failures.csv"):
                names.append(args.output_root / task / filename)
    if args.stage in {"all", "batch"}:
        names.extend([
            args.output_root / "batch_sensitivity" / "raw_batch_timings.csv",
            args.output_root / "batch_sensitivity" / "failures.csv",
        ])
    return names


def sanitize_command(command: list[str], args: argparse.Namespace) -> list[str]:
    replacements = {
        str(PROJECT_ROOT): "{BENCHMARK_ROOT}",
        str(args.pet_manifest): "{PET_MANIFEST}",
        str(args.flickr_manifest): "{FLICKR_MANIFEST}",
        str(args.output_root): "{OUTPUT_ROOT}",
    }
    sanitized = []
    for value in command:
        clean = value
        for prefix, replacement in replacements.items():
            clean = clean.replace(prefix, replacement)
        sanitized.append(clean)
    return sanitized


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def write_manifest(protocol: dict, commands: list[list[str]], args: argparse.Namespace) -> None:
    record = {
        "schema_version": "dicta-mock-review-run-v1",
        "run_id": args.run_id,
        "environment_id": args.environment_id,
        "git_commit": git_commit(),
        "protocol_sha256": file_sha256(args.protocol),
        "model_registry_sha256": file_sha256(PROJECT_ROOT / "configs" / "models.json"),
        "hf_registry_sha256": file_sha256(PROJECT_ROOT / "configs" / "hf_models.json"),
        "pet_manifest_sha256": file_sha256(args.pet_manifest) if args.pet_manifest.is_file() else "missing",
        "flickr_manifest_sha256": file_sha256(args.flickr_manifest) if args.flickr_manifest.is_file() else "missing",
        "commands": [sanitize_command(command, args) for command in commands],
        "note": "Path placeholders are intentional; no credentials, pod identifiers, or local account paths are retained.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    try:
        counts = validate_protocol(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Mock-review protocol validation failed: {exc}") from exc
    commands = build_commands(protocol, args)
    print(
        "Validated "
        f"{counts['pet_blip_validation']} Pet, "
        f"{counts['flickr_grouped_outcomes']} Flickr, and "
        f"{counts['batch_sensitivity']} batch-sensitivity cells."
    )
    if args.dry_run:
        for command in commands:
            print(shlex.join(sanitize_command(command, args)))
        return
    required_manifests = []
    if args.stage in {"all", "outcomes"}:
        required_manifests = [args.pet_manifest, args.flickr_manifest]
    missing = [path for path in required_manifests if not path.is_file()]
    if missing:
        raise SystemExit("Missing required manifest(s): " + ", ".join(str(path) for path in missing))
    existing = [path for path in protected_outputs(args) if path.exists()]
    if existing:
        raise SystemExit("Refusing to append to existing outputs: " + ", ".join(str(path) for path in existing))
    write_manifest(protocol, commands, args)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
