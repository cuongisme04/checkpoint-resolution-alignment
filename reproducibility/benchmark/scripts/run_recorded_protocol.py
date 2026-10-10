#!/usr/bin/env python3
"""Run the reconstructed 50-cell benchmark protocol.

The command groups in ``configs/recorded_protocol.json`` were reconstructed
from the locked canonical result rows. They describe the settings that produced
the published matrix, but are not presented as an original shell transcript.
Use ``--dry-run`` to validate and print every command without loading models or
datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "recorded_protocol.json"
VALID_PRECISIONS = {"fp16", "bf16", "fp32"}
TASK_KEYS = ("classification", "retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the recorded 28-classification/22-retrieval matrix."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--pet-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests" / "pet_test.csv",
    )
    parser.add_argument(
        "--flickr-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests" / "flickr30k_test1k.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "recorded_50_cell",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol and print commands without executing them.",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Protocol file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid protocol JSON in {path}: {exc}") from exc


def group_cells(group: dict) -> int:
    return len(group.get("models", [])) * len(group.get("resolutions", []))


def validate_protocol(protocol: dict) -> dict[str, int]:
    expected = protocol.get("expected_cells", {})
    expected_by_task = {
        "classification": int(expected.get("zeroshot_classification", -1)),
        "retrieval": int(expected.get("image_text_retrieval", -1)),
    }
    seen = {task: set() for task in TASK_KEYS}
    counts = {}

    for task in TASK_KEYS:
        groups = protocol.get(task)
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"protocol field {task!r} must be a non-empty list")
        for group_number, group in enumerate(groups, start=1):
            models = group.get("models", [])
            resolutions = group.get("resolutions", [])
            if not models or not resolutions:
                raise ValueError(f"{task} group {group_number} has no models or resolutions")
            if group.get("precision") not in VALID_PRECISIONS:
                raise ValueError(f"{task} group {group_number} has invalid precision")
            for field in ("image_batch_size", "warmup_batches", "timed_repeats"):
                if not isinstance(group.get(field), int) or group[field] <= 0:
                    raise ValueError(f"{task} group {group_number} has invalid {field}")
            if task == "retrieval":
                if not isinstance(group.get("text_batch_size"), int) or group["text_batch_size"] <= 0:
                    raise ValueError(f"retrieval group {group_number} has invalid text_batch_size")
            for model in models:
                for resolution in resolutions:
                    if not isinstance(resolution, int) or resolution <= 0:
                        raise ValueError(f"{task} group {group_number} has invalid resolution")
                    key = (model, resolution)
                    if key in seen[task]:
                        raise ValueError(f"duplicate {task} configuration: {key}")
                    seen[task].add(key)
        counts[task] = sum(group_cells(group) for group in groups)
        if counts[task] != expected_by_task[task]:
            raise ValueError(
                f"{task} has {counts[task]} cells; expected {expected_by_task[task]}"
            )

    paligemma = {
        (group["models"][0], group["resolutions"][0]): group
        for group in protocol["classification"]
        if group.get("adapter") == "paligemma_loglik"
        and len(group.get("models", [])) == 1
        and len(group.get("resolutions", [])) == 1
    }
    required_paligemma = {
        ("paligemma_3b_mix_224", 224): (1, "bf16", 2, 1, 16),
        ("paligemma_3b_mix_448", 448): (1, "bf16", 5, 1, 4),
    }
    for key, required_settings in required_paligemma.items():
        group = paligemma.get(key)
        if group is None:
            raise ValueError(f"missing recorded PaliGemma cell: {key}")
        actual = (
            group["image_batch_size"],
            group["precision"],
            group["warmup_batches"],
            group["timed_repeats"],
            group["prompt_chunk_size"],
        )
        if actual != required_settings:
            raise ValueError(f"PaliGemma settings changed for {key}: {actual}")

    return counts


def command_for_group(
    task: str,
    group: dict,
    python_executable: str,
    manifest: Path,
    output_dir: Path,
) -> list[str]:
    adapter = group["adapter"]
    if task == "classification" and adapter == "openclip":
        script = "bench_openclip_classification.py"
    elif task == "classification" and adapter == "blip_itc":
        script = "bench_hf_blip_classification.py"
    elif task == "classification" and adapter == "paligemma_loglik":
        script = "bench_hf_paligemma_classification.py"
    elif task == "retrieval" and adapter == "openclip":
        script = "bench_openclip_retrieval.py"
    elif task == "retrieval" and adapter == "blip_itc":
        script = "bench_hf_blip_retrieval.py"
    else:
        raise ValueError(f"unsupported task/adapter combination: {task}/{adapter}")

    command = [
        python_executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--manifest",
        str(manifest),
        "--dataset-name",
        "oxford_iiit_pet" if task == "classification" else "flickr30k_test1k",
        "--output-dir",
        str(output_dir),
        "--models",
        *group["models"],
        "--resolutions",
        *(str(value) for value in group["resolutions"]),
    ]

    if task == "classification":
        command.extend(["--batch-size", str(group["image_batch_size"])])
    else:
        command.extend(
            [
                "--image-batch-size",
                str(group["image_batch_size"]),
                "--text-batch-size",
                str(group["text_batch_size"]),
                "--max-images",
                "1000",
            ]
        )

    command.extend(
        [
            "--precision",
            group["precision"],
            "--warmup-batches",
            str(group["warmup_batches"]),
            "--timed-repeats",
            str(group["timed_repeats"]),
        ]
    )

    if task == "classification":
        command.append("--save-predictions")
    if adapter == "blip_itc" and task == "classification":
        command.extend(["--itm-rerank-topk", str(group.get("itm_rerank_topk", 0))])
    if adapter == "paligemma_loglik":
        command.extend(
            [
                "--primary-protocol",
                "loglik",
                "--score-normalization",
                group["score_normalization"],
                "--prompt-chunk-size",
                str(group["prompt_chunk_size"]),
            ]
        )
    command.append("--fail-fast")
    return command


def build_commands(protocol: dict, args: argparse.Namespace) -> list[tuple[list[str], dict]]:
    commands = []
    task_settings = {
        "classification": (args.pet_manifest, args.output_root / "classification"),
        "retrieval": (args.flickr_manifest, args.output_root / "retrieval"),
    }
    for task in TASK_KEYS:
        manifest, output_dir = task_settings[task]
        for group in protocol[task]:
            command = command_for_group(
                task, group, args.python_executable, manifest, output_dir
            )
            env = {}
            if group.get("expandable_cuda_segments"):
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            commands.append((command, env))
    return commands


def refuse_existing_outputs(output_root: Path) -> None:
    protected = (
        output_root / "classification" / "benchmark_rows.csv",
        output_root / "classification" / "predictions.csv",
        output_root / "classification" / "failures.csv",
        output_root / "retrieval" / "benchmark_rows.csv",
        output_root / "retrieval" / "failures.csv",
    )
    existing = [path for path in protected if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(
            "Refusing to append duplicate configurations to existing outputs: " + joined
        )


def write_run_manifest(
    output_root: Path, protocol_path: Path, protocol: dict, commands: list[tuple[list[str], dict]]
) -> None:
    protocol_bytes = protocol_path.read_bytes()
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "canonical_summary_sha256": protocol["canonical_summary_sha256"],
        "commands": [command for command, _ in commands],
        "environment_overrides": [env for _, env in commands],
        "note": "Generated locally; review for local paths before sharing.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    try:
        counts = validate_protocol(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Recorded protocol validation failed: {exc}") from exc
    commands = build_commands(protocol, args)
    print(
        f"Validated {counts['classification']} classification and "
        f"{counts['retrieval']} retrieval cells in {len(commands)} command groups."
    )

    if args.dry_run:
        for command, env_override in commands:
            prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_override.items())
            rendered = shlex.join(command)
            print(f"{prefix} {rendered}".strip())
        return

    for manifest in (args.pet_manifest, args.flickr_manifest):
        if not manifest.is_file():
            raise SystemExit(f"Required manifest not found: {manifest}")
    refuse_existing_outputs(args.output_root)
    write_run_manifest(args.output_root, args.protocol, protocol, commands)

    for index, (command, env_override) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        env = os.environ.copy()
        env.update(env_override)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
