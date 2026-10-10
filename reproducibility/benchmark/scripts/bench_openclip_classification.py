#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from charlie_vlm.csvio import append_rows
from charlie_vlm.datasets import ClassificationDataset
from charlie_vlm.manifests import class_names_by_id, load_classification_manifest
from charlie_vlm.measure import (
    device_name,
    inference_context,
    latency_summary,
    peak_memory_gb,
    peak_memory_reserved_gb,
    reset_peak_memory,
    resolve_device,
    timed_call,
)
from charlie_vlm.outcomes import (
    classification_outcome,
    configuration_id,
    configuration_record,
    file_sha256,
)
from charlie_vlm.openclip_utils import (
    finite_or_blank,
    load_model,
    model_metadata,
    normalize,
    require_openclip,
)
from charlie_vlm.registry import load_registry, select_models


DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot OpenCLIP classification benchmark for resolution and scale sweeps."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "classification")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models", nargs="+", default=["clip_vit_b32_openai", "clip_vit_l14_openai"])
    parser.add_argument("--resolutions", nargs="+", type=int)
    parser.add_argument("--prompt-template", default="a photo of a {class_name}.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-compact-outcomes", action="store_true")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--environment-id", default="unrecorded")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def format_prompt(template: str, class_name: str) -> str:
    try:
        return template.format(class_name=class_name)
    except KeyError as exc:
        raise ValueError("Prompt template must use the {class_name} placeholder") from exc


def encode_text_features(torch, model, tokenizer, class_names: list[str], template: str, device: str, precision: str):
    prompts = [format_prompt(template, class_name) for class_name in class_names]
    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            return normalize(model.encode_text(tokens))


def forward_classification(torch, model, images, text_features, device: str, precision: str):
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            image_features = normalize(model.encode_image(images))
            scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 100.0
            return scale * image_features @ text_features.T


def benchmark_one(torch, open_clip, spec, resolution: int, args, rows) -> tuple[dict, list[dict], list[dict]]:
    model, tokenizer, preprocess = load_model(open_clip, spec, resolution, args.device)
    metadata = model_metadata(model, spec, resolution)

    names = class_names_by_id(rows)
    class_ids = sorted(names)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    class_names = [names[class_id] for class_id in class_ids]
    dataset = ClassificationDataset(rows, preprocess, class_to_index)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    text_features = encode_text_features(
        torch, model, tokenizer, class_names, args.prompt_template, args.device, args.precision
    )
    correct_top1 = 0
    correct_top5 = 0
    sample_count = 0
    timed_examples = 0
    latencies = []
    predictions = []
    compact_outcomes = []
    scoring_path = (
        "sigmoid_dual_embedding"
        if spec.model_id.startswith("siglip")
        else "contrastive_dual_embedding"
    )
    config_id = configuration_id(
        model_id=spec.model_id,
        task="zeroshot_classification",
        dataset=args.dataset_name,
        resolution=resolution,
        image_batch_size=args.batch_size,
        text_batch_size=0,
        precision=args.precision,
        scoring_path=scoring_path,
        checkpoint_id=f"{spec.model_name}:{spec.pretrained}",
        protocol_variant=f"prompt={args.prompt_template}",
    )
    reset_peak_memory(torch, args.device)

    for batch_index, (images, targets, row_indexes) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        def run_batch():
            return forward_classification(
                torch, model, images, text_features, args.device, args.precision
            )

        logits = run_batch()
        if batch_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                logits, latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += targets.shape[0]

        top_k = min(5, logits.shape[-1])
        probabilities = logits.softmax(dim=-1)
        confidences, top1 = probabilities.max(dim=-1)
        top5 = logits.topk(top_k, dim=-1).indices
        correct_top1 += (top1 == targets).sum().item()
        correct_top5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        sample_count += targets.shape[0]

        if args.save_predictions:
            for offset, row_index in enumerate(row_indexes.tolist()):
                source_row = rows[row_index]
                predictions.append(
                    {
                        **metadata,
                        "dataset": args.dataset_name,
                        "image_path": str(source_row.image_path),
                        "true_class_id": source_row.class_id,
                        "true_class_name": source_row.class_name,
                        "predicted_class_id": class_ids[top1[offset].item()],
                        "predicted_class_name": class_names[top1[offset].item()],
                        "correct_top1": int(top1[offset].item() == targets[offset].item()),
                        "confidence": confidences[offset].item(),
                    }
                )

        if args.save_compact_outcomes:
            top5_correct = (top5 == targets.unsqueeze(1)).any(dim=1)
            for offset, row_index in enumerate(row_indexes.tolist()):
                source_row = rows[row_index]
                compact_outcomes.append(
                    classification_outcome(
                        config_id=config_id,
                        run_id=args.run_id,
                        environment_id=args.environment_id,
                        dataset=args.dataset_name,
                        source_sample_id=source_row.sample_id,
                        correct_top1=top1[offset].item() == targets[offset].item(),
                        correct_top5=bool(top5_correct[offset].item()),
                    )
                )

    latency = latency_summary(latencies)
    timed_seconds = sum(latencies) / 1000.0
    summary = {
        **metadata,
        "task": "zeroshot_classification",
        "dataset": args.dataset_name,
        "performance_metric": "top1_accuracy",
        "performance_value": correct_top1 / sample_count,
        "top1_accuracy": correct_top1 / sample_count,
        "top5_accuracy": correct_top5 / sample_count,
        "sample_count": sample_count,
        "class_count": len(class_ids),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "timed_repeats": args.timed_repeats,
        "throughput_examples_s": timed_examples / timed_seconds if timed_seconds else "",
        "peak_vram_gb": peak_memory_gb(torch, args.device),
        "peak_reserved_vram_gb": peak_memory_reserved_gb(torch, args.device),
        "device": args.device,
        "gpu_name": device_name(torch, args.device),
        "precision": args.precision,
        "prompt_template": args.prompt_template,
        "manifest": str(args.manifest),
        "configuration_id": config_id,
        **{key: finite_or_blank(value) for key, value in latency.items()},
    }
    return summary, predictions, compact_outcomes


def failure_row(spec, resolution: int, args, error: Exception) -> dict:
    return {
        "model_id": spec.model_id,
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "resolution": resolution,
        "dataset": args.dataset_name,
        "task": "zeroshot_classification",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main() -> None:
    args = parse_args()
    if args.timed_repeats < 1:
        raise SystemExit("--timed-repeats must be at least 1")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install PyTorch first: python3 -m pip install -r requirements.txt") from exc

    open_clip = require_openclip()
    registry, default_resolutions = load_registry(args.registry)
    specs = select_models(registry, args.models)
    resolutions = args.resolutions or default_resolutions
    args.device = resolve_device(torch, args.device)
    rows = load_classification_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]

    for spec in specs:
        for resolution in resolutions:
            print(f"Benchmarking {spec.model_id} at {resolution} on {args.dataset_name}")
            try:
                summary, predictions, compact_outcomes = benchmark_one(
                    torch, open_clip, spec, resolution, args, rows
                )
            except Exception as error:
                append_rows(args.output_dir / "failures.csv", [failure_row(spec, resolution, args, error)])
                if args.fail_fast:
                    raise
                print(f"Skipped {spec.model_id} at {resolution}: {error}")
                continue
            append_rows(args.output_dir / "benchmark_rows.csv", [summary])
            append_rows(
                args.output_dir / "configuration_index.csv",
                [configuration_record(
                    config_id=summary["configuration_id"],
                    run_id=args.run_id,
                    environment_id=args.environment_id,
                    adapter="openclip",
                    model_id=summary["model_id"],
                    checkpoint_id=f"{summary['model_name']}:{summary['pretrained']}",
                    task=summary["task"],
                    dataset=args.dataset_name,
                    manifest_sha256=file_sha256(args.manifest),
                    resolution=resolution,
                    image_batch_size=args.batch_size,
                    text_batch_size=0,
                    precision=args.precision,
                    scoring_path=("sigmoid_dual_embedding" if summary["model_id"].startswith("siglip") else "contrastive_dual_embedding"),
                    protocol_variant=f"prompt={args.prompt_template}",
                    warmup_count=args.warmup_batches,
                    timed_repeat_count=args.timed_repeats,
                )],
            )
            if predictions:
                append_rows(args.output_dir / "predictions.csv", predictions)
            if compact_outcomes:
                append_rows(args.output_dir / "compact_outcomes.csv", compact_outcomes)


if __name__ == "__main__":
    main()
