#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from charlie_vlm.csvio import append_rows
from charlie_vlm.datasets import RetrievalImageDataset
from charlie_vlm.manifests import load_retrieval_manifest
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
    configuration_id,
    configuration_record,
    file_sha256,
    retrieval_metrics_and_outcomes,
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
        description="OpenCLIP image-text retrieval benchmark for architecture and resolution sweeps."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models", nargs="+", default=["clip_rn50_openai", "clip_vit_b32_openai"])
    parser.add_argument("--resolutions", nargs="+", type=int)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-grouped-outcomes", action="store_true")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--environment-id", default="unrecorded")
    return parser.parse_args()


def limit_image_rows(rows, max_images: int | None):
    if not max_images:
        return rows
    allowed = []
    seen = set()
    for row in rows:
        if row.image_id not in seen:
            if len(seen) >= max_images:
                continue
            seen.add(row.image_id)
        allowed.append(row)
    return allowed


def encode_image_batch(torch, model, images, device: str, precision: str):
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            return normalize(model.encode_image(images))


def encode_text_batch(torch, model, tokenizer, captions: list[str], device: str, precision: str):
    tokens = tokenizer(captions).to(device)
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            return normalize(model.encode_text(tokens))


def encode_images(torch, model, dataset, args):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    features = []
    latencies = []
    timed_examples = 0
    for batch_index, (images, _) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)

        def run_batch():
            return encode_image_batch(torch, model, images, args.device, args.precision)

        batch_features = run_batch()
        if batch_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                batch_features, latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += images.shape[0]
        features.append(batch_features.detach().cpu())
    return torch.cat(features), latencies, timed_examples


def encode_captions(torch, model, tokenizer, rows, args):
    captions = [row.caption for row in rows]
    features = []
    latencies = []
    timed_examples = 0
    for batch_index, start in enumerate(range(0, len(captions), args.text_batch_size)):
        batch_captions = captions[start : start + args.text_batch_size]

        def run_batch():
            return encode_text_batch(
                torch, model, tokenizer, batch_captions, args.device, args.precision
            )

        batch_features = run_batch()
        if batch_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                batch_features, latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += len(batch_captions)
        features.append(batch_features.detach().cpu())
    return torch.cat(features), latencies, timed_examples


def benchmark_one(torch, open_clip, spec, resolution: int, args, rows) -> tuple[dict, list[dict]]:
    model, tokenizer, preprocess = load_model(open_clip, spec, resolution, args.device)
    metadata = model_metadata(model, spec, resolution)
    image_dataset = RetrievalImageDataset(rows, preprocess)
    reset_peak_memory(torch, args.device)
    image_features, image_latencies, timed_images = encode_images(torch, model, image_dataset, args)
    text_features, text_latencies, timed_captions = encode_captions(torch, model, tokenizer, rows, args)
    scoring_path = (
        "sigmoid_dual_embedding"
        if spec.model_id.startswith("siglip")
        else "contrastive_dual_embedding"
    )
    config_id = configuration_id(
        model_id=spec.model_id,
        task="image_text_retrieval",
        dataset=args.dataset_name,
        resolution=resolution,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        precision=args.precision,
        scoring_path=scoring_path,
        checkpoint_id=f"{spec.model_name}:{spec.pretrained}",
        protocol_variant="full_similarity_matrix",
    )
    metrics, outcomes = retrieval_metrics_and_outcomes(
        image_features,
        text_features,
        image_dataset.image_ids,
        rows,
        dataset=args.dataset_name,
        config_id=config_id,
        run_id=args.run_id,
        environment_id=args.environment_id,
    )

    image_seconds = sum(image_latencies) / 1000.0
    text_seconds = sum(text_latencies) / 1000.0
    image_latency = latency_summary(image_latencies)
    text_latency = latency_summary(text_latencies)
    summary = {
        **metadata,
        "task": "image_text_retrieval",
        "dataset": args.dataset_name,
        "performance_metric": "mean_recall_at_1",
        "performance_value": metrics["mean_recall_at_1"],
        "image_count": len(image_dataset),
        "caption_count": len(rows),
        "batch_size": args.image_batch_size,
        "text_batch_size": args.text_batch_size,
        "warmup_batches": args.warmup_batches,
        "timed_repeats": args.timed_repeats,
        "throughput_examples_s": timed_images / image_seconds if image_seconds else "",
        "text_throughput_examples_s": timed_captions / text_seconds if text_seconds else "",
        "peak_vram_gb": peak_memory_gb(torch, args.device),
        "peak_reserved_vram_gb": peak_memory_reserved_gb(torch, args.device),
        "device": args.device,
        "gpu_name": device_name(torch, args.device),
        "precision": args.precision,
        "manifest": str(args.manifest),
        "configuration_id": config_id,
        **metrics,
        **{key: finite_or_blank(value) for key, value in image_latency.items()},
        **{f"text_{key}": finite_or_blank(value) for key, value in text_latency.items()},
    }
    return summary, outcomes if args.save_grouped_outcomes else []


def failure_row(spec, resolution: int, args, error: Exception) -> dict:
    return {
        "model_id": spec.model_id,
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "resolution": resolution,
        "dataset": args.dataset_name,
        "task": "image_text_retrieval",
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
    rows = limit_image_rows(load_retrieval_manifest(args.manifest), args.max_images)

    for spec in specs:
        for resolution in resolutions:
            print(f"Benchmarking {spec.model_id} at {resolution} on {args.dataset_name}")
            try:
                summary, outcomes = benchmark_one(
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
                    image_batch_size=args.image_batch_size,
                    text_batch_size=args.text_batch_size,
                    precision=args.precision,
                    scoring_path=("sigmoid_dual_embedding" if summary["model_id"].startswith("siglip") else "contrastive_dual_embedding"),
                    protocol_variant="full_similarity_matrix",
                    warmup_count=args.warmup_batches,
                    timed_repeat_count=args.timed_repeats,
                )],
            )
            if outcomes:
                append_rows(args.output_dir / "grouped_outcomes.csv", outcomes)


if __name__ == "__main__":
    main()
