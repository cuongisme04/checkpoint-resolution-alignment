#!/usr/bin/env python3
"""Image-text retrieval benchmark for HuggingFace BLIP-1 ITM models.

Mirrors ``bench_openclip_retrieval.py`` but uses BLIP's ITC head (vision +
projection / text encoder + projection) to score each image-caption pair.
Output rows share the same schema as the OpenCLIP retrieval benchmark so all
rows can be combined by ``analyze_results.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from charlie_vlm.csvio import append_rows
from charlie_vlm.hf_registry import HFModelSpec, load_hf_registry, select_hf_models
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
from charlie_vlm.openclip_utils import finite_or_blank


DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "hf_models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuggingFace BLIP-1 ITM image-text retrieval benchmark (ITC scoring)."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--models", nargs="+", default=["blip_itm_base_coco", "blip_itm_large_coco"]
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=None,
        help="Override input resolution(s). Defaults to each model's native_resolution.",
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-grouped-outcomes", action="store_true")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--environment-id", default="unrecorded")
    return parser.parse_args()


def require_transformers():
    try:
        from transformers import (  # noqa: F401
            BlipForImageTextRetrieval,
            BlipProcessor,
        )
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install with 'pip install transformers'."
        ) from exc
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    return BlipForImageTextRetrieval, BlipProcessor


def torch_dtype(torch, precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_blip(spec: HFModelSpec, resolution: int, device: str, precision: str):
    BlipForImageTextRetrieval, BlipProcessor = require_transformers()
    import torch

    processor = BlipProcessor.from_pretrained(spec.hub_id)
    model = BlipForImageTextRetrieval.from_pretrained(
        spec.hub_id, torch_dtype=torch_dtype(torch, precision)
    ).to(device).eval()
    if processor.image_processor.size.get("height") != resolution:
        processor.image_processor.size = {"height": resolution, "width": resolution}
    return processor, model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def visual_tokens(spec: HFModelSpec, resolution: int) -> int | None:
    if not spec.patch_size or resolution % spec.patch_size:
        return None
    grid = resolution // spec.patch_size
    return grid * grid + 1


def model_metadata(spec: HFModelSpec, model, resolution: int) -> dict:
    params = count_parameters(model)
    tokens = visual_tokens(spec, resolution)
    proxy = params * tokens if tokens else ""
    return {
        "model_id": spec.model_id,
        "model_name": spec.model_id,
        "pretrained": spec.hub_id,
        "architecture_family": spec.architecture_family,
        "visual_architecture": spec.visual_architecture,
        "scale": spec.scale,
        "native_resolution": spec.native_resolution,
        "resolution": resolution,
        "pixel_count": resolution * resolution,
        "parameter_count": params,
        "parameter_count_m": params / 1e6,
        "patch_size": spec.patch_size or "",
        "visual_tokens": tokens or "",
        "parameter_token_proxy": proxy,
    }


def encode_images(torch, model, processor, rows, args):
    from PIL import Image

    unique = {}
    for row in rows:
        unique.setdefault(row.image_id, row.image_path)
    image_ids = list(unique.keys())
    image_paths = list(unique.values())

    features = []
    latencies = []
    timed_examples = 0
    for batch_index, start in enumerate(range(0, len(image_paths), args.image_batch_size)):
        paths = image_paths[start : start + args.image_batch_size]
        images = [Image.open(path).convert("RGB") for path in paths]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(
            args.device, dtype=torch_dtype(torch, args.precision)
        )

        def run_batch():
            with torch.inference_mode():
                with inference_context(torch, args.device, args.precision):
                    outputs = model.vision_model(
                        pixel_values=pixel_values,
                        interpolate_pos_encoding=True,
                        return_dict=True,
                    )
                    embeds = outputs.last_hidden_state[:, 0, :]
                    image_feats = model.vision_proj(embeds)
                    return image_feats / image_feats.norm(dim=-1, keepdim=True)

        batch_features = run_batch()
        if batch_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                batch_features, latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += len(paths)
        features.append(batch_features.detach().cpu())
    return image_ids, torch.cat(features), latencies, timed_examples


def encode_captions(torch, model, processor, rows, args):
    captions = [row.caption for row in rows]
    features = []
    latencies = []
    timed_examples = 0
    for batch_index, start in enumerate(range(0, len(captions), args.text_batch_size)):
        batch = captions[start : start + args.text_batch_size]
        inputs = processor(text=batch, padding=True, return_tensors="pt").to(args.device)

        def run_batch():
            with torch.inference_mode():
                with inference_context(torch, args.device, args.precision):
                    outputs = model.text_encoder(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        return_dict=True,
                    )
                    text_embeds = outputs.last_hidden_state[:, 0, :]
                    text_feats = model.text_proj(text_embeds)
                    return text_feats / text_feats.norm(dim=-1, keepdim=True)

        batch_features = run_batch()
        if batch_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                batch_features, latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += len(batch)
        features.append(batch_features.detach().cpu())
    return torch.cat(features), latencies, timed_examples


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


def benchmark_one(torch, spec, resolution, args, rows) -> tuple[dict, list[dict]]:
    processor, model = load_blip(spec, resolution, args.device, args.precision)
    metadata = model_metadata(spec, model, resolution)

    reset_peak_memory(torch, args.device)

    image_ids, image_features, image_latencies, timed_images = encode_images(
        torch, model, processor, rows, args
    )
    text_features, text_latencies, timed_captions = encode_captions(
        torch, model, processor, rows, args
    )
    config_id = configuration_id(
        model_id=spec.model_id,
        task="image_text_retrieval",
        dataset=args.dataset_name,
        resolution=resolution,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        precision=args.precision,
        scoring_path="blip_itc_dual_embedding",
        checkpoint_id=spec.hub_id,
        protocol_variant="full_similarity_matrix",
    )
    metrics, outcomes = retrieval_metrics_and_outcomes(
        image_features,
        text_features,
        image_ids,
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
        "image_count": len(image_ids),
        "caption_count": len(rows),
        "batch_size": args.image_batch_size,
        "text_batch_size": args.text_batch_size,
        "warmup_batches": args.warmup_batches,
        "timed_repeats": args.timed_repeats,
        "throughput_examples_s": (timed_images / image_seconds) if image_seconds else "",
        "text_throughput_examples_s": (timed_captions / text_seconds) if text_seconds else "",
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
        "model_name": spec.model_id,
        "pretrained": spec.hub_id,
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
        raise SystemExit("Install PyTorch first: pip install -r requirements.txt") from exc

    registry = load_hf_registry(args.registry)
    specs = select_hf_models(registry, args.models, require="retrieval")
    args.device = resolve_device(torch, args.device)
    rows = limit_image_rows(load_retrieval_manifest(args.manifest), args.max_images)

    for spec in specs:
        resolutions = args.resolutions or [spec.native_resolution]
        for resolution in resolutions:
            print(f"Benchmarking {spec.model_id} at {resolution} on {args.dataset_name}")
            try:
                summary, outcomes = benchmark_one(torch, spec, resolution, args, rows)
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
                    adapter="blip_itc",
                    model_id=summary["model_id"],
                    checkpoint_id=summary["pretrained"],
                    task=summary["task"],
                    dataset=args.dataset_name,
                    manifest_sha256=file_sha256(args.manifest),
                    resolution=resolution,
                    image_batch_size=args.image_batch_size,
                    text_batch_size=args.text_batch_size,
                    precision=args.precision,
                    scoring_path="blip_itc_dual_embedding",
                    protocol_variant="full_similarity_matrix",
                    warmup_count=args.warmup_batches,
                    timed_repeat_count=args.timed_repeats,
                )],
            )
            if outcomes:
                append_rows(args.output_dir / "grouped_outcomes.csv", outcomes)


if __name__ == "__main__":
    main()
