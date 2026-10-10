#!/usr/bin/env python3
"""Zero-shot classification benchmark for HuggingFace BLIP-1 ITM models.

Mirrors the OpenCLIP classification script so that all rows live in one
``benchmark_rows.csv`` and one ``analyze_results.py`` invocation can build
Pareto frontiers across families. The classification protocol is **ITC**:
encode the image once via the vision encoder + projection, encode each
``"a photo of a {class_name}."`` prompt via the text encoder + projection,
take argmax of normalised cosine similarity.

That is the fairest match to the OpenCLIP CLIP protocol because both produce
a single image feature and one feature per class, with O(1) image cost.

The script supports an optional ``--itm-rerank-topk`` flag that, after the
ITC argmax, takes the top-k ITC candidates and re-ranks them via the full ITM
head; the re-ranked top-1 is logged as a secondary metric named
``itm_rerank_top1_accuracy``. This is BLIP's intended evaluation pattern.
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
from charlie_vlm.openclip_utils import finite_or_blank


DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "hf_models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuggingFace BLIP-1 ITM zero-shot classification benchmark (ITC scoring)."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "classification"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["blip_itm_base_coco", "blip_itm_large_coco"],
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=None,
        help="Override input resolution(s). Defaults to each model's native_resolution.",
    )
    parser.add_argument("--prompt-template", default="a photo of a {class_name}.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-compact-outcomes", action="store_true")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--environment-id", default="unrecorded")
    parser.add_argument(
        "--itm-rerank-topk",
        type=int,
        default=0,
        help="If >0, re-rank the top-K ITC candidates per image via the ITM head and log itm_rerank_top1_accuracy.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def require_transformers():
    try:
        from transformers import (  # noqa: F401  (imported for side-effect)
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


def encode_text_features(torch, model, processor, class_names, template, device, precision):
    prompts = [template.format(class_name=name) for name in class_names]
    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            text_outputs = model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            text_features = model.text_proj(text_embeds)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, inputs


def encode_image_features(torch, model, pixel_values, device, precision):
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=True,
                return_dict=True,
            )
            image_embeds = vision_outputs.last_hidden_state[:, 0, :]
            image_features = model.vision_proj(image_embeds)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features, vision_outputs.last_hidden_state


def itm_rerank(torch, model, vision_hidden, text_inputs, candidate_indices, device, precision):
    """Re-rank top-K candidates per image using BLIP's ITM head.

    ``vision_hidden`` is the full sequence of image hidden states for one
    image (shape [1, seq, dim]). ``candidate_indices`` is a list of class
    indices to score. Returns the candidate index with the highest ITM score.
    """
    best_score = float("-inf")
    best_index = candidate_indices[0]
    image_attention = torch.ones(vision_hidden.shape[:-1], dtype=torch.long, device=device)
    for class_index in candidate_indices:
        input_ids = text_inputs.input_ids[class_index].unsqueeze(0)
        attention_mask = text_inputs.attention_mask[class_index].unsqueeze(0)
        with torch.inference_mode():
            with inference_context(torch, device, precision):
                fusion = model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=vision_hidden,
                    encoder_attention_mask=image_attention,
                    return_dict=True,
                )
                itm_logits = model.itm_head(fusion.last_hidden_state[:, 0, :])
                match_score = torch.softmax(itm_logits, dim=-1)[0, 1].item()
        if match_score > best_score:
            best_score = match_score
            best_index = class_index
    return best_index


def benchmark_one(torch, spec, resolution: int, args, rows) -> tuple[dict, list[dict], list[dict]]:
    processor, model = load_blip(spec, resolution, args.device, args.precision)
    metadata = model_metadata(spec, model, resolution)

    names = class_names_by_id(rows)
    class_ids = sorted(names)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    class_names = [names[class_id] for class_id in class_ids]

    text_features, text_inputs = encode_text_features(
        torch, model, processor, class_names, args.prompt_template, args.device, args.precision
    )

    correct_top1 = 0
    correct_top5 = 0
    correct_itm = 0
    sample_count = 0
    timed_examples = 0
    latencies = []
    predictions = []
    compact_outcomes = []
    config_id = configuration_id(
        model_id=spec.model_id,
        task="zeroshot_classification",
        dataset=args.dataset_name,
        resolution=resolution,
        image_batch_size=args.batch_size,
        text_batch_size=0,
        precision=args.precision,
        scoring_path="blip_itc_dual_embedding",
        checkpoint_id=spec.hub_id,
        protocol_variant=f"prompt={args.prompt_template};itm_rerank_topk={args.itm_rerank_topk}",
    )

    reset_peak_memory(torch, args.device)

    image_rows = rows[: args.max_samples] if args.max_samples else rows
    for batch_index in range(0, len(image_rows), args.batch_size):
        chunk = image_rows[batch_index : batch_index + args.batch_size]
        from PIL import Image

        images = [Image.open(row.image_path).convert("RGB") for row in chunk]
        targets = torch.tensor(
            [class_to_index[row.class_id] for row in chunk], device=args.device
        )

        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(
            args.device, dtype=torch_dtype(torch, args.precision)
        )

        def run_batch():
            return encode_image_features(torch, model, pixel_values, args.device, args.precision)

        (image_features, vision_hidden) = run_batch()
        if batch_index // args.batch_size >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                (_, _), latency_ms = timed_call(torch, args.device, run_batch)
                latencies.append(latency_ms)
                timed_examples += targets.shape[0]

        logits = image_features @ text_features.T
        top_k = min(5, logits.shape[-1])
        probabilities = logits.softmax(dim=-1)
        confidences, top1 = probabilities.max(dim=-1)
        top5 = logits.topk(top_k, dim=-1).indices
        correct_top1 += (top1 == targets).sum().item()
        correct_top5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        sample_count += targets.shape[0]

        rerank_top1 = top1.clone()
        if args.itm_rerank_topk > 0:
            topk = min(args.itm_rerank_topk, logits.shape[-1])
            top_candidates = logits.topk(topk, dim=-1).indices
            for offset in range(targets.shape[0]):
                candidates = top_candidates[offset].tolist()
                best = itm_rerank(
                    torch,
                    model,
                    vision_hidden[offset : offset + 1],
                    text_inputs,
                    candidates,
                    args.device,
                    args.precision,
                )
                rerank_top1[offset] = best
            correct_itm += (rerank_top1 == targets).sum().item()

        if args.save_predictions:
            for offset, row in enumerate(chunk):
                predictions.append(
                    {
                        **metadata,
                        "dataset": args.dataset_name,
                        "image_path": str(row.image_path),
                        "true_class_id": row.class_id,
                        "true_class_name": row.class_name,
                        "predicted_class_id": class_ids[top1[offset].item()],
                        "predicted_class_name": class_names[top1[offset].item()],
                        "correct_top1": int(top1[offset].item() == targets[offset].item()),
                        "confidence": confidences[offset].item(),
                        "itm_rerank_predicted_class_id": class_ids[rerank_top1[offset].item()]
                        if args.itm_rerank_topk > 0
                        else "",
                    }
                )

        if args.save_compact_outcomes:
            top5_correct = (top5 == targets.unsqueeze(1)).any(dim=1)
            for offset, row in enumerate(chunk):
                compact_outcomes.append(
                    classification_outcome(
                        config_id=config_id,
                        run_id=args.run_id,
                        environment_id=args.environment_id,
                        dataset=args.dataset_name,
                        source_sample_id=row.sample_id,
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
        "itm_rerank_top1_accuracy": (correct_itm / sample_count) if args.itm_rerank_topk > 0 else "",
        "sample_count": sample_count,
        "class_count": len(class_ids),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "timed_repeats": args.timed_repeats,
        "throughput_examples_s": (timed_examples / timed_seconds) if timed_seconds else "",
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
        "model_name": spec.model_id,
        "pretrained": spec.hub_id,
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
        raise SystemExit("Install PyTorch first: pip install -r requirements.txt") from exc

    registry = load_hf_registry(args.registry)
    specs = select_hf_models(registry, args.models, require="classification")
    args.device = resolve_device(torch, args.device)
    rows = load_classification_manifest(args.manifest)

    for spec in specs:
        resolutions = args.resolutions or [spec.native_resolution]
        for resolution in resolutions:
            print(f"Benchmarking {spec.model_id} at {resolution} on {args.dataset_name}")
            try:
                summary, predictions, compact_outcomes = benchmark_one(
                    torch, spec, resolution, args, rows
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
                    adapter="blip_itc",
                    model_id=summary["model_id"],
                    checkpoint_id=summary["pretrained"],
                    task=summary["task"],
                    dataset=args.dataset_name,
                    manifest_sha256=file_sha256(args.manifest),
                    resolution=resolution,
                    image_batch_size=args.batch_size,
                    text_batch_size=0,
                    precision=args.precision,
                    scoring_path="blip_itc_dual_embedding",
                    protocol_variant=f"prompt={args.prompt_template};itm_rerank_topk={args.itm_rerank_topk}",
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
