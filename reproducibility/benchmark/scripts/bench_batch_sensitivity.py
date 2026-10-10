#!/usr/bin/env python3
"""Limited cached-input batch-sensitivity microbenchmark.

This does not evaluate Pet or Flickr30k and must not replace the paper's
throughput-normalised operational latency.  Every adapter is timed at the
common boundary: image encoder plus projection/normalisation, from an already
resident contiguous CUDA tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from charlie_vlm.csvio import append_rows
from charlie_vlm.hf_registry import load_hf_registry, select_hf_models
from charlie_vlm.measure import (
    device_name,
    inference_context,
    peak_memory_gb,
    peak_memory_reserved_gb,
    reset_peak_memory_stats,
    resolve_device,
    timed_call,
)
from charlie_vlm.openclip_utils import load_model, normalize, require_openclip
from charlie_vlm.outcomes import OUTCOME_SCHEMA_VERSION, configuration_id
from charlie_vlm.registry import load_registry, select_models


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "mock_review_protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    section = protocol["batch_sensitivity"]
    if section["batch_sizes"] != [1, 32]:
        raise ValueError("batch-sensitivity protocol must retain batch sizes 1 and 32")
    if (section["blocks"], section["warmup_iterations_per_block"], section["timed_iterations_per_block"]) != (5, 10, 50):
        raise ValueError("batch-sensitivity block protocol changed")
    return protocol


def iter_cells(protocol: dict):
    section = protocol["batch_sensitivity"]
    for pair in section["pairs"]:
        for resolution in pair["resolutions"]:
            for batch_size in section["batch_sizes"]:
                yield pair["adapter"], pair["model"], resolution, batch_size


def contiguous_cached_input(torch, batch_size: int, resolution: int, device: str):
    generator = torch.Generator(device=device)
    generator.manual_seed(20260714 + resolution)
    one = torch.randn(
        (1, 3, resolution, resolution),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return one.repeat(batch_size, 1, 1, 1).contiguous()


def load_forward(torch, adapter: str, model_id: str, resolution: int, device: str, precision: str):
    if adapter == "openclip":
        open_clip = require_openclip()
        registry, _ = load_registry(PROJECT_ROOT / "configs" / "models.json")
        spec = select_models(registry, [model_id])[0]
        model, _, _ = load_model(open_clip, spec, resolution, device)

        def forward(images):
            with torch.inference_mode(), inference_context(torch, device, precision):
                return normalize(model.encode_image(images))

        checkpoint_id = f"{spec.model_name}:{spec.pretrained}"
    elif adapter == "blip_itc":
        from scripts.bench_hf_blip_retrieval import load_blip

        registry = load_hf_registry(PROJECT_ROOT / "configs" / "hf_models.json")
        spec = select_hf_models(registry, [model_id], require="retrieval")[0]
        _, model = load_blip(spec, resolution, device, precision)

        def forward(images):
            with torch.inference_mode(), inference_context(torch, device, precision):
                outputs = model.vision_model(
                    pixel_values=images,
                    interpolate_pos_encoding=True,
                    return_dict=True,
                )
                projected = model.vision_proj(outputs.last_hidden_state[:, 0, :])
                return projected / projected.norm(dim=-1, keepdim=True)

        checkpoint_id = spec.hub_id
    else:
        raise ValueError(f"unsupported adapter: {adapter}")
    return forward, checkpoint_id


def benchmark_cell(torch, protocol: dict, adapter: str, model_id: str, resolution: int, batch_size: int, args) -> list[dict]:
    section = protocol["batch_sensitivity"]
    precision = section["precision"]
    forward, checkpoint_id = load_forward(
        torch, adapter, model_id, resolution, args.device, precision
    )
    images = contiguous_cached_input(torch, batch_size, resolution, args.device)
    config_id = configuration_id(
        model_id=model_id,
        task="cached_image_microbenchmark",
        dataset="synthetic_cached_tensor",
        resolution=resolution,
        image_batch_size=batch_size,
        text_batch_size=0,
        precision=precision,
        scoring_path="image_encoder_projection_normalization",
        checkpoint_id=checkpoint_id,
        protocol_variant="five_blocks_10_warmup_50_timed",
    )
    output = []
    for block in range(section["blocks"]):
        for _ in range(section["warmup_iterations_per_block"]):
            forward(images)
        reset_peak_memory_stats(torch, args.device)
        block_rows = []
        for iteration in range(section["timed_iterations_per_block"]):
            _, batch_ms = timed_call(torch, args.device, lambda: forward(images))
            block_rows.append(
                {
                    "schema_version": OUTCOME_SCHEMA_VERSION,
                    "configuration_id": config_id,
                    "run_id": args.run_id,
                    "environment_id": args.environment_id,
                    "model_id": model_id,
                    "checkpoint_id": checkpoint_id,
                    "adapter": adapter,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "precision": precision,
                    "block": block + 1,
                    "iteration": iteration + 1,
                    "batch_latency_ms": batch_ms,
                    "latency_ms_per_image": batch_ms / batch_size,
                    "throughput_images_s": 1000.0 * batch_size / batch_ms,
                    "timing_scope": section["scope"],
                    "gpu_name": device_name(torch, args.device),
                }
            )
        allocated = peak_memory_gb(torch, args.device)
        reserved = peak_memory_reserved_gb(torch, args.device)
        for row in block_rows:
            row["peak_allocated_vram_gb"] = allocated
            row["peak_reserved_vram_gb"] = reserved
        output.extend(block_rows)
    return output


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the locked benchmark environment first") from exc
    protocol = load_protocol(args.protocol)
    args.device = resolve_device(torch, args.device)
    if not args.device.startswith("cuda"):
        raise SystemExit("The batch-sensitivity protocol requires a CUDA device")
    for adapter, model_id, resolution, batch_size in iter_cells(protocol):
        print(f"Batch microbenchmark: {model_id} @ {resolution}, B={batch_size}", flush=True)
        try:
            rows = benchmark_cell(
                torch, protocol, adapter, model_id, resolution, batch_size, args
            )
        except Exception as error:
            append_rows(
                args.output_dir / "failures.csv",
                [{
                    "run_id": args.run_id,
                    "environment_id": args.environment_id,
                    "adapter": adapter,
                    "model_id": model_id,
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }],
            )
            if args.fail_fast:
                raise
            continue
        append_rows(args.output_dir / "raw_batch_timings.csv", rows)


if __name__ == "__main__":
    main()
