#!/usr/bin/env python3
"""Zero-shot classification benchmark for HuggingFace PaliGemma.

The default protocol preserves the original experiment: for each
``(image, class)`` pair, compute the log-likelihood of the class prompt under
the PaliGemma decoder and choose the class with the highest score. The script
also supports a generation-primary protocol for PaliGemma-specific audits,
where the model is asked to return one class label or option number and the
output is parsed against the same class list.

Output goes into the same ``benchmark_rows.csv`` schema as the OpenCLIP and
BLIP benchmarks so the existing ``analyze_results.py`` pipeline can merge
protocol-specific runs when the dataset/output directory names keep them
separate.
"""
from __future__ import annotations

import argparse
import sys
import re
import random
from collections import Counter
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
    reset_peak_memory,
    resolve_device,
    timed_call,
)
from charlie_vlm.openclip_utils import finite_or_blank


DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "hf_models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuggingFace PaliGemma zero-shot classification benchmark via log-likelihood scoring."
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
        default=["paligemma_3b_mix_224", "paligemma_3b_mix_448"],
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=None,
        help="Override input resolution(s). Defaults to each model's native_resolution.",
    )
    parser.add_argument(
        "--prompt-template",
        default="a photo of a {class_name}.",
        help="Class prompt used for log-likelihood scoring.",
    )
    parser.add_argument(
        "--question-prefix",
        default="",
        help=(
            "Optional prefix prepended to the prompt (mostly used for PaliGemma's "
            "'<image> answer en: ' style). Default empty matches the CLIP-style 'a photo of a X' protocol."
        ),
    )
    parser.add_argument(
        "--score-normalization",
        choices=["sum", "mean"],
        default="sum",
        help=(
            "How to compare class-prompt negative log-likelihoods. 'sum' preserves "
            "the original protocol; 'mean' normalizes by scored suffix-token count "
            "for prompt-length audit runs."
        ),
    )
    parser.add_argument(
        "--primary-protocol",
        choices=["loglik", "generation"],
        default="loglik",
        help=(
            "Primary PaliGemma classification protocol. 'loglik' preserves the "
            "original class-prompt likelihood experiment. 'generation' asks the "
            "model for one label/option and reports parsed generation accuracy."
        ),
    )
    parser.add_argument(
        "--generation-audit",
        action="store_true",
        help=(
            "Additionally run constrained free-form generation and parse the output "
            "against the class list. Intended for small --max-samples audits, not "
            "the full benchmark matrix."
        ),
    )
    parser.add_argument(
        "--generation-prompt-template",
        default=(
            "answer en: What pet breed is shown in the image? "
            "Reply with exactly one label from this list: {class_list}."
        ),
    )
    parser.add_argument(
        "--generation-list-format",
        choices=["comma", "newline", "numbered"],
        default="comma",
        help="How to format the class options inside the generation prompt.",
    )
    parser.add_argument(
        "--generation-class-order",
        choices=["sorted", "shuffle"],
        default="sorted",
        help="Class order used in the generation prompt. Use shuffle to audit order bias.",
    )
    parser.add_argument(
        "--generation-class-order-seed",
        type=int,
        default=0,
        help="Deterministic seed used when --generation-class-order=shuffle.",
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1, help="Images per forward pass.")
    parser.add_argument(
        "--prompt-chunk-size",
        type=int,
        default=16,
        help="Number of class prompts to score in a single forward call (per image). "
        "Trades VRAM for speed. 37 (all-at-once) is fastest but uses ~5 GB for logits at bf16; 16 is safer.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def require_transformers():
    try:
        from transformers import (  # noqa: F401
            AutoProcessor,
            PaliGemmaForConditionalGeneration,
        )
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install with 'pip install transformers'."
        ) from exc
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    return AutoProcessor, PaliGemmaForConditionalGeneration


def torch_dtype(torch, precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_paligemma(spec: HFModelSpec, device: str, precision: str):
    AutoProcessor, PaliGemmaForConditionalGeneration = require_transformers()
    import torch

    processor = AutoProcessor.from_pretrained(spec.hub_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        spec.hub_id, torch_dtype=torch_dtype(torch, precision)
    ).to(device).eval()
    return processor, model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def visual_tokens(spec: HFModelSpec, resolution: int) -> int | None:
    if not spec.patch_size or resolution % spec.patch_size:
        return None
    grid = resolution // spec.patch_size
    return grid * grid


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


def normalize_label_text(value: str) -> str:
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def label_aliases(class_ids: list[str], class_names: list[str]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for index, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        for candidate in {class_id, class_name}:
            aliases[normalize_label_text(candidate)] = index
        if class_id == "german_shorthaired":
            aliases["german shorthaired pointer"] = index
    return aliases


def parse_option_number(
    normalized_text: str, option_class_indices: list[int] | None
) -> tuple[int | None, str | None]:
    if option_class_indices is None:
        return None, None
    match = re.fullmatch(
        r"(?:the answer is |answer is |option |label |class |number )?(\d{1,3})",
        normalized_text,
    )
    if not match:
        return None, None
    option_number = int(match.group(1))
    if 1 <= option_number <= len(option_class_indices):
        return option_class_indices[option_number - 1], "option_number"
    return None, "unknown_option_number"


def parse_generated_label(
    generated_text: str,
    class_ids: list[str],
    class_names: list[str],
    option_class_indices: list[int] | None = None,
) -> tuple[int | None, str]:
    normalized = normalize_label_text(generated_text)
    if not normalized:
        return None, "empty"

    aliases = label_aliases(class_ids, class_names)
    if normalized in aliases:
        return aliases[normalized], "exact"

    option_index, option_status = parse_option_number(normalized, option_class_indices)
    if option_status is not None:
        return option_index, option_status

    matches = [
        index
        for alias, index in aliases.items()
        if len(alias) >= 3 and re.search(rf"\b{re.escape(alias)}\b", normalized)
    ]
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0], "substring"
    if len(unique_matches) > 1:
        return None, "ambiguous"
    return None, "unknown"


def generation_class_indices(class_count: int, order: str, seed: int) -> list[int]:
    indices = list(range(class_count))
    if order == "sorted":
        return indices
    if order == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(indices)
        return indices
    raise ValueError(f"unsupported generation class order: {order}")


def format_generation_class_list(
    class_names: list[str],
    option_class_indices: list[int],
    list_format: str,
) -> str:
    ordered_names = [class_names[index] for index in option_class_indices]
    if list_format == "comma":
        return ", ".join(ordered_names)
    if list_format == "newline":
        return "\n".join(ordered_names)
    if list_format == "numbered":
        return "\n".join(
            f"{option_number}. {class_name}"
            for option_number, class_name in enumerate(ordered_names, start=1)
        )
    raise ValueError(f"unsupported generation list format: {list_format}")


def precompute_class_prompts(processor, prefix: str, template: str, class_names: list[str]):
    """Tokenise the suffix string for each class once. The image-conditioned
    portion of the prompt is left to the processor at scoring time."""
    return [f"{prefix}{template.format(class_name=name)}" for name in class_names]


def score_class_loglik(
    torch, model, processor, image, prompts: list[str], device: str, precision: str,
    chunk_size: int = 16,
    normalization: str = "sum",
) -> tuple[list[float], object]:
    """Return a list of class log-likelihoods plus the raw model outputs of the
    last chunk (kept so the caller can reuse them if needed).

    Implementation: PaliGemma's forward only computes the prefix-LM
    attention mask (bidirectional on image + prefix, causal on suffix) when
    BOTH ``labels`` and ``token_type_ids`` are passed (see
    ``_merge_input_ids_with_image_features`` in
    ``transformers/models/paligemma/modeling_paligemma.py``, the
    ``if token_type_ids is not None and labels is not None`` branch). The
    canonical way to populate both is to pass the class prompt as the
    processor's ``suffix=`` argument; the processor returns ``token_type_ids``
    (0 for image + prefix, 1 for suffix tokens) and pre-masked ``labels``
    (-100 outside the suffix span).

    For efficiency we batch ``chunk_size`` prompts at a time against the same
    image (so the vision encoder still runs once per chunk and the decoder
    forward processes ``chunk_size`` prompts in parallel). The model's built-in
    ``outputs.loss`` is the BATCH-MEAN cross-entropy, which is not what we
    need; instead we compute the per-row negative log-likelihood manually from
    ``outputs.logits`` and the labels, masking with ignore_index=-100 so only
    suffix tokens contribute.

    With the default ``normalization='sum'``, ``log_lik`` returned per class is
    the TOTAL log P(suffix tokens | image, prefix), matching the original
    experiment. ``normalization='mean'`` divides by the number of scored suffix
    tokens and is intended for audits of class-name length bias.
    """
    if normalization not in {"sum", "mean"}:
        raise ValueError("normalization must be 'sum' or 'mean'")
    log_liks: list[float] = []
    last_outputs = None
    target_dtype = torch_dtype(torch, precision)
    for chunk_start in range(0, len(prompts), chunk_size):
        chunk = prompts[chunk_start : chunk_start + chunk_size]
        inputs = processor(
            text=[""] * len(chunk),
            suffix=chunk,
            images=[image] * len(chunk),
            return_tensors="pt",
            padding="longest",
        ).to(device, dtype=target_dtype)

        with torch.inference_mode():
            with inference_context(torch, device, precision):
                outputs = model(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    token_type_ids=inputs.token_type_ids,
                    labels=inputs.labels,
                )
        last_outputs = outputs

        if outputs.logits is None:
            raise RuntimeError(
                "PaliGemma forward returned no logits; verify the processor was "
                "called with both text='' and suffix=<prompt> so token_type_ids "
                "and labels are populated."
            )

        logits = outputs.logits
        labels = inputs.labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch, seq_minus_one, vocab = shift_logits.shape
        flat_nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch, seq_minus_one)
        nll_per_row = flat_nll.sum(dim=-1)
        if normalization == "mean":
            token_counts = (shift_labels != -100).sum(dim=-1).clamp_min(1)
            nll_per_row = nll_per_row / token_counts
        log_liks.extend((-nll_per_row).tolist())
    return log_liks, last_outputs


def generation_prompt(
    template: str,
    class_names: list[str],
    option_class_indices: list[int],
    list_format: str,
) -> str:
    class_list = format_generation_class_list(
        class_names, option_class_indices, list_format
    )
    return template.format(class_list=class_list)


def generate_label_text(
    torch,
    model,
    processor,
    image,
    prompt: str,
    device: str,
    precision: str,
    max_new_tokens: int,
) -> str:
    target_dtype = torch_dtype(torch, precision)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, dtype=target_dtype
    )
    with torch.inference_mode():
        with inference_context(torch, device, precision):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    input_length = inputs.input_ids.shape[-1]
    generated_ids = generated[0, input_length:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def benchmark_one(torch, spec, resolution: int, args, rows) -> tuple[dict, list[dict]]:
    processor, model = load_paligemma(spec, args.device, args.precision)
    metadata = model_metadata(spec, model, resolution)

    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"height": resolution, "width": resolution}

    names = class_names_by_id(rows)
    class_ids = sorted(names)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    class_names = [names[class_id] for class_id in class_ids]

    prompts = precompute_class_prompts(
        processor, args.question_prefix, args.prompt_template, class_names
    )
    run_loglik = args.primary_protocol == "loglik"
    run_generation = args.generation_audit or args.primary_protocol == "generation"
    generation_indices = generation_class_indices(
        len(class_names), args.generation_class_order, args.generation_class_order_seed
    )
    gen_prompt = generation_prompt(
        args.generation_prompt_template,
        class_names,
        generation_indices,
        args.generation_list_format,
    )

    image_rows = rows[: args.max_samples] if args.max_samples else rows

    correct_loglik_top1 = 0
    correct_generation = 0
    sample_count = 0
    timed_examples = 0
    latencies = []
    predictions = []
    generation_status_counts: Counter[str] = Counter()
    generated_prediction_counts: Counter[str] = Counter()

    reset_peak_memory(torch, args.device)

    from PIL import Image
    from tqdm import tqdm

    progress = tqdm(image_rows, desc=f"{spec.model_id}@{resolution}", unit="img")
    for image_index, row in enumerate(progress):
        image = Image.open(row.image_path).convert("RGB")

        def run_loglik_one():
            return score_class_loglik(
                torch, model, processor, image, prompts, args.device, args.precision,
                chunk_size=args.prompt_chunk_size,
                normalization=args.score_normalization,
            )

        def run_generation_one():
            return generate_label_text(
                torch,
                model,
                processor,
                image,
                gen_prompt,
                args.device,
                args.precision,
                args.generation_max_new_tokens,
            )

        target_index = class_to_index[row.class_id]

        log_liks: list[float] = []
        best_index = None
        correct: int | str = ""
        if run_loglik:
            log_liks, _ = run_loglik_one()
            best_index = max(range(len(log_liks)), key=lambda idx: log_liks[idx])
            correct = int(best_index == target_index)
            correct_loglik_top1 += correct

        generated_text = ""
        generated_status = ""
        generated_index = None
        generation_correct: int | str = ""
        generation_ran = False
        if args.primary_protocol == "generation":
            generated_text = run_generation_one()
            generation_ran = True

        if image_index >= args.warmup_batches:
            for _ in range(args.timed_repeats):
                if args.primary_protocol == "generation":
                    _, latency_ms = timed_call(torch, args.device, run_generation_one)
                else:
                    (_, _), latency_ms = timed_call(torch, args.device, run_loglik_one)
                latencies.append(latency_ms)
                timed_examples += 1

        sample_count += 1

        if run_generation:
            if not generation_ran:
                generated_text = run_generation_one()
                generation_ran = True
            generated_index, generated_status = parse_generated_label(
                generated_text,
                class_ids,
                class_names,
                option_class_indices=generation_indices,
            )
            generation_correct = int(generated_index == target_index)
            correct_generation += generation_correct
            generation_status_counts[generated_status] += 1
            if generated_index is not None:
                generated_prediction_counts[class_ids[generated_index]] += 1

        primary_index = generated_index if args.primary_protocol == "generation" else best_index
        primary_correct = generation_correct if args.primary_protocol == "generation" else correct

        if args.save_predictions:
            predictions.append(
                {
                    **metadata,
                    "dataset": args.dataset_name,
                    "image_path": str(row.image_path),
                    "true_class_id": row.class_id,
                    "true_class_name": row.class_name,
                    "primary_protocol": args.primary_protocol,
                    "primary_predicted_class_id": class_ids[primary_index]
                    if primary_index is not None
                    else "",
                    "primary_predicted_class_name": class_names[primary_index]
                    if primary_index is not None
                    else "",
                    "primary_correct": primary_correct,
                    "predicted_class_id": class_ids[best_index]
                    if best_index is not None
                    else "",
                    "predicted_class_name": class_names[best_index]
                    if best_index is not None
                    else "",
                    "correct_top1": correct,
                    "log_likelihood": log_liks[best_index]
                    if best_index is not None
                    else "",
                    "score_normalization": args.score_normalization,
                    "generated_text": generated_text,
                    "generated_parse_status": generated_status,
                    "generated_predicted_class_id": class_ids[generated_index]
                    if generated_index is not None
                    else "",
                    "generated_predicted_class_name": class_names[generated_index]
                    if generated_index is not None
                    else "",
                    "generation_correct": generation_correct,
                }
            )

    latency = latency_summary(latencies)
    timed_seconds = sum(latencies) / 1000.0
    primary_correct_count = (
        correct_generation
        if args.primary_protocol == "generation"
        else correct_loglik_top1
    )
    primary_accuracy = primary_correct_count / sample_count
    most_common_generated = generated_prediction_counts.most_common(1)
    most_common_generated_class_id = most_common_generated[0][0] if most_common_generated else ""
    most_common_generated_count = most_common_generated[0][1] if most_common_generated else 0
    generation_parse_failures = sum(
        generation_status_counts.get(status, 0)
        for status in ("empty", "unknown", "unknown_option_number", "ambiguous")
    )
    summary = {
        **metadata,
        "task": "zeroshot_classification",
        "dataset": args.dataset_name,
        "performance_metric": (
            "generation_top1_accuracy"
            if args.primary_protocol == "generation"
            else "top1_accuracy"
        ),
        "performance_value": primary_accuracy,
        "top1_accuracy": primary_accuracy,
        "primary_protocol": args.primary_protocol,
        "loglik_top1_accuracy": (correct_loglik_top1 / sample_count)
        if run_loglik and sample_count
        else "",
        "generation_top1_accuracy": (correct_generation / sample_count)
        if run_generation and sample_count
        else "",
        "sample_count": sample_count,
        "class_count": len(class_ids),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "timed_repeats": args.timed_repeats,
        "throughput_examples_s": (timed_examples / timed_seconds) if timed_seconds else "",
        "peak_vram_gb": peak_memory_gb(torch, args.device),
        "device": args.device,
        "gpu_name": device_name(torch, args.device),
        "precision": args.precision,
        "prompt_template": args.prompt_template,
        "score_normalization": args.score_normalization,
        "generation_audit": int(run_generation),
        "generation_prompt_template": args.generation_prompt_template,
        "generation_list_format": args.generation_list_format,
        "generation_class_order": args.generation_class_order,
        "generation_class_order_seed": (
            args.generation_class_order_seed
            if args.generation_class_order == "shuffle"
            else ""
        ),
        "generation_audit_accuracy": (correct_generation / sample_count)
        if run_generation and sample_count
        else "",
        "generation_empty_rate": (generation_status_counts.get("empty", 0) / sample_count)
        if run_generation and sample_count
        else "",
        "generation_unknown_rate": (
            (
                generation_status_counts.get("unknown", 0)
                + generation_status_counts.get("unknown_option_number", 0)
            )
            / sample_count
        )
        if run_generation and sample_count
        else "",
        "generation_ambiguous_rate": (generation_status_counts.get("ambiguous", 0) / sample_count)
        if run_generation and sample_count
        else "",
        "generation_exact_rate": (generation_status_counts.get("exact", 0) / sample_count)
        if run_generation and sample_count
        else "",
        "generation_substring_rate": (generation_status_counts.get("substring", 0) / sample_count)
        if run_generation and sample_count
        else "",
        "generation_option_number_rate": (
            generation_status_counts.get("option_number", 0) / sample_count
        )
        if run_generation and sample_count
        else "",
        "generation_parse_failure_rate": (generation_parse_failures / sample_count)
        if run_generation and sample_count
        else "",
        "generation_unique_prediction_count": len(generated_prediction_counts)
        if run_generation
        else "",
        "generation_most_common_predicted_class_id": most_common_generated_class_id
        if run_generation
        else "",
        "generation_most_common_prediction_rate": (
            most_common_generated_count / sample_count
        )
        if run_generation and sample_count
        else "",
        "manifest": str(args.manifest),
        **{key: finite_or_blank(value) for key, value in latency.items()},
    }
    return summary, predictions


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
                summary, predictions = benchmark_one(torch, spec, resolution, args, rows)
            except Exception as error:
                append_rows(args.output_dir / "failures.csv", [failure_row(spec, resolution, args, error)])
                if args.fail_fast:
                    raise
                print(f"Skipped {spec.model_id} at {resolution}: {error}")
                continue
            append_rows(args.output_dir / "benchmark_rows.csv", [summary])
            if predictions:
                append_rows(args.output_dir / "predictions.csv", predictions)


if __name__ == "__main__":
    main()
