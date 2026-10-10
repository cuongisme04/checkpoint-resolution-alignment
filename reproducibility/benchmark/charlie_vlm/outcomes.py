"""Privacy-safe outcome records for paired and grouped uncertainty analysis.

The benchmark summaries are sufficient for point estimates, but not for
paired resampling.  This module emits stable hashed sample identifiers and
contains no image paths, captions, class names, or account-specific values.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable


OUTCOME_SCHEMA_VERSION = "dicta-outcomes-v1"
RETRIEVAL_K_VALUES = (1, 5, 10)


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_record(
    *,
    config_id: str,
    run_id: str,
    environment_id: str,
    adapter: str,
    model_id: str,
    checkpoint_id: str,
    task: str,
    dataset: str,
    manifest_sha256: str,
    resolution: int,
    image_batch_size: int,
    text_batch_size: int,
    precision: str,
    scoring_path: str,
    protocol_variant: str,
    warmup_count: int,
    timed_repeat_count: int,
) -> dict:
    """Create the path-free index row that makes an opaque config ID auditable."""
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "configuration_id": config_id,
        "run_id": run_id,
        "environment_id": environment_id,
        "adapter": adapter,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "task": task,
        "dataset": dataset,
        "manifest_sha256": manifest_sha256,
        "resolution": int(resolution),
        "image_batch_size": int(image_batch_size),
        "text_batch_size": int(text_batch_size),
        "precision": precision,
        "scoring_path": scoring_path,
        "protocol_variant": protocol_variant,
        "warmup_count": int(warmup_count),
        "timed_repeat_count": int(timed_repeat_count),
    }


def stable_sample_id(dataset: str, source_id: str) -> str:
    """Hash a dataset-local identifier without retaining paths or raw IDs."""
    if not dataset.strip() or not source_id.strip():
        raise ValueError("dataset and source_id must be non-empty")
    return f"sample-{_stable_digest([OUTCOME_SCHEMA_VERSION, dataset, source_id])}"


def configuration_id(
    *,
    model_id: str,
    task: str,
    dataset: str,
    resolution: int,
    image_batch_size: int,
    text_batch_size: int,
    precision: str,
    scoring_path: str,
    checkpoint_id: str,
    protocol_variant: str,
) -> str:
    """Return an immutable ID for one measured operational configuration."""
    payload = {
        "schema": OUTCOME_SCHEMA_VERSION,
        "model_id": model_id,
        "task": task,
        "dataset": dataset,
        "resolution": int(resolution),
        "image_batch_size": int(image_batch_size),
        "text_batch_size": int(text_batch_size),
        "precision": precision,
        "scoring_path": scoring_path,
        "checkpoint_id": checkpoint_id,
        "protocol_variant": protocol_variant,
    }
    return f"config-{_stable_digest(payload)}"


def classification_outcome(
    *,
    config_id: str,
    run_id: str,
    environment_id: str,
    dataset: str,
    source_sample_id: str,
    correct_top1: bool,
    correct_top5: bool,
) -> dict:
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "configuration_id": config_id,
        "run_id": run_id,
        "environment_id": environment_id,
        "dataset": dataset,
        "sample_id": stable_sample_id(dataset, source_sample_id),
        "correct_top1": int(correct_top1),
        "correct_top5": int(correct_top5),
    }


def retrieval_metrics_and_outcomes(
    image_features,
    text_features,
    image_ids: list[str],
    rows: Iterable,
    *,
    dataset: str,
    config_id: str,
    run_id: str,
    environment_id: str,
) -> tuple[dict, list[dict]]:
    """Compute retrieval metrics and one resampling unit per source image.

    Each grouped row stores image-to-text hit indicators and the number of
    text-to-image successes among that image's captions.  Resampling these
    rows therefore preserves the natural five-caption Flickr30k group.
    """
    rows = list(rows)
    similarities = image_features @ text_features.T
    expected_shape = (len(image_ids), len(rows))
    if tuple(similarities.shape) != expected_shape:
        raise ValueError(
            f"similarity matrix has shape {tuple(similarities.shape)}; expected {expected_shape}"
        )
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    if len(image_index) != len(image_ids):
        raise ValueError("image_ids must be unique")

    correct_captions: dict[int, set[int]] = defaultdict(set)
    caption_image_indexes = []
    caption_ids = [row.caption_id for row in rows]
    if len(set(caption_ids)) != len(caption_ids):
        raise ValueError("caption_id values must be unique")
    for caption_index, row in enumerate(rows):
        if row.image_id not in image_index:
            raise ValueError(f"caption references unknown image id: {row.image_id}")
        target_image = image_index[row.image_id]
        correct_captions[target_image].add(caption_index)
        caption_image_indexes.append(target_image)

    if len(correct_captions) != len(image_ids):
        raise ValueError("every image must have at least one caption")

    image_hits = {k: [0] * len(image_ids) for k in RETRIEVAL_K_VALUES}
    caption_hit_counts = {k: [0] * len(image_ids) for k in RETRIEVAL_K_VALUES}
    for k in RETRIEVAL_K_VALUES:
        text_k = min(k, similarities.shape[1])
        image_k = min(k, similarities.shape[0])
        for current_image in range(similarities.shape[0]):
            top_captions = similarities[current_image].topk(text_k).indices.tolist()
            image_hits[k][current_image] = int(
                any(index in correct_captions[current_image] for index in top_captions)
            )
        for caption_index, target_image in enumerate(caption_image_indexes):
            top_images = similarities[:, caption_index].topk(image_k).indices.tolist()
            caption_hit_counts[k][target_image] += int(target_image in top_images)

    caption_counts = [len(correct_captions[index]) for index in range(len(image_ids))]
    metrics = {}
    for k in RETRIEVAL_K_VALUES:
        metrics[f"image_to_text_recall_at_{k}"] = sum(image_hits[k]) / len(image_ids)
        metrics[f"text_to_image_recall_at_{k}"] = sum(caption_hit_counts[k]) / len(rows)
    metrics["mean_recall_at_1"] = (
        metrics["image_to_text_recall_at_1"]
        + metrics["text_to_image_recall_at_1"]
    ) / 2.0
    metrics["standard_mean_recall"] = sum(
        metrics[f"{direction}_recall_at_{k}"]
        for direction in ("image_to_text", "text_to_image")
        for k in RETRIEVAL_K_VALUES
    ) / 6.0

    outcomes = []
    for image_position, image_id in enumerate(image_ids):
        outcome = {
            "schema_version": OUTCOME_SCHEMA_VERSION,
            "configuration_id": config_id,
            "run_id": run_id,
            "environment_id": environment_id,
            "dataset": dataset,
            "sample_id": stable_sample_id(dataset, image_id),
            "caption_count": caption_counts[image_position],
        }
        for k in RETRIEVAL_K_VALUES:
            outcome[f"image_to_text_hit_at_{k}"] = image_hits[k][image_position]
            outcome[f"text_to_image_hits_at_{k}"] = caption_hit_counts[k][image_position]
        outcomes.append(outcome)
    return metrics, outcomes
