#!/usr/bin/env python3
"""Download the Karpathy 1k Flickr30k test split via HuggingFace and emit a
retrieval manifest in the schema the Charlie Version retrieval benchmark
expects (``image_id,image_path,caption_id,caption``).

The official Karpathy split is bundled into ``nlphuji/flickr30k`` on the
HuggingFace Hub. That dataset stores PIL images plus per-row metadata
(``split`` in {``train``, ``val``, ``test``}, ``filename``, list of captions),
so we filter to ``test`` rows, materialise each image as a JPEG on disk, and
emit one CSV row per (image, caption) pair (5 captions per image).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Charlie Version retrieval manifest for Flickr30k Karpathy 1k test split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/flickr30k"),
        help="Where to materialise the JPEG images. Created if missing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/flickr30k_test1k.csv"),
        help="Where to write the retrieval manifest CSV.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Cap on number of test images. Defaults to the canonical Karpathy 1k subset.",
    )
    parser.add_argument(
        "--hub-id",
        default="nlphuji/flickr30k",
        help="HuggingFace dataset identifier. The default has a 'test' split that matches Karpathy.",
    )
    return parser.parse_args()


def load_test_split(hub_id: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - env without HF datasets
        raise SystemExit(
            "HuggingFace 'datasets' is required to download Flickr30k. "
            "Install with 'pip install datasets'."
        ) from exc

    # The ``nlphuji/flickr30k`` repo only has a 'test' configuration name that
    # contains the whole dataset; the ``split`` column inside that split tells
    # us which Karpathy partition each row belongs to.
    dataset = load_dataset(hub_id, split="test")

    test_rows = [row for row in dataset if str(row.get("split", "")).lower() == "test"]
    if not test_rows:
        raise SystemExit(
            f"Loaded {len(dataset)} rows from {hub_id} but found no rows tagged "
            "split='test'. Check whether the dataset layout has changed."
        )
    return test_rows


def materialise_image(image, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target_path, format="JPEG", quality=95)


def caption_field(row) -> list[str]:
    captions = row.get("caption")
    if isinstance(captions, list):
        return [str(item) for item in captions]
    if isinstance(captions, str):
        return [captions]
    raise SystemExit(
        f"Could not interpret caption field for row {row.get('img_id', row.get('filename'))}"
    )


def main() -> None:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    test_rows = load_test_split(args.hub_id)

    output_rows = []
    image_count = 0
    for entry in test_rows:
        if image_count >= args.max_images:
            break
        image_id = str(entry.get("img_id", entry.get("filename", image_count)))
        filename = str(entry.get("filename", f"{image_id}.jpg"))
        image_path = args.data_root / "images" / filename
        materialise_image(entry["image"], image_path)
        for offset, caption in enumerate(caption_field(entry)):
            output_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "caption_id": f"{image_id}-{offset}",
                    "caption": caption.strip(),
                }
            )
        image_count += 1

    if not output_rows:
        raise SystemExit("No retrieval rows produced; check the Flickr30k download.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "image_path", "caption_id", "caption"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"Wrote {len(output_rows)} rows ({image_count} unique images) to {args.output}"
    )


if __name__ == "__main__":
    main()
