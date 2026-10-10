#!/usr/bin/env python3
"""Download Oxford-IIIT-Pet via torchvision and emit a classification manifest.

The classification benchmark expects a CSV with columns
``sample_id,image_path,class_id,class_name``. Oxford-IIIT-Pet ships with one folder of
JPEGs plus a ``trainval.txt`` / ``test.txt`` split file rather than the
folders-per-class layout that ``make_classification_manifest.py`` consumes, so
this script reads the split file directly and writes the manifest in one shot.

Default behaviour matches the locked experiment matrix: emit the official 3,669
image test split with 37 fine-grained classes. Pass ``--split trainval`` to
also dump the trainval split if needed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Charlie Version classification manifest for Oxford-IIIT-Pet."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/oxford_pet"),
        help="Where torchvision will download the dataset. Created if missing.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "trainval"),
        default="test",
        help="Which official split to write. Defaults to the 3,669-image test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/pet_test.csv"),
        help="Where to write the manifest CSV.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the torchvision download step. Use this if the dataset is already on disk.",
    )
    return parser.parse_args()


def class_id_from_filename(stem: str) -> str:
    """Oxford-IIIT-Pet filenames are <Breed_Name>_<index>. The breed name (with
    underscores) is the canonical class id used by torchvision."""
    return stem.rsplit("_", 1)[0]


def readable_class_name(class_id: str) -> str:
    return class_id.replace("_", " ").lower()


def ensure_dataset(data_root: Path, split: str, skip_download: bool) -> Path:
    """Trigger the torchvision download and return the path to the images dir."""
    data_root.mkdir(parents=True, exist_ok=True)
    if not skip_download:
        try:
            from torchvision.datasets import OxfordIIITPet
        except ImportError as exc:  # pragma: no cover - env without torchvision
            raise SystemExit(
                "torchvision is required to download Oxford-IIIT-Pet. "
                "Install via 'pip install torchvision' or pass --skip-download."
            ) from exc
        OxfordIIITPet(
            root=str(data_root),
            split=split,
            target_types="category",
            download=True,
        )

    images_dir = data_root / "oxford-iiit-pet" / "images"
    if not images_dir.is_dir():
        raise SystemExit(
            f"Expected {images_dir} after download. If the dataset is at a "
            "different location pass --data-root and --skip-download."
        )
    return images_dir


def read_split_ids(data_root: Path, split: str) -> list[str]:
    split_file = data_root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
    if not split_file.is_file():
        raise SystemExit(
            f"Missing split file {split_file}. Was --skip-download used before "
            "the dataset was actually downloaded?"
        )
    ids = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split()[0])
    if not ids:
        raise SystemExit(f"Split file {split_file} contained no entries")
    return ids


def main() -> None:
    args = parse_args()
    images_dir = ensure_dataset(args.data_root, args.split, args.skip_download)
    image_ids = read_split_ids(args.data_root, args.split)

    rows = []
    missing = 0
    for image_id in image_ids:
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            missing += 1
            continue
        class_id = class_id_from_filename(image_id)
        rows.append(
            {
                "sample_id": image_id,
                "image_path": str(image_path.resolve()),
                "class_id": class_id,
                "class_name": readable_class_name(class_id),
            }
        )

    if missing:
        print(f"Warning: {missing} images referenced by the split file were not on disk")
    if not rows:
        raise SystemExit("Manifest would be empty; something is wrong with the dataset layout")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "image_path", "class_id", "class_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    class_count = len({row["class_id"] for row in rows})
    print(
        f"Wrote {len(rows)} rows ({class_count} classes) for the {args.split} split to {args.output}"
    )


if __name__ == "__main__":
    main()
