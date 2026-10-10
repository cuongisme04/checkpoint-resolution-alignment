#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a classification manifest from class-named image folders."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Folder containing one subfolder per class.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def readable_name(class_dir: Path) -> str:
    return class_dir.name.replace("_", " ").replace("-", " ")


def main() -> None:
    args = parse_args()
    class_dirs = sorted(path for path in args.image_root.iterdir() if path.is_dir())
    rows = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "sample_id": str(image_path.relative_to(args.image_root)),
                    "image_path": str(image_path.resolve()),
                    "class_id": class_dir.name,
                    "class_name": readable_name(class_dir),
                }
            )

    if not rows:
        raise SystemExit(f"No images found below {args.image_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "image_path", "class_id", "class_name"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image rows across {len(class_dirs)} class folders to {args.output}")


if __name__ == "__main__":
    main()
