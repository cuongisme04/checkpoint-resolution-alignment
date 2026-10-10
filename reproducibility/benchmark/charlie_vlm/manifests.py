from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ClassificationRow:
    image_path: Path
    class_id: str
    class_name: str
    sample_id: str


@dataclass(frozen=True)
class RetrievalRow:
    image_id: str
    image_path: Path
    caption_id: str
    caption: str


def _resolve_path(manifest_path: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else manifest_path.parent / path


def load_classification_manifest(path: Path) -> list[ClassificationRow]:
    rows = []
    sample_ids = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "class_id", "class_name"}
        _require_columns(path, reader.fieldnames, required)
        for row in reader:
            raw_image_path = row["image_path"]
            parsed = ClassificationRow(
                image_path=_resolve_path(path, raw_image_path),
                class_id=row["class_id"],
                class_name=row["class_name"],
                sample_id=(row.get("sample_id") or "").strip()
                or f"{row['class_id']}/{Path(raw_image_path).name}",
            )
            if parsed.sample_id in sample_ids:
                raise ValueError(
                    f"Classification manifest {path} has duplicate "
                    f"sample_id {parsed.sample_id!r}"
                )
            sample_ids.add(parsed.sample_id)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Classification manifest {path} has no rows")
    return rows


def load_retrieval_manifest(path: Path) -> list[RetrievalRow]:
    rows = []
    caption_ids = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "image_path", "caption_id", "caption"}
        _require_columns(path, reader.fieldnames, required)
        for row in reader:
            parsed = RetrievalRow(
                image_id=row["image_id"],
                image_path=_resolve_path(path, row["image_path"]),
                caption_id=row["caption_id"],
                caption=row["caption"],
            )
            if parsed.caption_id in caption_ids:
                raise ValueError(
                    f"Retrieval manifest {path} has duplicate "
                    f"caption_id {parsed.caption_id!r}"
                )
            caption_ids.add(parsed.caption_id)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Retrieval manifest {path} has no rows")
    return rows


def class_names_by_id(rows: Iterable[ClassificationRow]) -> dict[str, str]:
    names = {}
    for row in rows:
        previous = names.setdefault(row.class_id, row.class_name)
        if previous != row.class_name:
            raise ValueError(f"Class id {row.class_id} maps to both {previous!r} and {row.class_name!r}")
    return names


def _require_columns(path: Path, actual: list[str] | None, required: set[str]) -> None:
    missing = required - set(actual or [])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {', '.join(sorted(missing))}")
