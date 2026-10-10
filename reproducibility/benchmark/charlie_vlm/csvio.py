from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields = read_header(path) if path.exists() else []
    fieldnames = _merge_fields(existing_fields, rows)
    existing_rows = read_rows(path) if existing_fields and existing_fields != fieldnames else []

    mode = "a" if existing_fields == fieldnames else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            writer.writerows(existing_rows)
        writer.writerows(rows)


def write_rows(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    fields = fieldnames or _merge_fields([], materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _merge_fields(existing_fields: list[str], rows: Iterable[dict]) -> list[str]:
    fields = list(existing_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields
