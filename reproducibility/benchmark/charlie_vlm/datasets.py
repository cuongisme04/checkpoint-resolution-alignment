from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PIL import Image

from .manifests import ClassificationRow, RetrievalRow


class ClassificationDataset:
    def __init__(self, rows: list[ClassificationRow], transform, class_to_index: dict[str, int]):
        self.rows = rows
        self.transform = transform
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = _load_rgb(row.image_path)
        return self.transform(image), self.class_to_index[row.class_id], index


class RetrievalImageDataset:
    def __init__(self, rows: list[RetrievalRow], transform):
        unique = OrderedDict()
        for row in rows:
            unique.setdefault(row.image_id, row.image_path)
        self.image_ids = list(unique.keys())
        self.image_paths = list(unique.values())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image = _load_rgb(self.image_paths[index])
        return self.transform(image), index


def _load_rgb(path: Path):
    with Image.open(path) as image:
        return image.convert("RGB")
