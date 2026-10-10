from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    pretrained: str
    architecture_family: str
    visual_architecture: str
    scale: str
    native_resolution: int

    @classmethod
    def from_json(cls, row: dict) -> "ModelSpec":
        return cls(
            model_id=row["id"],
            model_name=row["model_name"],
            pretrained=row["pretrained"],
            architecture_family=row["architecture_family"],
            visual_architecture=row["visual_architecture"],
            scale=row["scale"],
            native_resolution=int(row["native_resolution"]),
        )


def load_registry(path: Path) -> tuple[dict[str, ModelSpec], list[int]]:
    with path.open() as handle:
        payload = json.load(handle)

    specs = [ModelSpec.from_json(row) for row in payload["models"]]
    registry = {spec.model_id: spec for spec in specs}
    if len(registry) != len(specs):
        raise ValueError(f"Duplicate model id in {path}")

    resolutions = [int(value) for value in payload["default_resolutions"]]
    return registry, resolutions


def select_models(registry: dict[str, ModelSpec], model_ids: Iterable[str]) -> list[ModelSpec]:
    ids = list(model_ids)
    if ids == ["all"]:
        return list(registry.values())

    missing = [model_id for model_id in ids if model_id not in registry]
    if missing:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model ids {missing}. Available ids: {options}")
    return [registry[model_id] for model_id in ids]
