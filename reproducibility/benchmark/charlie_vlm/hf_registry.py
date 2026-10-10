"""Registry loader for HuggingFace transformers-based VLMs.

Mirrors ``charlie_vlm/registry.py`` but with fields that match the
``transformers`` ecosystem (one ``hub_id`` instead of an OpenCLIP
``model_name`` plus ``pretrained`` pair) and adds task-availability flags so
that retrieval-incompatible models such as PaliGemma can be filtered out
automatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class HFModelSpec:
    model_id: str
    hub_id: str
    architecture_family: str
    visual_architecture: str
    scale: str
    native_resolution: int
    patch_size: int | None
    supports_classification: bool
    supports_retrieval: bool
    notes: str

    @classmethod
    def from_json(cls, row: dict) -> "HFModelSpec":
        patch_size_raw = row.get("patch_size")
        return cls(
            model_id=row["id"],
            hub_id=row["hub_id"],
            architecture_family=row["architecture_family"],
            visual_architecture=row["visual_architecture"],
            scale=row.get("scale", ""),
            native_resolution=int(row["native_resolution"]),
            patch_size=int(patch_size_raw) if patch_size_raw is not None else None,
            supports_classification=bool(row.get("supports_classification", True)),
            supports_retrieval=bool(row.get("supports_retrieval", True)),
            notes=str(row.get("notes", "")),
        )


def load_hf_registry(path: Path) -> dict[str, HFModelSpec]:
    with path.open() as handle:
        payload = json.load(handle)
    specs = [HFModelSpec.from_json(row) for row in payload["models"]]
    registry = {spec.model_id: spec for spec in specs}
    if len(registry) != len(specs):
        raise ValueError(f"Duplicate model id in {path}")
    return registry


def select_hf_models(
    registry: dict[str, HFModelSpec],
    model_ids: Iterable[str],
    require: str | None = None,
) -> list[HFModelSpec]:
    """Return the requested specs, optionally filtering to ones that advertise
    support for a particular task (``'classification'`` or ``'retrieval'``)."""
    ids = list(model_ids)
    if ids == ["all"]:
        specs = list(registry.values())
    else:
        missing = [model_id for model_id in ids if model_id not in registry]
        if missing:
            options = ", ".join(sorted(registry))
            raise ValueError(f"Unknown HF model ids {missing}. Available ids: {options}")
        specs = [registry[model_id] for model_id in ids]

    if require == "classification":
        specs = [spec for spec in specs if spec.supports_classification]
    elif require == "retrieval":
        specs = [spec for spec in specs if spec.supports_retrieval]
    elif require is not None:
        raise ValueError(f"Unknown require filter: {require}")
    return specs
