from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def mark_frontier(
    rows: list[dict],
    performance_key: str,
    cost_keys: tuple[str, ...],
    output_key: str,
    group_keys: tuple[str, ...] = ("task", "dataset", "performance_metric"),
) -> None:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[tuple(row.get(key, "") for key in group_keys)].append(index)

    for row in rows:
        row[output_key] = "False"

    for indexes in grouped.values():
        candidates = [index for index in indexes if _usable(rows[index], performance_key, cost_keys)]
        for index in candidates:
            if not any(
                _dominates(rows[other], rows[index], performance_key, cost_keys)
                for other in candidates
                if other != index
            ):
                rows[index][output_key] = "True"


def select_frontier(rows: Iterable[dict], key: str) -> list[dict]:
    return [row for row in rows if str(row.get(key, "")).lower() == "true"]


def _dominates(left: dict, right: dict, performance_key: str, cost_keys: tuple[str, ...]) -> bool:
    left_performance = as_float(left[performance_key])
    right_performance = as_float(right[performance_key])
    left_costs = [as_float(left[key]) for key in cost_keys]
    right_costs = [as_float(right[key]) for key in cost_keys]

    no_worse = left_performance >= right_performance and all(
        left_cost <= right_cost for left_cost, right_cost in zip(left_costs, right_costs)
    )
    strict = left_performance > right_performance or any(
        left_cost < right_cost for left_cost, right_cost in zip(left_costs, right_costs)
    )
    return no_worse and strict


def _usable(row: dict, performance_key: str, cost_keys: tuple[str, ...]) -> bool:
    values = [as_float(row.get(performance_key, ""))]
    values.extend(as_float(row.get(key, "")) for key in cost_keys)
    return all(math.isfinite(value) for value in values)


def as_float(value) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
