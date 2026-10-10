from __future__ import annotations

import math
import statistics
import time
from contextlib import nullcontext
from typing import Callable, TypeVar


T = TypeVar("T")


def device_name(torch_module, device: str) -> str:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        index = torch_module.cuda.current_device()
        return torch_module.cuda.get_device_name(index)
    return "cpu"


def resolve_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def inference_context(torch_module, device: str, precision: str):
    if device.startswith("cuda") and precision == "fp16":
        return torch_module.autocast(device_type="cuda", dtype=torch_module.float16)
    if device.startswith("cuda") and precision == "bf16":
        return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def synchronize(torch_module, device: str) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.synchronize()


def reset_peak_memory(torch_module, device: str) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()


def reset_peak_memory_stats(torch_module, device: str) -> None:
    """Reset CUDA peaks without emptying the allocator cache."""
    if device.startswith("cuda"):
        torch_module.cuda.reset_peak_memory_stats()


def peak_memory_gb(torch_module, device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    return torch_module.cuda.max_memory_allocated() / 1024**3


def peak_memory_reserved_gb(torch_module, device: str) -> float:
    """Return the CUDA allocator's peak reserved memory for the current device."""
    if not device.startswith("cuda"):
        return 0.0
    return torch_module.cuda.max_memory_reserved() / 1024**3


def timed_call(torch_module, device: str, fn: Callable[[], T]) -> tuple[T, float]:
    synchronize(torch_module, device)
    start = time.perf_counter()
    value = fn()
    synchronize(torch_module, device)
    return value, (time.perf_counter() - start) * 1000.0


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "latency_mean_ms": math.nan,
            "latency_std_ms": math.nan,
            "latency_p50_ms": math.nan,
            "latency_p95_ms": math.nan,
        }
    return {
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_std_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
