#!/usr/bin/env python3
"""Replay the released analyses and compare them with the recorded results."""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "build" / "verification"
PYTHON = sys.executable


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(
            equivalent(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            equivalent(a, b) for a, b in zip(expected, actual))
    if expected == actual:
        return True
    try:
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def read_result(path):
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    input_paths = sorted((ROOT / "results").rglob("*"))
    before = {str(p.relative_to(ROOT)): digest(p) for p in input_paths if p.is_file()}
    steps = []

    def run(name, *args):
        print(name, flush=True)
        with (LOGS / f"{name}.log").open("w") as log:
            subprocess.run([PYTHON, *args], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        steps.append(name)

    for line in (ROOT / "reproducibility/benchmark/SOURCE_SHA256.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Release benchmark source changed: {name}")

    run("analysis_tests", "-m", "unittest", "discover", "-s", "analysis", "-p", "test_*.py", "-v")
    run("benchmark_tests", "-m", "unittest", "discover", "-s", "reproducibility/benchmark/tests", "-t", "reproducibility/benchmark", "-v")
    run("historical_protocol", "reproducibility/benchmark/scripts/run_recorded_protocol.py", "--dry-run")
    run("focused_protocol", "reproducibility/benchmark/scripts/run_mock_review_protocol.py", "--dry-run")
    run("canonical_analysis", "analysis/regenerate_results.py")
    support = "results/supporting/focused_run"
    run("focused_statistics", "analysis/mock_review_analysis.py",
        "--summary", "results/canonical/dicta_mock_review_54_with_ci.csv",
        "--classification-outcomes", f"{support}/classification/compact_outcomes.csv",
        "--retrieval-outcomes", f"{support}/retrieval/grouped_outcomes.csv",
        "--output-dir", "analysis/processed/mock_review", "--require-complete")
    run("batch_statistics", "analysis/batch_sensitivity_analysis.py",
        "--input", f"{support}/batch_sensitivity/raw_batch_timings.csv",
        "--output-dir", "analysis/processed/mock_review")
    run("camera_ready_pairing", "analysis/camera_ready_blip_pairing.py")
    for figure in ("cost", "resolution", "proxy"):
        run(f"figure_{figure}", f"analysis/render_camera_ready_{figure}_figure.py")

    comparisons = []
    expected_root = ROOT / "results/expected/focused"
    for expected in sorted(expected_root.rglob("*")):
        if not expected.is_file():
            continue
        actual = ROOT / "analysis/processed/mock_review" / expected.relative_to(expected_root)
        if not equivalent(read_result(expected), read_result(actual)):
            raise RuntimeError(f"Reproduced result differs: {expected.relative_to(expected_root)}")
        comparisons.append(str(expected.relative_to(ROOT)))
    expected = ROOT / "results/expected/blip_pet_336_vs_384.csv"
    actual = ROOT / "analysis/processed/camera_ready/blip_pet_336_vs_384.csv"
    if not equivalent(read_result(expected), read_result(actual)):
        raise RuntimeError("Camera-ready paired contrast differs from the recorded result")
    comparisons.append(str(expected.relative_to(ROOT)))
    after = {name: digest(ROOT / name) for name in before}
    if before != after:
        raise RuntimeError("A retained input or expected-result file changed")
    import numpy
    import matplotlib
    report = {"status": "PASS", "python": platform.python_version(),
              "numpy": numpy.__version__, "matplotlib": matplotlib.__version__,
              "steps": steps, "matched_result_files": comparisons,
              "preserved_input_files": len(before), "gpu_inference_performed": False,
              "numeric_tolerance": {"relative": 1e-12, "absolute": 1e-12}}
    (LOGS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
