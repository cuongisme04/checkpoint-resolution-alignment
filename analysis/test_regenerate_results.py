#!/usr/bin/env python3
"""Regression tests for the deterministic DICTA analysis pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analysis" / "regenerate_results.py"
CANONICAL = (
    REPO_ROOT
    / "results" / "historical" / "summary_50.csv"
)
EXPECTED_SHA256 = "2d2fa4103fb557776d9ad5d78f9d0ed030242ba34cde581f8051409bc1e17ea1"
EXPANDED_SHA256 = "4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(directory: Path):
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class AnalysisPipelineTest(unittest.TestCase):
    def test_two_runs_are_byte_identical_and_match_locked_diagnostics(self):
        before = sha256(CANONICAL)
        self.assertEqual(before, EXPECTED_SHA256)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(root / "mpl-cache")
            for output_dir in (first, second):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--input",
                        str(CANONICAL),
                        "--output-dir",
                        str(output_dir),
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("validated_rows=50", completed.stdout)
                self.assertIn("cross_task_joint_intersection=6", completed.stdout)

            self.assertEqual(file_hashes(first), file_hashes(second))
            for figure in (first / "figures").glob("*.pdf"):
                payload = figure.read_bytes()
                self.assertNotIn(
                    b"/Subtype /Type3",
                    payload,
                    f"Type 3 font found in {figure.name}",
                )
            manifest = json.loads((first / "analysis_manifest.json").read_text())
            self.assertEqual(manifest["input"]["sha256"], EXPECTED_SHA256)
            self.assertEqual(manifest["artifact_trace"], "figure_table_trace.json")
            trace = json.loads((first / manifest["artifact_trace"]).read_text())
            artifact_paths = {item["path"] for item in trace["artifacts"]}
            self.assertIn(
                "figures/performance_vs_tau_img_both_tasks.pdf",
                artifact_paths,
            )
            self.assertIn(
                "figures/performance_vs_resolution_both_tasks.pdf",
                artifact_paths,
            )
            self.assertEqual(trace["source"]["sha256"], EXPECTED_SHA256)
            self.assertEqual(trace["source"]["row_count"], 50)
            self.assertIn("marginal Wilson summaries", json.dumps(trace))
            self.assertFalse(
                manifest["diagnostics"]["flickr_joint_frontier_changed"]
            )
            self.assertEqual(
                manifest["diagnostics"]["frontier_counts"][
                    "pareto_latency_vram"
                ],
                {
                    "image_text_retrieval": 11,
                    "zeroshot_classification": 10,
                },
            )

            with (first / "summary_with_tau_and_pareto.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 50)
            added = [
                row
                for row in rows
                if row["model_id"] == "clip_vit_b16_openai"
                and row["task"] == "zeroshot_classification"
                and row["resolution"] == "224"
            ]
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0]["pareto_latency_vram"], "True")
            self.assertEqual(
                added[0]["legacy_pareto_latency_vram_batch_p50"], "False"
            )

        self.assertEqual(sha256(CANONICAL), before)

    def test_default_input_is_authoritative_expanded_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "expanded"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("validated_rows=54", completed.stdout)
            manifest = json.loads(
                (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["input"]["sha256"], EXPANDED_SHA256)
            trace = json.loads(
                (output_dir / "figure_table_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trace["source"]["row_count"], 54)
            self.assertIn("image-grouped bootstrap summaries", json.dumps(trace))


if __name__ == "__main__":
    unittest.main()
