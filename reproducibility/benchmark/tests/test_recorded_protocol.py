from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_recorded_protocol.py"
SPEC = importlib.util.spec_from_file_location("run_recorded_protocol", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class RecordedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = PROJECT_ROOT / "configs" / "recorded_protocol.json"
        cls.protocol = json.loads(cls.protocol_path.read_text())

    def test_cell_counts_and_paligemma_settings_are_locked(self):
        counts = RUNNER.validate_protocol(self.protocol)
        self.assertEqual(counts, {"classification": 28, "retrieval": 22})

    def test_dry_run_builds_all_command_groups_with_explicit_timing_flags(self):
        args = argparse.Namespace(
            pet_manifest=Path("pet.csv"),
            flickr_manifest=Path("flickr.csv"),
            output_root=Path("outputs"),
            python_executable="python3",
        )
        commands = RUNNER.build_commands(self.protocol, args)
        self.assertEqual(len(commands), 15)
        for command, _ in commands:
            self.assertIn("--precision", command)
            self.assertIn("--warmup-batches", command)
            self.assertIn("--timed-repeats", command)
            self.assertIn("--fail-fast", command)

    def test_paligemma_448_uses_five_warmups_and_batch_one(self):
        args = argparse.Namespace(
            pet_manifest=Path("pet.csv"),
            flickr_manifest=Path("flickr.csv"),
            output_root=Path("outputs"),
            python_executable="python3",
        )
        commands = RUNNER.build_commands(self.protocol, args)
        matches = [
            (command, env)
            for command, env in commands
            if "paligemma_3b_mix_448" in command
        ]
        self.assertEqual(len(matches), 1)
        command, env = matches[0]
        self.assertEqual(command[command.index("--batch-size") + 1], "1")
        self.assertEqual(command[command.index("--warmup-batches") + 1], "5")
        self.assertEqual(command[command.index("--timed-repeats") + 1], "1")
        self.assertEqual(command[command.index("--prompt-chunk-size") + 1], "4")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_protocol_matches_every_locked_canonical_configuration(self):
        configured = set()
        task_names = {
            "classification": "zeroshot_classification",
            "retrieval": "image_text_retrieval",
        }
        for task, canonical_task in task_names.items():
            for group in self.protocol[task]:
                text_batch_size = group.get("text_batch_size", 0)
                for model in group["models"]:
                    for resolution in group["resolutions"]:
                        configured.add(
                            (
                                canonical_task,
                                model,
                                resolution,
                                group["image_batch_size"],
                                text_batch_size,
                                group["precision"],
                                group["warmup_batches"],
                                group["timed_repeats"],
                            )
                        )

        repo_root = PROJECT_ROOT.parents[1]
        canonical_path = (
            repo_root
            / "results" / "historical" / "summary_50.csv"
        )
        with canonical_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        canonical = {
            (
                row["task"],
                row["model_id"],
                int(row["resolution"]),
                int(row["batch_size"]),
                int(row["text_batch_size"] or 0),
                row["precision"],
                int(row["warmup_batches"]),
                int(row["timed_repeats"]),
            )
            for row in rows
        }
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(canonical), 50)
        self.assertEqual(configured, canonical)


if __name__ == "__main__":
    unittest.main()
