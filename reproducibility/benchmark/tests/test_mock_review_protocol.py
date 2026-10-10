from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_mock_review_protocol.py"
SPEC = importlib.util.spec_from_file_location("run_mock_review_protocol", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class MockReviewProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = PROJECT_ROOT / "configs" / "mock_review_protocol.json"
        cls.protocol = json.loads(cls.protocol_path.read_text())
        cls.args = argparse.Namespace(
            protocol=cls.protocol_path,
            pet_manifest=Path("pet.csv"),
            flickr_manifest=Path("flickr.csv"),
            output_root=Path("outputs"),
            run_id="test-run",
            environment_id="test-env",
            python_executable="python3",
            stage="all",
        )

    def test_expected_cell_counts_are_locked(self):
        self.assertEqual(
            RUNNER.validate_protocol(self.protocol),
            {
                "pet_blip_validation": 6,
                "flickr_grouped_outcomes": 24,
                "batch_sensitivity": 24,
            },
        )

    def test_commands_retain_outcome_and_trace_flags(self):
        commands = RUNNER.build_commands(self.protocol, self.args)
        self.assertEqual(len(commands), 6)
        outcome_commands = commands[:-1]
        for command in outcome_commands:
            for flag in ("--precision", "--warmup-batches", "--timed-repeats", "--run-id", "--environment-id", "--fail-fast"):
                self.assertIn(flag, command)
            self.assertTrue(
                "--save-compact-outcomes" in command
                or "--save-grouped-outcomes" in command
            )
            self.assertNotIn("--save-predictions", command)
        batch = commands[-1]
        self.assertIn("bench_batch_sensitivity.py", " ".join(batch))
        self.assertIn("--run-id", batch)
        self.assertIn("--environment-id", batch)

    def test_sanitized_commands_do_not_retain_project_root(self):
        command = RUNNER.build_commands(self.protocol, self.args)[0]
        rendered = " ".join(RUNNER.sanitize_command(command, self.args))
        self.assertNotIn(str(PROJECT_ROOT), rendered)


if __name__ == "__main__":
    unittest.main()
