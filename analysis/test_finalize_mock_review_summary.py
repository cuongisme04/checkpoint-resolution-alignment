from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("finalize_mock_review_summary.py")
SPEC = importlib.util.spec_from_file_location("finalize_mock_review_summary", MODULE_PATH)
FINALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FINALIZER)


FIELDS = [
    "configuration_id",
    "task",
    "model_id",
    "resolution",
    "performance_value",
    "ci_lower_95",
    "ci_upper_95",
    "ci_half_width_95",
    "ci_method",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class FinalizeMockReviewSummaryTest(unittest.TestCase):
    def fixture_rows(self):
        retrieval = [
            {
                "configuration_id": f"ret-{index}",
                "task": "image_text_retrieval",
                "model_id": "retrieval-model",
                "resolution": "224",
                "performance_value": "0.8",
                "ci_lower_95": "0",
                "ci_upper_95": "1",
                "ci_half_width_95": "0.5",
                "ci_method": "legacy",
            }
            for index in range(24)
        ]
        focused_classification = [
            {
                "configuration_id": f"cls-new-{index}",
                "task": "zeroshot_classification",
                "model_id": "classification-model",
                "resolution": "384",
                "performance_value": "0.7",
                "ci_lower_95": "0",
                "ci_upper_95": "1",
                "ci_half_width_95": "0.5",
                "ci_method": "legacy",
            }
            for index in range(6)
        ]
        retained_classification = [
            {
                "configuration_id": f"cls-old-{index}",
                "task": "zeroshot_classification",
                "model_id": "classification-model",
                "resolution": "224",
                "performance_value": "0.6",
                "ci_lower_95": "0.55",
                "ci_upper_95": "0.65",
                "ci_half_width_95": "0.05",
                "ci_method": "sample_bootstrap",
            }
            for index in range(24)
        ]
        classification_intervals = [
            {
                "configuration_id": f"cls-new-{index}",
                "performance_value": "0.7",
                "ci_lower_95": "0.65",
                "ci_upper_95": "0.75",
                "ci_method": "sample_bootstrap",
            }
            for index in range(6)
        ]
        retrieval_intervals = [
            {
                "configuration_id": f"ret-{index}",
                "performance_value": "0.8",
                "ci_lower_95": "0.77",
                "ci_upper_95": "0.83",
                "ci_method": "image_grouped_bootstrap",
            }
            for index in range(24)
        ]
        return (
            retrieval + focused_classification + retained_classification,
            classification_intervals,
            retrieval_intervals,
        )

    def test_finalize_attaches_all_new_intervals(self):
        summary, classification, retrieval = self.fixture_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("summary.csv", "class.csv", "ret.csv")]
            for path, rows in zip(paths, (summary, classification, retrieval)):
                write_csv(path, rows)
            output = root / "final.csv"
            rows = FINALIZER.finalize(*paths, output)
            self.assertEqual(len(rows), 54)
            self.assertEqual(rows[0]["ci_method"], "image_grouped_bootstrap")
            self.assertAlmostEqual(float(rows[0]["ci_half_width_95"]), 0.03)
            self.assertEqual(rows[24]["ci_method"], "sample_bootstrap")
            self.assertEqual(rows[-1]["ci_lower_95"], "0.55")

    def test_finalize_rejects_point_estimate_mismatch(self):
        summary, classification, retrieval = self.fixture_rows()
        retrieval[0]["performance_value"] = "0.81"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("summary.csv", "class.csv", "ret.csv")]
            for path, rows in zip(paths, (summary, classification, retrieval)):
                write_csv(path, rows)
            with self.assertRaisesRegex(FINALIZER.FinalizeError, "point estimate mismatch"):
                FINALIZER.finalize(*paths, root / "final.csv")


if __name__ == "__main__":
    unittest.main()
