from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("merge_mock_review_results.py")
SPEC = importlib.util.spec_from_file_location("merge_mock_review_results", MODULE_PATH)
MERGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MERGE)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def new_row(model: str, task: str, dataset: str, resolution: int, batch: int, precision: str, text_batch: int = 0) -> dict:
    return {
        "model_id": model,
        "task": task,
        "dataset": dataset,
        "resolution": str(resolution),
        "performance_metric": "top1_accuracy" if task == "zeroshot_classification" else "mean_recall_at_1",
        "performance_value": "0.8",
        "batch_size": str(batch),
        "text_batch_size": str(text_batch) if text_batch else "",
        "precision": precision,
        "gpu_name": MERGE.EXPECTED_GPU,
        "configuration_id": f"{model}-{task}-{resolution}",
        "throughput_examples_s": "100",
        "peak_vram_gb": "2",
        "manifest": "/private/path/manifest.csv",
    }


class MergeMockReviewResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline.csv"
        self.classification = self.root / "classification.csv"
        self.retrieval = self.root / "retrieval.csv"
        self.protocol = self.root / "protocol.json"
        self.output = self.root / "merged.csv"

        retained = [
            {"model_id": f"retained_{index}", "task": "zeroshot_classification", "dataset": "oxford_iiit_pet", "resolution": "224"}
            for index in range(24)
        ]
        historical_blip = [
            {"model_id": model, "task": "zeroshot_classification", "dataset": "oxford_iiit_pet", "resolution": str(resolution)}
            for model in sorted(MERGE.BLIP_MODELS) for resolution in (336, 448)
        ]
        historical_retrieval = [
            {"model_id": f"old_{index}", "task": "image_text_retrieval", "dataset": "flickr30k_test1k", "resolution": "224"}
            for index in range(22)
        ]
        write_csv(self.baseline, retained + historical_blip + historical_retrieval)

        class_group = {
            "models": sorted(MERGE.BLIP_MODELS), "resolutions": [336, 384, 448],
            "image_batch_size": 32, "precision": "fp16",
        }
        retrieval_group = {
            "models": [f"model_{index}" for index in range(8)], "resolutions": [224, 336, 448],
            "image_batch_size": 32, "text_batch_size": 256, "precision": "fp16",
        }
        self.protocol.write_text(json.dumps({"classification": [class_group], "retrieval": [retrieval_group]}))
        write_csv(self.classification, [
            new_row(model, "zeroshot_classification", "oxford_iiit_pet", resolution, 32, "fp16")
            for model in class_group["models"] for resolution in class_group["resolutions"]
        ])
        write_csv(self.retrieval, [
            new_row(model, "image_text_retrieval", "flickr30k_test1k", resolution, 32, "fp16", 256)
            for model in retrieval_group["models"] for resolution in retrieval_group["resolutions"]
        ])
        self.checksum = hashlib.sha256(self.baseline.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def merge(self):
        return MERGE.merge_results(
            self.baseline, self.classification, self.retrieval, self.protocol, self.output,
            expected_baseline_sha256=self.checksum,
        )

    def test_merges_to_54_unique_rows_and_sanitizes_paths(self):
        rows = self.merge()
        self.assertEqual(len(rows), 54)
        self.assertEqual(len({(row["model_id"], row["task"], row["resolution"]) for row in rows}), 54)
        self.assertNotIn("/private/path", self.output.read_text())

    def test_rejects_incomplete_retrieval_summary(self):
        _, rows = MERGE.read_csv(self.retrieval)
        write_csv(self.retrieval, rows[:-1])
        with self.assertRaisesRegex(MERGE.MergeError, "configuration mismatch"):
            self.merge()

    def test_rejects_wrong_gpu(self):
        _, rows = MERGE.read_csv(self.classification)
        rows[0]["gpu_name"] = "NVIDIA RTX A6000"
        write_csv(self.classification, rows)
        with self.assertRaisesRegex(MERGE.MergeError, "unexpected GPU"):
            self.merge()

    def test_rejects_changed_baseline(self):
        with self.assertRaisesRegex(MERGE.MergeError, "checksum changed"):
            MERGE.merge_results(
                self.baseline, self.classification, self.retrieval, self.protocol, self.output,
                expected_baseline_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
