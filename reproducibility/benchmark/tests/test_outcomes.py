from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


from charlie_vlm.manifests import load_classification_manifest, load_retrieval_manifest
from charlie_vlm.outcomes import (
    configuration_id,
    retrieval_metrics_and_outcomes,
    stable_sample_id,
)


class _Indexes:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _TopK:
    def __init__(self, indexes):
        self.indices = _Indexes(indexes)


class _Vector:
    def __init__(self, values):
        self.values = list(values)

    def topk(self, k):
        order = sorted(range(len(self.values)), key=lambda index: self.values[index], reverse=True)
        return _TopK(order[:k])


class _Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]) if self.rows else 0)

    @property
    def T(self):
        return _Matrix(zip(*self.rows))

    def __matmul__(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("shape mismatch")
        output = []
        for row in self.rows:
            output.append([
                sum(row[k] * other.rows[k][column] for k in range(self.shape[1]))
                for column in range(other.shape[1])
            ])
        return _Matrix(output)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_selector, column = key
            if row_selector != slice(None):
                raise TypeError("unsupported selector")
            return _Vector(row[column] for row in self.rows)
        return _Vector(self.rows[key])


@dataclass(frozen=True)
class _RetrievalRow:
    image_id: str
    caption_id: str


class OutcomeTest(unittest.TestCase):
    def test_stable_id_is_path_free_and_repeatable(self):
        first = stable_sample_id("oxford_iiit_pet", "Abyssinian_1")
        second = stable_sample_id("oxford_iiit_pet", "Abyssinian_1")
        self.assertEqual(first, second)
        self.assertNotIn("Abyssinian", first)
        self.assertTrue(first.startswith("sample-"))

    def test_configuration_id_changes_with_checkpoint_and_protocol(self):
        base = dict(
            model_id="model",
            task="task",
            dataset="data",
            resolution=224,
            image_batch_size=32,
            text_batch_size=0,
            precision="fp16",
            scoring_path="dual",
            checkpoint_id="checkpoint-a",
            protocol_variant="prompt-a",
        )
        original = configuration_id(**base)
        for key, value in (
            ("checkpoint_id", "checkpoint-b"),
            ("resolution", 448),
            ("image_batch_size", 1),
            ("precision", "bf16"),
            ("scoring_path", "other"),
            ("protocol_variant", "prompt-b"),
        ):
            changed = dict(base)
            changed[key] = value
            self.assertNotEqual(original, configuration_id(**changed), key)

    def test_grouped_retrieval_metrics_reconstruct_exactly(self):
        # Two orthogonal image features and four captions, two per image.
        images = _Matrix([[1.0, 0.0], [0.0, 1.0]])
        texts = _Matrix([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        rows = [
            _RetrievalRow("image-a", "a-0"),
            _RetrievalRow("image-a", "a-1"),
            _RetrievalRow("image-b", "b-0"),
            _RetrievalRow("image-b", "b-1"),
        ]
        metrics, outcomes = retrieval_metrics_and_outcomes(
            images,
            texts,
            ["image-a", "image-b"],
            rows,
            dataset="toy",
            config_id="config",
            run_id="run",
            environment_id="env",
        )
        self.assertEqual(metrics["mean_recall_at_1"], 1.0)
        self.assertEqual(metrics["standard_mean_recall"], 1.0)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(sum(row["caption_count"] for row in outcomes), 4)
        self.assertEqual(sum(row["text_to_image_hits_at_1"] for row in outcomes), 4)
        self.assertNotIn("image-a", str(outcomes))
        forbidden = {"image_path", "caption", "class_name", "token", "pod_id"}
        self.assertTrue(all(not (forbidden & set(row)) for row in outcomes))

    def test_grouped_retrieval_rejects_duplicate_caption_ids_and_bad_shape(self):
        images = _Matrix([[1.0, 0.0]])
        texts = _Matrix([[1.0, 0.0]])
        duplicate = [_RetrievalRow("a", "same"), _RetrievalRow("a", "same")]
        with self.assertRaisesRegex(ValueError, "shape"):
            retrieval_metrics_and_outcomes(
                images, texts, ["a"], duplicate,
                dataset="toy", config_id="c", run_id="r", environment_id="e"
            )
        duplicate_texts = _Matrix([[1.0, 0.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "caption_id"):
            retrieval_metrics_and_outcomes(
                images, duplicate_texts, ["a"], duplicate,
                dataset="toy", config_id="c", run_id="r", environment_id="e"
            )

    def test_manifest_loaders_reject_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classification = root / "classification.csv"
            with classification.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "class_id", "class_name"])
                writer.writeheader()
                writer.writerows([
                    {"sample_id": "same", "image_path": "a.jpg", "class_id": "a", "class_name": "a"},
                    {"sample_id": "same", "image_path": "b.jpg", "class_id": "a", "class_name": "a"},
                ])
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_classification_manifest(classification)

            retrieval = root / "retrieval.csv"
            with retrieval.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "caption_id", "caption"])
                writer.writeheader()
                writer.writerows([
                    {"image_id": "a", "image_path": "a.jpg", "caption_id": "same", "caption": "one"},
                    {"image_id": "a", "image_path": "a.jpg", "caption_id": "same", "caption": "two"},
                ])
            with self.assertRaisesRegex(ValueError, "duplicate caption_id"):
                load_retrieval_manifest(retrieval)


if __name__ == "__main__":
    unittest.main()
