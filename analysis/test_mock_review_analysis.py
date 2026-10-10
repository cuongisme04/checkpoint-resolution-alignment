from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("mock_review_analysis.py")
SPEC = importlib.util.spec_from_file_location("mock_review_analysis", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def retrieval_rows(prefix: str, count: int = 20):
    rows = []
    for index in range(count):
        rows.append(
            {
                "configuration_id": prefix,
                "sample_id": f"sample-{index:03d}",
                "caption_count": "5",
                "image_to_text_hit_at_1": str(int(index % 3 != 0)),
                "text_to_image_hits_at_1": str(index % 6),
            }
        )
    return rows


def classification_rows(prefix: str, count: int = 20):
    return [
        {
            "configuration_id": prefix,
            "sample_id": f"sample-{index:03d}",
            "correct_top1": str(int(index % 4 != 0)),
        }
        for index in range(count)
    ]


class MockReviewAnalysisTest(unittest.TestCase):
    def test_mismatch_is_zero_at_reference_and_symmetric(self):
        rows = [
            {"model_id": "clip_vit_b16_openai", "task": "t", "dataset": "d", "performance_metric": "p", "performance_value": "0.7", "resolution": "112", "native_resolution": "224"},
            {"model_id": "clip_vit_b16_openai", "task": "t", "dataset": "d", "performance_metric": "p", "performance_value": "0.9", "resolution": "224", "native_resolution": "224"},
            {"model_id": "clip_vit_b16_openai", "task": "t", "dataset": "d", "performance_metric": "p", "performance_value": "0.8", "resolution": "448", "native_resolution": "224"},
        ]
        detail, sweeps = ANALYSIS.mismatch_and_regret_rows(rows)
        by_resolution = {row["evaluation_resolution"]: row for row in detail}
        self.assertEqual(float(by_resolution[224]["resolution_mismatch_log2"]), 0.0)
        self.assertEqual(float(by_resolution[112]["resolution_mismatch_log2"]), 1.0)
        self.assertEqual(float(by_resolution[448]["resolution_mismatch_log2"]), 1.0)
        self.assertEqual(float(by_resolution[224]["normalized_regret_percent"]), 0.0)
        self.assertEqual(sweeps[0]["aggregation_status"], "included")
        self.assertEqual(sweeps[0]["closest_to_reference_has_best_point_estimate"], 1)

    def test_short_sweep_is_visible_but_excluded(self):
        rows = [
            {"model_id": "clip_vit_b16_openai", "task": "t", "dataset": "d", "performance_metric": "p", "performance_value": "0.7", "resolution": "224", "native_resolution": "224"},
            {"model_id": "clip_vit_b16_openai", "task": "t", "dataset": "d", "performance_metric": "p", "performance_value": "0.8", "resolution": "448", "native_resolution": "224"},
        ]
        detail, sweeps = ANALYSIS.mismatch_and_regret_rows(rows)
        self.assertEqual(len(detail), 2)
        self.assertEqual(sweeps[0]["aggregation_status"], "excluded_fewer_than_three")
        self.assertEqual(sweeps[0]["reference_resolution"], 224)
        self.assertEqual(sweeps[0]["peak_resolution"], "448")

    def test_permutation_null_is_deterministic(self):
        rows = [
            {"model_id": "clip_vit_b16_openai", "task": task, "dataset": task, "performance_metric": "p", "performance_value": value, "resolution": str(resolution), "native_resolution": "224"}
            for task in ("a", "b")
            for resolution, value in ((160, "0.7"), (224, "0.9"), (448, "0.6"))
        ]
        detail, sweeps = ANALYSIS.mismatch_and_regret_rows(rows)
        first = ANALYSIS.permutation_median_correlation_null(
            detail, sweeps, seed=11, repetitions=250
        )
        second = ANALYSIS.permutation_median_correlation_null(
            detail, sweeps, seed=11, repetitions=250
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["permutation_observed_median_spearman"], 1.0)
        self.assertGreater(first["permutation_one_sided_p"], 0.0)
        self.assertLessEqual(first["permutation_one_sided_p"], 1.0)

    def test_alignment_table_includes_short_sweep_reason(self):
        rows = [
            {"model_id": "clip_vit_b32_openai", "task": "image_text_retrieval", "dataset": "d", "performance_metric": "p", "performance_value": "0.7", "resolution": "160", "native_resolution": "224"},
            {"model_id": "clip_vit_b32_openai", "task": "image_text_retrieval", "dataset": "d", "performance_metric": "p", "performance_value": "0.8", "resolution": "224", "native_resolution": "224"},
        ]
        _, sweeps = ANALYSIS.mismatch_and_regret_rows(rows)
        rendered = ANALYSIS.render_alignment_sweep_table(sweeps)
        self.assertIn("CLIP B/32 & Flickr & 2 & 224 & 224 & --", rendered)
        self.assertIn("release config.", rendered)

    def test_grouped_bootstrap_is_deterministic_and_order_invariant(self):
        rows = retrieval_rows("a")
        first = ANALYSIS.grouped_retrieval_bootstrap(rows, seed=7, repetitions=500, chunk_size=50)
        second = ANALYSIS.grouped_retrieval_bootstrap(list(reversed(rows)), seed=7, repetitions=500, chunk_size=50)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci_lower_95"], first["performance_value"])
        self.assertGreaterEqual(first["ci_upper_95"], first["performance_value"])

    def test_classification_bootstrap_is_deterministic_and_order_invariant(self):
        rows = classification_rows("a")
        first = ANALYSIS.classification_bootstrap(
            rows, seed=7, repetitions=500, chunk_size=50
        )
        second = ANALYSIS.classification_bootstrap(
            list(reversed(rows)), seed=7, repetitions=500, chunk_size=50
        )
        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 20)
        self.assertAlmostEqual(first["performance_value"], 0.75)
        self.assertLessEqual(first["ci_lower_95"], first["performance_value"])
        self.assertGreaterEqual(first["ci_upper_95"], first["performance_value"])

    def test_identical_paired_configs_have_zero_contrast(self):
        rows = retrieval_rows("a")
        result = ANALYSIS.paired_blip_contrast(
            rows, rows, rows,
            task="image_text_retrieval",
            seed=9,
            repetitions=300,
            chunk_size=50,
        )
        self.assertAlmostEqual(result["contrast_384_vs_336_448_mean"], 0.0)
        self.assertAlmostEqual(result["ci_lower_95"], 0.0)
        self.assertAlmostEqual(result["ci_upper_95"], 0.0)

    def test_paired_configs_require_identical_sample_ids(self):
        first = retrieval_rows("a")
        second = retrieval_rows("b")[:-1]
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "identical sample-ID"):
            ANALYSIS.require_identical_sample_ids([first, second])

    def test_final_mode_rejects_missing_blip_384(self):
        summary = [
            {"model_id": model, "task": task, "resolution": str(resolution), "performance_value": "0.5"}
            for model in ANALYSIS.BLIP_MODELS
            for task in ("zeroshot_classification", "image_text_retrieval")
            for resolution in (336, 448)
        ]
        with self.assertRaisesRegex(ANALYSIS.AnalysisError, "missing BLIP rows"):
            ANALYSIS.blip_contrast_status(
                summary, {}, {}, seed=1, repetitions=10, require_complete=True
            )


if __name__ == "__main__":
    unittest.main()
