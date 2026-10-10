from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("batch_sensitivity_analysis.py")
SPEC = importlib.util.spec_from_file_location("batch_sensitivity_analysis", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def complete_rows() -> list[dict[str, str]]:
    rows = []
    for pair in range(12):
        for batch in (1, 32):
            for block in range(1, 6):
                for iteration in range(1, 51):
                    per_image = 4.0 / (2 if batch == 32 else 1)
                    rows.append({
                        "configuration_id": f"config-{pair}-{batch}",
                        "model_id": f"model-{pair // 2}",
                        "resolution": str(224 + pair),
                        "batch_size": str(batch),
                        "precision": "fp16",
                        "block": str(block),
                        "iteration": str(iteration),
                        "batch_latency_ms": str(per_image * batch),
                        "latency_ms_per_image": str(per_image),
                        "throughput_images_s": str(1000.0 / per_image),
                        "timing_scope": "cached CUDA tensor",
                        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
                        "peak_allocated_vram_gb": str(1.0 + batch / 100),
                        "peak_reserved_vram_gb": str(1.1 + batch / 100),
                    })
    return rows


class BatchSensitivityAnalysisTest(unittest.TestCase):
    def test_complete_protocol_produces_24_summaries_and_12_pairs(self):
        summaries, comparisons = ANALYSIS.analyse(complete_rows())
        self.assertEqual(len(summaries), 24)
        self.assertEqual(len(comparisons), 12)
        self.assertEqual({row["observation_count"] for row in summaries}, {250})
        self.assertEqual({row["block_count"] for row in summaries}, {5})
        self.assertEqual({row["batch32_throughput_speedup"] for row in comparisons}, {2.0})
        self.assertEqual({row["batch32_per_image_latency_ratio"] for row in comparisons}, {0.5})

    def test_incomplete_protocol_is_rejected(self):
        with self.assertRaisesRegex(ANALYSIS.BatchAnalysisError, "6,000 rows"):
            ANALYSIS.analyse(complete_rows()[:-1])


if __name__ == "__main__":
    unittest.main()
