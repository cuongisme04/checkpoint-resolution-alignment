# Expected output schema

## Historical protocol outputs

The recorded 50-cell runner writes under the selected `--output-root`:

```text
classification/benchmark_rows.csv
classification/predictions.csv
classification/failures.csv        # only if a configuration errors
retrieval/benchmark_rows.csv
retrieval/failures.csv              # only if a configuration errors
run_manifest.json
```

Successful completion of the recorded protocol produces 28 classification rows
and 22 retrieval rows. A row is uniquely identified operationally by:

```text
(model_id, task, dataset, resolution, batch_size, text_batch_size, precision)
```

`text_batch_size` is populated only for retrieval. Core summary fields include:

- model/checkpoint metadata: `model_id`, `pretrained`, architecture metadata,
  resolution, parameter count, patch size, and visual-token proxy;
- protocol metadata: image `batch_size`, retrieval `text_batch_size`, precision,
  warmup count, repeat count, manifest, device, and `gpu_name`;
- performance: classification top-1/top-5 or both retrieval directions at
  Recall@1/5/10 plus mean Recall@1;
- operational cost: throughput, peak VRAM, and batch latency mean/std/p50/p95.

The `latency_*_ms` values are per-batch measurements. Downstream analysis
computes amortised per-image cost as:

```text
tau_img_ms = 1000 / throughput_examples_s
```

The large historical classification prediction file contains per-image
correctness used by the descriptive bootstrap. It is excluded from Git;
the retained focused-run outcomes described below are included separately.

## Focused protocol outputs

The focused runner writes under its selected `--output-root`:

```text
classification/benchmark_rows.csv
classification/compact_outcomes.csv
classification/configuration_index.csv
retrieval/benchmark_rows.csv
retrieval/grouped_outcomes.csv
retrieval/configuration_index.csv
batch_sensitivity/raw_batch_timings.csv
run_manifest.json
```

Successful completion produces:

- **6 classification summaries** and **22,014 compact Pet outcomes**
  (6 configurations × 3,669 test images);
- **24 retrieval summaries** and **24,000 grouped Flickr30k outcomes**
  (24 configurations × 1,000 images, with hit counts for each image's five captions);
- **6,000 batch-timing observations** across 24 cached-input timing configurations;
- separate classification and retrieval configuration indexes, plus a run
  manifest recording commands and protocol, registry, and dataset-manifest hashes.

Failure files may also be written if configurations error. Compact outcomes
use hashed sample identifiers and correctness/hit counts rather than image
paths or caption text. The curated release retains these outcomes, indexes,
and timings under `results/supporting/focused_run/`, with the sanitized manifest
named `run_manifest_outcomes.json`; see the [retained-data guide](../results/README.md).

## Canonical and processed outputs

The published canonical summary,
`results/canonical/dicta_mock_review_54_with_ci.csv`, contains **54 configurations**:
30 Pet classification rows and 24 Flickr30k retrieval rows. The retained
historical baseline, `results/historical/summary_50.csv`, contains **50 configurations**:
28 classification rows and 22 retrieval rows. These are distinct inputs; the
historical runner alone does not produce the merged 54-row paper matrix.
Focused results replace overlapping historical rows and add the four exact-384
BLIP configurations. The separate batch-sensitivity cells are not part of the
54-row accuracy/retrieval matrix.

The canonical summary retains confidence-interval and legacy Pareto fields.
`analysis/regenerate_results.py` validates it and writes:

- `analysis/processed/summary_with_tau_and_pareto.csv`;
- task-specific Pareto CSVs;
- `cross_task_joint_frontier_intersection.csv`;
- an input checksum/analysis manifest;
- a figure/table trace manifest;
- generated LaTeX table fragments;
- vector PDF figures.

The focused alignment analysis additionally writes per-sweep mismatch and
regret rows, sweep correlations, a JSON summary containing clustered-bootstrap
and within-sweep permutation-null results, and the generated compact evidence
table at
`analysis/processed/mock_review/tables/alignment_sweeps.tex`.

Pareto membership is deterministic from point estimates. Confidence intervals
are displayed descriptive summaries and do not determine frontier membership.
