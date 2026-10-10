# Expected output schema

## Raw benchmark outputs

The recorded runner writes under the selected `--output-root`:

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

The large classification prediction file contains per-image correctness used
by the descriptive bootstrap. It is deliberately excluded from Git.

## Canonical and processed outputs

The published canonical summary contains 50 rows with confidence-interval and
legacy Pareto fields. `analysis/regenerate_results.py` validates it and writes:

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
