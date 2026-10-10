# Reproduce the paper's numerical analysis

Run commands from the package root. The analysis requires NumPy and
Matplotlib; see `requirements-analysis.txt`. No GPU is needed for these steps.

## Complete verification

```sh
python3 scripts/verify_release.py
```

This runs existing numerical/protocol tests, both protocol dry runs, the
statistical replay, and the three camera-ready figure scripts. It compares
the replayed results with `results/expected/`, using exact labels and a
1e-12 tolerance for floating-point values, and confirms that input files did
not change. Logs and a JSON report are written to `build/verification/`.

## Canonical table and frontier analysis

```sh
python3 analysis/regenerate_results.py
```

The default is `results/canonical/dicta_mock_review_54_with_ci.csv`.
The script checks the immutable checksum, 54 configurations, task counts,
batches, precision, throughput, GPU name, and operational keys. It computes
`tau_img_ms = 1000 / throughput_examples_s`, regenerates compact CSVs and
LaTeX tables, and creates diagnostic figures under `analysis/processed/`.
The historical 50-row baseline remains in `results/historical/summary_50.csv`
for regression testing and reconstruction of the original protocol.

## Focused bootstrap, alignment and batch analyses

```sh
python3 analysis/mock_review_analysis.py \
  --summary results/canonical/dicta_mock_review_54_with_ci.csv \
  --classification-outcomes results/supporting/focused_run/classification/compact_outcomes.csv \
  --retrieval-outcomes results/supporting/focused_run/retrieval/grouped_outcomes.csv \
  --output-dir analysis/processed/mock_review \
  --require-complete

python3 analysis/batch_sensitivity_analysis.py \
  --input results/supporting/focused_run/batch_sensitivity/raw_batch_timings.csv \
  --output-dir analysis/processed/mock_review

python3 analysis/camera_ready_blip_pairing.py
```

These reproduce the six focused Pet bootstrap summaries, 24 grouped
retrieval summaries, paired BLIP anchor contrasts, mismatch/regret analysis,
the conditional permutation test, batch comparisons, and the exploratory
336-versus-384 Pet contrast. Seeds and replication counts are recorded in the
outputs. Repeated resampling of saved outcomes is not a new model experiment.

## Camera-ready figures

```sh
python3 analysis/render_camera_ready_cost_figure.py
python3 analysis/render_camera_ready_resolution_figure.py
python3 analysis/render_camera_ready_proxy_figure.py
```

Outputs are under `analysis/processed/camera_ready/figures/`. Each script
writes a vector PDF, a render for inspection, and a trace connecting plotted
points to unchanged canonical rows. The cost and resolution figures use the
52 primary dual-embedding rows. The secondary parameter-token proxy retains
all 54 rows and is not interpreted as FLOPs or a comparable cross-family cost.

## Interpretation

Latency is amortised operational time per image under the recorded batch and
adapter boundary, not batch-one request latency. Recomputed frontier flags
remain descriptive diagnostics. The fixed-reference association is
conditional on these checkpoints and sampled resolutions. Neither a positive
association nor a pointwise interval establishes a causal mechanism, an
unsampled optimum, or generality to untested generative tasks.

`merge_mock_review_results.py` and `finalize_mock_review_summary.py` preserve
the strict integration procedure for future GPU reruns. Their inputs are raw
benchmark summaries and validated interval files; they must not overwrite
the released canonical CSV. Run each script with `--help` for arguments.
