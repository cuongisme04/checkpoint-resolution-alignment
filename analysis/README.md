# Reproduce the paper's numerical analysis

Run commands from the package root. The analysis requires NumPy and
Matplotlib; see `requirements-analysis.txt`. No GPU is needed for these steps.

## Complete verification

```sh
python3 scripts/verify_release.py
```

This runs existing numerical/protocol tests, both protocol dry runs, the
statistical replay, the exploratory BLIP top-five diagnostic, and the three
camera-ready figure scripts. It compares
the replayed results with `results/expected/`, using exact labels and a
1e-12 tolerance for floating-point values, and confirms that input files did
not change. Logs and a JSON report are written to `build/verification/`.

[GitHub Actions](https://github.com/cuongisme04/checkpoint-resolution-alignment/actions/workflows/verify.yml)
runs the same command on Ubuntu 24.04 with Python 3.11 for pushes to `main`,
new version tags, pull requests, and manual runs. Reports and step logs are
attached to each run for 30 days. This checks analysis replay and protocol
construction; it does not download datasets or models or repeat GPU inference.

## Evidence guide

The table maps the study's numerical evidence to retained inputs and analysis
scripts. Linked outputs are checked-in reference results. Running the commands
below writes regenerated outputs under `analysis/processed/`, leaving those
references unchanged. All bootstrap and permutation analyses listed here use
10,000 replicates; seeds refer to statistical replay, not GPU inference.
In the contrast formulas, `yR` is performance at resolution `R` as a fraction
from 0 to 1 (Pet accuracy or mean bidirectional retrieval Recall@1).

| Evidence and retained input | Script and output | Resampling | Interpretation limit |
|---|---|---|---|
| Accuracy, operational costs and frontiers: [54-row canonical summary](../results/canonical/dicta_mock_review_54_with_ci.csv) | [regenerate_results.py](regenerate_results.py) → `analysis/processed/summary_with_tau_and_pareto.csv`, tables and figure trace | Deterministic transformation; no resampling or seed | Frontiers use point estimates. Adapter timing boundaries differ; costs are not intrinsic cross-family efficiency rankings. |
| Six focused BLIP Pet configurations: [image outcomes](../results/supporting/focused_run/classification/compact_outcomes.csv) | [mock_review_analysis.py](mock_review_analysis.py) → [classification_bootstrap.csv](../results/expected/focused/classification_bootstrap.csv) | Individual images; seed `20260714` | Test-sample uncertainty for these six cells. Other historical Pet intervals remain retained summaries without full per-image outcomes. |
| All 24 Flickr30k configurations: [grouped retrieval outcomes](../results/supporting/focused_run/retrieval/grouped_outcomes.csv) | [mock_review_analysis.py](mock_review_analysis.py) → [retrieval_grouped_bootstrap.csv](../results/expected/focused/retrieval_grouped_bootstrap.csv) | Image groups, each retaining its five captions; seed `20260714` | Resamples saved retrieval hits; does not rerun inference or reconstruct retrieval candidate sets in each replicate. |
| BLIP 384-pixel anchor contrast, `y384 − (y336 + y448) / 2`: the focused Pet and retrieval outcomes above | [mock_review_analysis.py](mock_review_analysis.py) → [blip_384_contrasts.csv](../results/expected/focused/blip_384_contrasts.csv) | Paired images for Pet; paired image groups for retrieval; seed `20260714` | A contrast among sampled resolutions, not proof of an optimum between them or a causal explanation. |
| Fixed-reference mismatch–regret association: canonical summary above | [mock_review_analysis.py](mock_review_analysis.py) → [per-row regret](../results/expected/focused/normalized_regret_rows.csv), [sweep correlations](../results/expected/focused/mismatch_sweep_correlations.csv), [summary](../results/expected/focused/mismatch_summary.json) | Median-correlation interval: checkpoint clusters, seed `20260714`. Conditional test: permute regret within each eligible sweep, seed `20260715` | 13 eligible sweeps from seven checkpoints; fixed references and sampled grids. The checkpoint interval differs from test-sample intervals; the permutation test does not establish causality. |
| Exploratory BLIP Pet comparison, `100 × (y336 − y384)` percentage points: focused Pet outcomes above | [camera_ready_blip_pairing.py](camera_ready_blip_pairing.py) → [blip_pet_336_vs_384.csv](../results/expected/blip_pet_336_vs_384.csv) | 3,669 matched image pairs per model; seed `20260915`, fresh PCG64 generator per model | Pointwise 95% percentile intervals; comparison chosen after inspecting sweeps, without selection or multiplicity correction. |
| Batch sensitivity: [6,000 timing observations](../results/supporting/focused_run/batch_sensitivity/raw_batch_timings.csv) | [batch_sensitivity_analysis.py](batch_sensitivity_analysis.py) → [configuration summaries](../results/expected/focused/batch_configuration_summary.csv), [batch comparisons](../results/expected/focused/batch_pair_comparisons.csv) | Deterministic summaries; no resampling or seed | Cached-input image-encoder timing at batches 1 and 32. Timing percentiles describe observations, not confidence intervals or full-adapter costs. |
| Final exploratory BLIP Pet top-one/top-five diagnostic: focused Pet outcomes and configuration index above | [blip_pet_top5_diagnostic.py](blip_pet_top5_diagnostic.py) → [blip_pet_top5_diagnostic.json](../results/expected/blip_pet_top5_diagnostic.json) | Descriptive counts on 3,669 matched image pairs per model; no resampling or seed | All 208 base and 167 large top-one correctness changes retain the true breed in the top five at both resolutions. This does not identify competing labels, score margins, statistical equivalence, or a causal mechanism. |

Use the explicit canonical-summary argument in the focused analysis command
below; that script's standalone default still points to the historical
50-row baseline. The complete verifier supplies the canonical input.

Input scope and checksums are documented in [results/README.md](../results/README.md).
The focused [run manifest](../results/supporting/focused_run/run_manifest_outcomes.json)
and [classification](../results/supporting/focused_run/classification/configuration_index.csv)
and [retrieval](../results/supporting/focused_run/retrieval/configuration_index.csv)
configuration indexes retain protocol and input provenance.
[Source provenance](../reproducibility/benchmark/SOURCE_PROVENANCE.md) distinguishes
the original recovered source from the packaged benchmark source;
[checkpoint version limitations](../reproducibility/README.md#checkpoint-versions-and-provenance)
explain what the release does not pin.

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

## Final exploratory BLIP top-five diagnostic

```sh
python3 analysis/blip_pet_top5_diagnostic.py
```

This dependency-free script replays the descriptive audit added to the final
discussion and supplement. It validates the retained input hashes, image-ID
pairing, matching protocols, logical consistency of top-one/top-five outcomes,
and agreement with canonical accuracies. It writes
`analysis/processed/camera_ready/blip_pet_top5_diagnostic.json`.
The checked-in expected JSON preserves the counts and interpretation from the
11 October 2026 manuscript audit. Tests also check row-order independence,
contrast direction, invalid pairs, and cases where a changed top-one outcome
does not retain the true class in the top five.

The 336-minus-384 top-five differences are -0.27 percentage points for base
and +0.05 for large. These are descriptive observations selected after seeing
the sweeps. Image resampling, visual-token count, and positional-grid
interpolation vary together; the cause of the exception remains unresolved.
No model inference is performed. This script and its expected output were
added after the paper-cited `v1.0.4`; the retained inputs and canonical results
are unchanged.

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
