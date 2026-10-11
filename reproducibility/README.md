# Benchmark reproducibility package

This directory separates two reproducibility targets:

1. **Published-result analysis:** reproducible from the tracked 54-row canonical
   summary with no model downloads or GPU; the locked historical 50-row input
   remains preserved for provenance.
2. **GPU benchmark rerun:** source, checkpoint identifiers, dataset builders,
   and both the recorded 50-cell and focused validation protocols are provided.
   The focused run was executed from a clean checkout; it is not a full
   cross-pod repeat of every historical Pet cell.

The first target is validated deterministically. The second remains
**partially reproducible** because only the focused subset was independently
executed and timings were not replicated across pods.

## Package map

- `benchmark/charlie_vlm/`: recovered shared benchmark utilities.
- `benchmark/scripts/bench_*.py`: recovered OpenCLIP, BLIP, and PaliGemma
  adapters used for the reported sweep.
- `benchmark/configs/models.json`: exact OpenCLIP checkpoint identifiers.
- `benchmark/configs/hf_models.json`: exact BLIP and PaliGemma Hub identifiers.
- `benchmark/configs/recorded_protocol.json`: reconstructed executable form of
  the settings recorded in the historical 50-row baseline.
- `benchmark/scripts/run_recorded_protocol.py`: validates, dry-runs, or executes
  those settings without silently appending to an existing result CSV.
- `benchmark/configs/mock_review_protocol.json`: focused BLIP-384, grouped
  Flickr30k, and batch-sensitivity protocol prepared for the post-review run.
- `benchmark/scripts/run_mock_review_protocol.py`: validates all 6 Pet, 24
  Flickr30k, and 24 cached-input timing cells without loading models in dry-run
  mode.
- `../analysis/mock_review_analysis.py`: grouped bootstrap, paired BLIP-384
  contrast, and checkpoint--resolution mismatch analysis.
- `../analysis/merge_mock_review_results.py`: strict 50-to-54-row integration
  of the validated focused summaries without modifying the locked baseline.
- `../analysis/finalize_mock_review_summary.py`: strict interval attachment for
  the expanded canonical summary.
- `../analysis/batch_sensitivity_analysis.py`: validates and summarises the
  6,000 retained microbenchmark timings.
- `../analysis/regenerate_results.py`: authoritative published-result analysis,
  Pareto, table, and figure entry point.
- `recorded_environment.md`: captured RunPod hardware and software details.
- `OUTPUT_SCHEMA.md`: expected raw and processed files and field semantics.

## CPU-only validation

From the repository root (`checkpoint-resolution-alignment/`), with the
[analysis dependencies](../README.md#reproduce-the-analysis-without-a-gpu) installed:

```sh
python3 -m unittest discover \
  -s reproducibility/benchmark/tests \
  -t reproducibility/benchmark -v
python3 reproducibility/benchmark/scripts/run_recorded_protocol.py --dry-run
python3 reproducibility/benchmark/scripts/run_mock_review_protocol.py --dry-run
python3 analysis/regenerate_results.py
python3 -m unittest \
  analysis/test_regenerate_results.py \
  analysis/test_mock_review_analysis.py -v
```

The dry run must report 28 classification cells, 22 retrieval cells, and 15
command groups. It does not import PyTorch, download a dataset, or load a model.
The mock-review dry run must additionally report 6 Pet BLIP validation cells,
24 Flickr30k grouped-outcome cells, and 24 batch-sensitivity cells.
The recorded environments and stop conditions are described below. The
release includes sanitized focused-run inputs under `../results/supporting/`;
run `python3 scripts/verify_release.py` from the repository root for a complete
CPU replay against the expected statistics.

## Focused post-review protocol

The new protocol deliberately avoids a batch-one full-dataset evaluation.
Batch sizes one and 32 are tested only on a representative 24-cell cached-CUDA
microbenchmark. Its common timing boundary is the image encoder plus
projection/normalisation; preprocessing, host-to-device transfer, text
encoding, similarity construction, and dataset metrics are excluded. These
measurements assess batch sensitivity and must not replace `tau_img_ms` in the
primary operational frontier.

The revised classification output is path-free and retains one hashed sample
ID with top-1/top-5 correctness per configuration. Retrieval retains one
hashed row per image, including directional hit counts for that image's five
captions. A separate configuration index records checkpoint, manifest
checksum, scoring path, batch sizes, precision, and timing settings. The run
manifest sanitizes command paths and retains registry and input checksums.

From the repository root, inspect the exact commands before a pod is provisioned:

```sh
python3 reproducibility/benchmark/scripts/run_mock_review_protocol.py --dry-run
```

## GPU environment

Reported rows come from rented RunPod pods, each with one NVIDIA RTX 6000 Ada
Generation GPU (48 GB). The focused run used a separate recorded software
environment from the retained historical Pet rows. Start the following block
from the repository root to create a compatible environment. It enters
`reproducibility/benchmark/`; stay in that directory for dataset preparation
and the recorded sweep below.

```sh
cd reproducibility/benchmark
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The requirements reproduce the recovered pinned environment specification. The
live setup record separately confirmed the PyTorch, torchvision, OpenCLIP, and
Transformers versions; the remaining pins were not independently captured with
`pip freeze`. CUDA compatibility still depends on the selected pod image and
driver. The original container image identifier was not recorded.

PaliGemma is gated on Hugging Face. Authenticate outside the repository using
the current Hugging Face CLI workflow. Never place a token in a command file,
configuration file, `.env`, or Git history. `HF_HOME` may be pointed to a
persistent path outside the repository to avoid repeated downloads.

## Dataset preparation

Dataset sources, author credits and usage terms are documented in the
[main README](../README.md#dataset-sources-attribution-and-terms). The code's
MIT license does not apply to downloaded dataset images, captions or model weights.

The benchmark scripts create absolute image paths in local manifest files.
Those manifests and downloaded images are ignored by Git.

Run the commands in this section from `reproducibility/benchmark/`. If you
followed the GPU environment block, you are already there; otherwise, enter
that directory from the repository root and activate the benchmark environment
before continuing.

Oxford-IIIT Pet uses the official 3,669-image, 37-class test split distributed
through `torchvision==0.19.1`:

```sh
python3 scripts/make_oxford_pet_manifest.py \
  --data-root datasets/oxford_pet \
  --split test \
  --output datasets/manifests/pet_test.csv
```

Flickr30k uses the 1,000-image Karpathy test subset exposed by
`nlphuji/flickr30k`. The builder filters the dataset's internal `split` field
to `test`, materialises the first 1,000 test images, and emits five caption rows
per image:

```sh
python3 scripts/make_flickr30k_manifest.py \
  --data-root datasets/flickr30k \
  --output datasets/manifests/flickr30k_test1k.csv \
  --max-images 1000
```

The exact Hugging Face dataset revision was not captured in the original run.
This is a remaining reproducibility limitation; verify the resulting counts
before an expensive rerun.

## Recorded 50-cell sweep

From `reproducibility/benchmark/`, inspect every command first:

```sh
python3 scripts/run_recorded_protocol.py --dry-run
```

Then execute from the same directory on a suitable CUDA host:

```sh
python3 scripts/run_recorded_protocol.py \
  --output-root outputs/recorded_50_cell
```

The runner refuses to append if benchmark or prediction files already exist in
that output root. Select a new output directory for a new replication. It also
writes an ignored `run_manifest.json`; review that file for local paths before
sharing it.

The protocol records the actual settings rather than claiming a uniform run:

- OpenCLIP classification: image batch 64, FP16, 2 warmup batches, 3 repeats.
- BLIP classification: image batch 32, FP16, 2 warmup batches, 3 repeats.
- OpenCLIP retrieval: image batch 64, text batch 256, FP16, 2 warmups, 3 repeats.
- BLIP retrieval: image batch 32, text batch 128, FP16, 2 warmups, 3 repeats.
- PaliGemma-224: batch 1, BF16, 2 warmups, 1 repeat, prompt chunk 16.
- PaliGemma-448: batch 1, BF16, 5 warmups, 1 repeat, prompt chunk 4.

Raw prediction files generated by new historical-protocol runs may contain
local image paths and are not included. The released focused-run outcomes use
hashed image identifiers and retain only correctness/hit counts. They support
replay of the focused intervals and paired contrasts; historical Pet intervals
for the other checkpoints remain retained summaries.

## Published-result analysis

The authoritative tracked input is:

```text
results/canonical/dicta_mock_review_54_with_ci.csv
```

Its SHA-256 is
`4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7`.
If continuing from the benchmark-directory commands above, run `cd ../..`
first to return to the repository root. Then run:

```sh
python3 analysis/regenerate_results.py \
  --input results/canonical/dicta_mock_review_54_with_ci.csv
```

The pipeline validates all 54 configurations, computes
`tau_img_ms = 1000 / throughput_examples_s`, recomputes point-estimate Pareto
frontiers, and regenerates processed CSVs, LaTeX tables, and vector figures.
It never modifies the canonical input.

The historical 50-row source remains locked at
`results/historical/summary_50.csv`
with SHA-256
`2d2fa4103fb557776d9ad5d78f9d0ed030242ba34cde581f8051409bc1e17ea1`.

`tau_img_ms` is amortised operational time per image under the recorded batch
protocol. It is not batch-one response latency. Per-batch latency fields remain
secondary measurements and must always be interpreted with their batch sizes.

## Scoring-path interpretation

The recovered registries retain historical metadata so reruns reproduce the
raw output schema. Interpret them with these corrections:

- BLIP was evaluated through ITC dual-embedding scoring; the optional ITM
  reranker was disabled (`itm_rerank_topk = 0`). The legacy
  `architecture_family` label describes the full checkpoint, not the evaluated
  scoring path.
- The BLIP registry's legacy `native_resolution = 384` key denotes the
  downstream COCO checkpoint's default image-processing configuration. It does
  not mean BLIP was pretrained natively at 384; BLIP pretraining used 224.
  Every evaluated resolution is passed explicitly by the recorded protocol.
- SigLIP versus OpenCLIP is a checkpoint comparison, not a controlled loss
  ablation.
- PaliGemma uses adapted summed class-prompt log-likelihood scoring. It is not
  directly equivalent to contrastive zero-shot classification.

The processed summary's `evaluated_scoring_path` field is authoritative for
analysis and figures.

## Data and privacy policy

Keep datasets, manifests containing absolute paths, raw image predictions,
checkpoints, model caches, access tokens, cloud resource identifiers, billing
details, and new run logs outside the release. The curated path-free outcomes
and timings under `results/supporting/` are intentional scientific inputs;
see `results/README.md` for their scope and limitations.
