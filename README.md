# Testing Checkpoint–Resolution Alignment in Frozen Dual-Embedding Vision–Language Checkpoints: Accuracy and Recorded-Protocol Costs

Code and compact results for the paper accepted at **DICTA 2026**.

**Author:** Trong Cuong Tran, The University of Sydney, Sydney, Australia.

The study evaluates frozen vision–language checkpoints on Oxford-IIIT-Pet
classification and Flickr30k retrieval. The canonical matrix contains 54
configurations: 52 dual-embedding configurations and two supplementary
PaliGemma class-likelihood diagnostics. Recorded latency and memory use
adapter-specific protocols and do not establish intrinsic cross-family
efficiency rankings.

## What is included

| Directory | Contents |
|---|---|
| `analysis/` | Statistical analysis, figure scripts, and numerical tests |
| `reproducibility/benchmark/` | Benchmark adapters, checkpoint registries, dataset builders, protocols, and tests |
| `results/canonical/` | Immutable 54-row results used by the paper |
| `results/historical/` | One immutable 50-row baseline needed by provenance and regression checks |
| `results/supporting/focused_run/` | Hashed image outcomes, grouped retrieval hits, configuration records, and timing observations |
| `results/expected/` | Published statistical outputs for reproduction checks |

The source images, captions, dataset manifests containing local paths, model
weights, access credentials, and cloud-account information are not included.
The manuscript is maintained separately so that this code release does not
distribute obsolete submission drafts or an unvalidated camera-ready PDF.

Historical filenames containing `mock_review` identify the focused internal
validation run. They are retained for provenance; they do not denote official
conference reviews.

## Experiment hardware and protocol

All 54 reported configurations used **one NVIDIA RTX 6000 Ada Generation GPU
with 48 GB VRAM**, rented through **RunPod**. The retained historical results
and the focused validation run used the same GPU model but different software
environments:

| Component | Retained historical environment | Focused validation environment |
|---|---|---|
| NVIDIA driver | 570.124.06 | 550.127.05 |
| PyTorch CUDA runtime | 12.1 | 12.4 |
| Python | 3.12.3 | 3.11.10 |
| PyTorch | 2.4.1 | 2.4.1+cu124 |

Both environments recorded torchvision 0.19.1, open_clip_torch 2.26.1 and
transformers 4.44.2. The [full environment record](reproducibility/recorded_environment.md)
includes the available CPU, RAM and operating-system details, and explicitly
marks information that was not retained.

| Evaluated adapter | Precision | Classification image batch | Retrieval image / text batch |
|---|---|---:|---:|
| OpenCLIP (CLIP, SigLIP, MobileCLIP) | FP16 | 64 | 64 / 256, where evaluated |
| BLIP ITC dual embedding | FP16 | 32 | 32 / 128 |
| PaliGemma class-likelihood scoring | BF16 | 1 | Not evaluated |

OpenCLIP and BLIP used two warm-up batches and three timed repeats.
PaliGemma used one timed repeat, with two warm-up batches at 224 pixels and
five at 448 pixels. The separate cached-input batch-sensitivity experiment
compared batches 1 and 32 under its own recorded timing protocol.

Reported primary latency is `1000 / throughput_examples_s` in milliseconds
per image under the recorded batch protocol; it is not batch-one response
latency. Memory is peak allocated GPU memory. Adapter timing boundaries differ,
so these costs do not provide a shared cross-family efficiency ranking.

For reproduction, the package also records the
[checkpoint registries and resolution/prompt/protocol settings](reproducibility/benchmark/configs/),
[dataset preparation and test splits](reproducibility/README.md#dataset-preparation),
[statistical analysis commands](analysis/README.md), and
[retained outcomes, provenance and checksums](results/README.md).
Seeds and resampling counts are recorded in the analysis scripts and their
outputs. The focused run repeated six BLIP Pet cells and all 24 Flickr30k
cells; it did not independently repeat every historical Pet configuration.

## Reproduce the analysis without a GPU

Use Python 3.11. Create an isolated environment and install the
analysis dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 scripts/verify_release.py
```

The verification command runs the numerical and protocol tests, regenerates
the 54-row analysis and figures, replays the recorded bootstrap/permutation
and batch analyses, and checks them against the supplied expected results.
It also verifies the additional exploratory 336-versus-384 BLIP Pet contrast.
It downloads no models or datasets and runs no GPU inference. Derived outputs
are written to ignored `analysis/processed/` and verification logs to `build/`.

The original result CSVs are never overwritten. See
[analysis/README.md](analysis/README.md) for individual commands and statistical
interpretation, and [results/README.md](results/README.md) for the retained data.

## Run new benchmark measurements

Read [reproducibility/README.md](reproducibility/README.md) for the recorded
environment, dataset preparation, checkpoint access, and exact command
generation. CPU-only protocol checks are:

```sh
python3 reproducibility/benchmark/scripts/run_recorded_protocol.py --dry-run
python3 reproducibility/benchmark/scripts/run_mock_review_protocol.py --dry-run
```

The historical protocol has 50 cells. The focused protocol evaluates six BLIP
Pet cells, 24 Flickr30k cells, and 24 cached-input batch-timing cells. Their
merged paper matrix has 54 configurations. A dry run validates configuration
and command construction; it does not constitute a new GPU replication.

## Scope and limitations

- Reproducing saved-output statistics is distinct from repeating model inference.
- The focused GPU run did not repeat all historical Pet configurations or establish cross-hardware timing generality.
- The two PaliGemma cells use different released resolution-specific checkpoints and do not validate a frozen-weight resolution sweep on generative tasks.
- The extra BLIP Pet comparison is exploratory, with pointwise intervals and no correction for choosing the comparison after examining the sweeps.
- Historical Pet intervals are retained from the recorded summary; original per-image outcomes for every historical checkpoint are not included.

## Citation and licensing

Author and title metadata are provided in [CITATION.cff](CITATION.cff).
Stable code snapshot: [v1.0.2](https://github.com/cuongisme04/checkpoint-resolution-alignment/tree/v1.0.2).
Proceedings DOI and publication details will be added when available.

No project-wide license has been assigned to this repository. Third-party
dependencies retain their own licenses. Public availability is not a grant
of an open-source license.
