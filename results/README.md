# Retained scientific data

The release keeps numerical evidence, including exceptions and limitations.
It excludes working conversations and discarded document drafts.

## Immutable summaries

- `canonical/dicta_mock_review_54_with_ci.csv`: all 54 paper configurations; SHA-256 `4b20a406983ff3d43dcba2cd1d7319a830cc6cd1ab4340221f65bad71cdcbbf7`.
- `historical/summary_50.csv`: the historical 50-row baseline used to validate the original recorded protocol and strict result merger; SHA-256 `2d2fa4103fb557776d9ad5d78f9d0ed030242ba34cde581f8051409bc1e17ea1`.

The historical file has been relocated from the working repository's archive
directory. Its bytes are unchanged. Original column names, checkpoint names,
run labels, and original run commit hashes remain intact for provenance.

## Supporting inputs

`supporting/focused_run/` contains:

- 22,014 path-free Pet outcome rows: six configurations × 3,669 image identifiers.
- 24,000 grouped Flickr30k rows: 24 configurations × 1,000 image identifiers, each associated with five captions through hit counts.
- 6,000 batch-timing observations from 24 configurations.
- Separate classification and retrieval configuration indexes.
- The sanitized recorded run manifest with protocol, registry, and dataset-manifest hashes.

Sample identifiers are hashed. These files contain correctness/hit counts
and timing measurements, not images, caption text, predicted class text,
dataset download credentials, or absolute image paths. Retained run and
environment labels describe the experiment rather than paid cloud resource IDs.

The saved outcomes support replay of the reported focused-run intervals and
paired contrasts. They do not supply per-image outputs for every historical
Pet model. The canonical file retains those original summary intervals.

`expected/` contains the statistical outputs used for verification. Generated
plots, tables, and analysis summaries are recreated under `analysis/processed/`.

`expected/blip_pet_top5_diagnostic.json` preserves the final manuscript's
11 October 2026 exploratory diagnostic, published with its replay script
after `v1.0.4`. It uses the same retained Pet outcomes and configuration index;
it adds no GPU measurement and does not change the canonical matrix. Its
hashes identify the three inputs, and its limitations distinguish descriptive
ranking changes from an unresolved causal explanation.

## Provenance

The package uses the benchmark and numerical analysis from the working branch
at commit `ad7fa1d4890ec687d1a2fecf868b430fba110407`, with the October
camera-ready figure scripts and exploratory paired-analysis script. Packaging
changes relocate input files and add release documentation and verification;
they do not change the recorded measurements or benchmark adapter logic.

The original focused run records code commit
`45f20f1af8459ecbd0ba760a9e80f7418d69eabd`. These identifiers document the
private development history; a clean public snapshot need not expose that
history. Recovered adapter checksums remain available under
`reproducibility/benchmark/RECOVERED_SOURCE_SHA256.txt`.
