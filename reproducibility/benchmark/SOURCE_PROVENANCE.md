# Benchmark source provenance

The original adapters, shared modules, checkpoint registries, dataset builders
and PaliGemma parser test were recovered from the experiment workspace on
14 July 2026. `RECOVERED_SOURCE_SHA256.txt` preserves the checksums of that
initial recovery. It is a historical record, not a checksum list for the
current source.

Commit `b973b6b` subsequently added path-free outcome retention and batch
measurement support, changing `manifests.py`, `measure.py`, the four
OpenCLIP/BLIP benchmark adapters, and two classification-manifest builders.
Those eight files therefore differ from their initial recovery checksums.
The release uses their later versions from development commit
`ad7fa1d4890ec687d1a2fecf868b430fba110407`.

Check the actual release source with:

```sh
shasum -a 256 -c reproducibility/benchmark/SOURCE_SHA256.txt
```

`SOURCE_SHA256.txt` covers the current benchmark Python files, configurations
and requirements. The release's protocol test points to the relocated,
byte-identical 50-row baseline. No benchmark adapter logic was changed during
release packaging.

Protocol runners and their tests were added during manuscript development.
The focused GPU protocol was executed; a full independent rerun of every
historical configuration and a cross-hardware timing replication were not
performed. Historical orchestration scripts and handover notes are omitted;
their relevant settings are represented by the retained executable protocols,
environment record and sanitized run manifest.
