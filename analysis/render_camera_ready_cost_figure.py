#!/usr/bin/env python3
"""Replot the locked 52 primary cells with adapter protocols in separate panels.

Run from any directory with Python and matplotlib available:
    python analysis/render_camera_ready_cost_figure.py

The canonical CSV, archived figures, and result tables are read-only. This
script changes presentation only: all performance values, throughput-derived
horizontal positions, and 95% confidence limits come from the original input.
No frontier membership is read or plotted, and no model execution is performed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True

from regenerate_results import (
    DEFAULT_INPUT,
    FAMILY_MARKERS,
    RESOLUTION_COLOURS,
    configure_publication_matplotlib,
    load_and_validate,
    plot_family,
    publication_legend_handles,
    record_sort_key,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/processed/camera_ready/figures"
PDF_PATH = OUTPUT / "performance_vs_recorded_adapter_costs.pdf"
TRACE_PATH = OUTPUT / "performance_vs_recorded_adapter_costs.trace.json"
TASKS = ("zeroshot_classification", "image_text_retrieval")
ADAPTERS = ("OpenCLIP", "BLIP")


def adapter(record):
    return "BLIP" if record.model_id.startswith("blip") else "OpenCLIP"


def main() -> None:
    records, _, input_hash = load_and_validate(DEFAULT_INPUT)
    primary = [r for r in records if not r.model_id.startswith("paligemma")]
    expected_counts = {
        ("zeroshot_classification", "OpenCLIP"): 18,
        ("zeroshot_classification", "BLIP"): 10,
        ("image_text_retrieval", "OpenCLIP"): 14,
        ("image_text_retrieval", "BLIP"): 10,
    }
    counts = Counter((r.task, adapter(r)) for r in primary)
    if dict(counts) != expected_counts or len(records) != 54:
        raise ValueError(f"Unexpected data coverage: {dict(counts)!r}")
    for record in primary:
        expected_batch = 32 if adapter(record) == "BLIP" else 64
        if record.numeric["batch_size"] != expected_batch:
            raise ValueError("Figure labels do not match the recorded image batch")

    configure_publication_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.55), sharey="row")
    fig.subplots_adjust(
        left=0.09, right=0.985, top=0.855, bottom=0.24,
        hspace=0.43, wspace=0.15,
    )
    fig.text(
        0.5, 0.99, "Performance under recorded adapter protocols",
        ha="center", va="top", fontsize=10, weight="bold",
    )
    fig.text(
        0.5, 0.949,
        "Batches and timer boundaries differ; no cross-family Pareto frontier.",
        ha="center", va="top", fontsize=8.3,
    )

    point_trace = []
    for row, task in enumerate(TASKS):
        for col, protocol in enumerate(ADAPTERS):
            ax = axes[row, col]
            selected = sorted(
                (r for r in primary if r.task == task and adapter(r) == protocol),
                key=record_sort_key,
            )
            for record in selected:
                numeric = record.numeric
                y = numeric["performance_value"] * Decimal(100)
                lower = numeric["ci_lower_95"] * Decimal(100)
                upper = numeric["ci_upper_95"] * Decimal(100)
                colour = RESOLUTION_COLOURS[record.resolution]
                ax.errorbar(
                    float(numeric["tau_img_ms"]), float(y),
                    yerr=([float(y - lower)], [float(upper - y)]),
                    fmt=FAMILY_MARKERS[plot_family(record)],
                    color=colour, markerfacecolor=colour,
                    markeredgecolor="black", markeredgewidth=0.4,
                    markersize=5.2, ecolor=colour, elinewidth=0.7,
                    capsize=1.7, zorder=3,
                )
                point_trace.append({
                    "model_id": record.model_id,
                    "task": task,
                    "adapter_panel": protocol,
                    "resolution": record.resolution,
                    "image_batch_size": int(numeric["batch_size"]),
                    "tau_img_ms": str(numeric["tau_img_ms"]),
                    "performance_percent": str(y),
                    "ci_lower_95_percent": str(lower),
                    "ci_upper_95_percent": str(upper),
                })

            label = "abcd"[row * 2 + col]
            task_label = "Pet" if row == 0 else "Flickr30k"
            batch = 64 if protocol == "OpenCLIP" else 32
            ax.set_title(
                f"({label}) {task_label}: {protocol} (image batch {batch})",
                fontsize=8.6, pad=6,
            )
            ax.set_xscale("log")
            # Common scales within each adapter column, separate protocol panels.
            if protocol == "OpenCLIP":
                ax.set_xlim(0.075, 15)
                ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
            else:
                ax.set_xlim(0.17, 15)
                ax.xaxis.set_major_locator(FixedLocator([0.2, 1, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim((38, 98) if row == 0 else (25, 96))
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            for record in selected:
                if not (
                    x_min < float(record.numeric["tau_img_ms"]) < x_max
                    and y_min < float(record.numeric["ci_lower_95"] * 100)
                    and float(record.numeric["ci_upper_95"] * 100) < y_max
                ):
                    raise ValueError(f"Plot axes would clip {record.key!r}")
            ax.grid(True, which="major", linewidth=0.4, color="#d0d0d0")
            ax.grid(True, which="minor", axis="x", linewidth=0.3, color="#e8e8e8")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8, length=3)
            if col == 0:
                ax.set_ylabel(
                    "Top-1 accuracy (%)" if row == 0 else "Mean Recall@1 (%)",
                    fontsize=8.5,
                )

    fig.text(
        0.54, 0.166,
        r"Measured image-path time $\tau_{\rm img}$ (ms/image; log scale)",
        ha="center", va="center", fontsize=8.7,
    )
    resolution_handles, family_handles = publication_legend_handles(
        primary, include_frontier=False,
    )
    fig.legend(
        handles=resolution_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.065), ncol=5, frameon=False,
        handletextpad=0.4, columnspacing=1.6, fontsize=8,
    )
    fig.legend(
        handles=family_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.009), ncol=4, frameon=False,
        handletextpad=0.4, columnspacing=1.7, fontsize=8,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PDF_PATH, format="pdf", metadata={
            "Title": "Performance under recorded adapter protocols",
            "Creator": "analysis/render_camera_ready_cost_figure.py",
            "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)

    TRACE_PATH.write_text(json.dumps({
        "source": str(DEFAULT_INPUT.relative_to(ROOT)),
        "source_sha256": input_hash,
        "plot_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "output_pdf": str(PDF_PATH.relative_to(ROOT)),
        "output_pdf_sha256": sha256_file(PDF_PATH),
        "primary_cells_plotted": len(primary),
        "supplementary_paligemma_cells_omitted": len(records) - len(primary),
        "changes": "Separate adapter panels and explicit protocol labels only; unchanged values and intervals.",
        "points": point_trace,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PDF_PATH.relative_to(ROOT)} ({len(primary)} unchanged cells)")


if __name__ == "__main__":
    main()
