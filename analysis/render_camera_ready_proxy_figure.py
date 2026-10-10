#!/usr/bin/env python3
"""Combine the two archived secondary proxy plots with readable shared legends.

Run with Python and matplotlib available:
    python analysis/render_camera_ready_proxy_figure.py

Preserves all 54 original point estimates and the original proxy calculation,
including both PaliGemma classification diagnostics and MobileCLIP's documented
parameter-count-only fallback. Like the archived proxy plots, this figure shows
point estimates without confidence bars. No original files are overwritten.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True

from regenerate_results import (
    DEFAULT_INPUT,
    FAMILY_MARKERS,
    RESOLUTION_COLOURS,
    configure_publication_matplotlib,
    load_and_validate,
    parameter_token_proxy_value,
    plot_family,
    publication_legend_handles,
    record_sort_key,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/processed/camera_ready/figures"
PDF_PATH = OUTPUT / "performance_vs_proxy.pdf"
TRACE_PATH = OUTPUT / "performance_vs_proxy.trace.json"


def main() -> None:
    records, _, input_hash = load_and_validate(DEFAULT_INPUT)
    if len(records) != 54:
        raise ValueError("Expected the locked 54-cell source")
    configure_publication_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.7))
    fig.subplots_adjust(
        left=0.09, right=0.985, top=0.84, bottom=0.29, wspace=0.26,
    )
    fig.text(
        0.5, 0.99, "Secondary parameter-token proxy",
        ha="center", va="top", fontsize=10, weight="bold",
    )
    fig.text(
        0.5, 0.938,
        "Descriptive only: not FLOPs or an intrinsic cross-family efficiency ranking.",
        ha="center", va="top", fontsize=8.2,
    )
    points = []
    for panel, (ax, task) in enumerate(zip(
        axes, ("zeroshot_classification", "image_text_retrieval"),
    )):
        selected = sorted((r for r in records if r.task == task), key=record_sort_key)
        for record in selected:
            proxy = parameter_token_proxy_value(record)
            performance = record.numeric["performance_value"] * Decimal(100)
            ax.plot(
                float(proxy), float(performance),
                marker=FAMILY_MARKERS[plot_family(record)], linestyle="none",
                markerfacecolor=RESOLUTION_COLOURS[record.resolution],
                markeredgecolor="black", markeredgewidth=0.4,
                markersize=5.6 if record.model_id.startswith("paligemma") else 5.1,
                zorder=3,
            )
            points.append({
                "model_id": record.model_id, "task": task,
                "resolution": record.resolution,
                "plotted_proxy": str(proxy),
                "proxy_basis": "parameter_count_only" if not (record.raw.get("parameter_token_proxy") or "").strip() else "recorded_parameter_token_proxy",
                "performance_percent": str(performance),
            })
        ax.set_title(
            "(a) Oxford-IIIT Pet classification" if panel == 0 else "(b) Flickr30k retrieval",
            fontsize=8.8, pad=7,
        )
        ax.set_ylabel("Top-1 accuracy (%)" if panel == 0 else "Mean Recall@1 (%)", fontsize=8.5)
        ax.set_xscale("log")
        if panel == 0:
            ax.set_xlim(4.5e7, 5.5e12)
            ax.xaxis.set_major_locator(FixedLocator([1e8, 1e9, 1e10, 1e11, 1e12]))
            ax.set_ylim(38, 98)
        else:
            ax.set_xlim(1.7e9, 7e11)
            ax.xaxis.set_major_locator(FixedLocator([1e10, 1e11]))
            ax.set_ylim(25, 96)
        ax.xaxis.set_minor_formatter(NullFormatter())
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        for record in selected:
            if not (
                x_min < float(parameter_token_proxy_value(record)) < x_max
                and y_min < float(record.numeric["performance_value"] * 100) < y_max
            ):
                raise ValueError(f"Plot axes would clip {record.key!r}")
        ax.tick_params(labelsize=8, length=3)
        ax.grid(True, which="major", color="#d0d0d0", linewidth=0.4)
        ax.grid(True, which="minor", axis="x", color="#e8e8e8", linewidth=0.3)
        ax.set_axisbelow(True)

    fig.text(
        0.54, 0.207, "Parameter-token proxy (log scale)",
        ha="center", va="center", fontsize=8.7,
    )
    fig.text(
        0.54, 0.155, "MobileCLIP uses parameter count alone; PaliGemma points are likelihood diagnostics.",
        ha="center", va="center", fontsize=7.7,
    )
    resolution_handles, _ = publication_legend_handles(records, include_frontier=False)
    fig.legend(
        handles=resolution_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.076), ncol=5, frameon=False,
        handletextpad=0.4, columnspacing=1.7, fontsize=8,
    )
    family_labels = {
        "CLIP": "CLIP", "SigLIP": "SigLIP", "MobileCLIP": "MobileCLIP",
        "BLIP": "BLIP ITC", "PaliGemma": "PaliGemma likelihood",
    }
    family_handles = [Line2D(
        [0], [0], marker=marker, linestyle="none", color="black",
        markerfacecolor="white", markeredgewidth=0.6, markersize=5,
        label=family_labels[family],
    ) for family, marker in FAMILY_MARKERS.items()]
    fig.legend(
        handles=family_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.007), ncol=5, frameon=False,
        handletextpad=0.4, columnspacing=1.25, fontsize=7.8,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", metadata={
        "Title": "Secondary parameter-token proxy",
        "Creator": "analysis/render_camera_ready_proxy_figure.py",
        "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)
    if len(points) != 54:
        raise ValueError("Unexpected plotted point count")
    TRACE_PATH.write_text(json.dumps({
        "source": str(DEFAULT_INPUT.relative_to(ROOT)),
        "source_sha256": input_hash,
        "plot_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "output_pdf": str(PDF_PATH.relative_to(ROOT)),
        "output_pdf_sha256": sha256_file(PDF_PATH),
        "cells_plotted": len(points),
        "changes": "Combined task panels, removed point annotations, shared external legends, and explicit proxy limitations. All original point estimates and proxy values preserved; no new measurements.",
        "points": points,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PDF_PATH.relative_to(ROOT)} ({len(points)} unchanged cells)")


if __name__ == "__main__":
    main()
