#!/usr/bin/env python3
"""Render the locked primary resolution sweeps with a non-overlapping legend.

Run with Python and matplotlib available:
    python analysis/render_camera_ready_resolution_figure.py

All 52 primary cells and their existing 95% confidence limits are read from the
canonical CSV. The single MobileCLIP cell is shown without a connecting line.
Straight lines join measured points from the same checkpoint only; they are
visual guides, not additional evaluations or estimates of intermediate values.
No models are executed and no existing input or archived figure is modified.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True

from regenerate_results import (
    DEFAULT_INPUT,
    FAMILY_MARKERS,
    MODEL_ORDER,
    READABLE_NAMES,
    configure_publication_matplotlib,
    load_and_validate,
    plot_family,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/processed/camera_ready/figures"
PDF_PATH = OUTPUT / "performance_vs_resolution.pdf"
TRACE_PATH = OUTPUT / "performance_vs_resolution.trace.json"
STYLES = {
    "clip_rn50_openai": ("#0072B2", "-"),
    "clip_vit_b32_openai": ("#E69F00", "--"),
    "clip_vit_b16_openai": ("#009E73", "-."),
    "clip_vit_l14_openai": ("#AA66AA", ":"),
    "siglip_b16_webli": ("#D55E00", "-"),
    "mobileclip_s1_datacompdr": ("#369CB8", "none"),
    "blip_itm_base_coco": ("#777777", "--"),
    "blip_itm_large_coco": ("#222222", ":"),
}


def main() -> None:
    records, _, input_hash = load_and_validate(DEFAULT_INPUT)
    primary = [r for r in records if not r.model_id.startswith("paligemma")]
    if len(records) != 54 or len(primary) != 52:
        raise ValueError("Expected the locked 54-cell source with 52 primary cells")

    configure_publication_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.7))
    fig.subplots_adjust(
        left=0.09, right=0.985, top=0.91, bottom=0.35, wspace=0.26,
    )
    points = []
    line_groups = []
    for panel, (ax, task) in enumerate(zip(
        axes, ("zeroshot_classification", "image_text_retrieval"),
    )):
        by_model = defaultdict(list)
        for record in primary:
            if record.task == task:
                by_model[record.model_id].append(record)

        for model_id in MODEL_ORDER:
            if model_id not in by_model:
                continue
            selected = sorted(by_model[model_id], key=lambda r: r.resolution)
            colour, style = STYLES[model_id]
            resolutions = [r.resolution for r in selected]
            values = [r.numeric["performance_value"] * Decimal(100) for r in selected]
            if len(selected) >= 2:
                ax.plot(
                    resolutions, [float(value) for value in values],
                    color=colour, linestyle=style, linewidth=0.9,
                    alpha=0.9, zorder=2,
                )
                line_groups.append({
                    "model_id": model_id, "task": task,
                    "sampled_resolutions_connected": resolutions,
                })
            for record, value in zip(selected, values):
                lower = record.numeric["ci_lower_95"] * Decimal(100)
                upper = record.numeric["ci_upper_95"] * Decimal(100)
                ax.errorbar(
                    record.resolution, float(value),
                    yerr=([float(value - lower)], [float(upper - value)]),
                    fmt=FAMILY_MARKERS[plot_family(record)],
                    color=colour,
                    markerfacecolor="white" if len(selected) == 1 else colour,
                    markeredgecolor=colour,
                    markeredgewidth=0.8 if len(selected) == 1 else 0.55,
                    markersize=5.5 if len(selected) == 1 else 4.4,
                    elinewidth=0.65, capsize=1.6, zorder=4,
                )
                points.append({
                    "model_id": model_id,
                    "task": task,
                    "resolution": record.resolution,
                    "performance_percent": str(value),
                    "ci_lower_95_percent": str(lower),
                    "ci_upper_95_percent": str(upper),
                    "connected_to_other_measured_resolutions": len(selected) > 1,
                })

        ax.axvline(224, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(384, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
        ax.set_title(
            "(a) Oxford-IIIT Pet classification" if panel == 0 else "(b) Flickr30k retrieval",
            fontsize=8.8, pad=8,
        )
        ax.set_ylabel("Top-1 accuracy (%)" if panel == 0 else "Mean Recall@1 (%)", fontsize=8.5)
        ax.set_xlim(145, 465)
        ax.set_xticks([160, 224, 336, 384, 448])
        ax.set_ylim((38, 98) if panel == 0 else (25, 96))
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        for record in (r for r in primary if r.task == task):
            if not (
                x_min < record.resolution < x_max
                and y_min < float(record.numeric["ci_lower_95"] * 100)
                and float(record.numeric["ci_upper_95"] * 100) < y_max
            ):
                raise ValueError(f"Plot axes would clip {record.key!r}")
        ax.tick_params(labelsize=8, length=3)
        ax.grid(True, color="#dddddd", linewidth=0.35)
        ax.set_axisbelow(True)

    fig.text(
        0.54, 0.27, "Evaluation resolution (pixels)",
        ha="center", va="center", fontsize=8.7,
    )
    fig.text(
        0.54, 0.218,
        "Lines join sampled points; intermediate resolutions were not evaluated.",
        ha="center", va="center", fontsize=7.8,
    )
    reference_handles = [
        Line2D([0], [0], color="#888888", linestyle="--", linewidth=0.8,
               label="224 px reference (CLIP / SigLIP / MobileCLIP)"),
        Line2D([0], [0], color="#888888", linestyle=":", linewidth=0.8,
               label="384 px reference (BLIP)"),
    ]
    fig.legend(
        handles=reference_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.133), ncol=2, frameon=False,
        handletextpad=0.5, columnspacing=1.9, fontsize=7.7,
    )
    model_handles = []
    for model_id in MODEL_ORDER:
        if model_id not in STYLES:
            continue
        representative = next(r for r in primary if r.model_id == model_id)
        colour, style = STYLES[model_id]
        model_handles.append(Line2D(
            [0], [0], color=colour, linestyle=style, linewidth=0.9,
            marker=FAMILY_MARKERS[plot_family(representative)],
            markerfacecolor="white" if style == "none" else colour,
            markeredgewidth=0.6, markersize=4.5,
            label=READABLE_NAMES[model_id],
        ))
    fig.legend(
        handles=model_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False,
        handletextpad=0.5, columnspacing=1.2, fontsize=7.8,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", metadata={
        "Title": "Measured within-checkpoint resolution sweeps",
        "Creator": "analysis/render_camera_ready_resolution_figure.py",
        "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)
    if len(points) != 52 or len(line_groups) != 14:
        raise ValueError("Unexpected point or within-checkpoint line count")
    TRACE_PATH.write_text(json.dumps({
        "source": str(DEFAULT_INPUT.relative_to(ROOT)),
        "source_sha256": input_hash,
        "plot_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "output_pdf": str(PDF_PATH.relative_to(ROOT)),
        "output_pdf_sha256": sha256_file(PDF_PATH),
        "primary_cells_plotted": len(points),
        "reference_resolutions_px": {"CLIP_SigLIP_MobileCLIP": 224, "BLIP": 384},
        "points": points,
        "within_checkpoint_lines": line_groups,
        "changes": "Checkpoint-coloured lines and shared legends; all 52 primary cells shown, including the unconnected MobileCLIP singleton, with existing 95% confidence limits. No new measurements.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PDF_PATH.relative_to(ROOT)} ({len(points)} unchanged primary cells)")


if __name__ == "__main__":
    main()
