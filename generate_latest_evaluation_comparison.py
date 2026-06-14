#!/usr/bin/env python3
"""Generate the direct latest base-vs-fixed evaluation comparison plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_evaluation_plots import LEVEL_LABELS, LEVELS, setup_style

INPUT_PATH = Path("evaluation_comparison_latest.json")
OUTPUT_PATH = Path("plots/base_vs_fixed_comparison.png")

SERIES = [
    ("base_untrained", "Base untrained", "#64748B", "--"),
    ("fixed_trained", "Fixed trained", "#06B6D4", "-"),
    ("heuristic", "Heuristic", "#F59E0B", ":"),
]

METRICS = [
    ("grade", "Composite grade"),
    ("reward", "Episode reward"),
    ("revenue", "Final revenue"),
    ("security", "Final security"),
    ("sleepers_caught", "Sleepers caught"),
]


def main() -> None:
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    agents = payload["agents"]
    averages = payload["level_macro_averages"]
    delta = payload["fixed_minus_base"]

    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    flat = axes.flatten()
    x = list(range(len(LEVELS)))
    labels = [LEVEL_LABELS[level] for level in LEVELS]

    for ax, (metric, title) in zip(flat, METRICS):
        for key, label, color, linestyle in SERIES:
            ax.plot(
                x,
                agents[key][metric],
                marker="o",
                linewidth=2.5,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        ax.set_xticks(x, labels, rotation=15)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    summary = flat[5]
    summary.axis("off")
    summary.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Level-macro benchmark summary",
                "",
                f"Fixed grade: {averages['fixed_trained']['grade']:.4f}",
                f"Base grade: {averages['base_untrained']['grade']:.4f}",
                f"Heuristic grade: {averages['heuristic']['grade']:.4f}",
                "",
                "Fixed minus base:",
                f"  Grade: {delta['grade']:+.4f}",
                f"  Reward: {delta['reward']:+.3f}",
                f"  Revenue: {delta['revenue']:+.2f}",
                f"  Security: {delta['security']:+.2f}",
                f"  Sleepers caught: {delta['sleepers_caught']:+.2f}",
                "",
                "Verdict: composite grade improved,",
                "but Level-4/5 security and capture",
                "performance regressed versus base.",
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    handles, legend_labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Latest matched benchmark: base vs fixed trained vs heuristic", y=1.01)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
