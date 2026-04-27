#!/usr/bin/env python3
"""Plot before/after training benchmark comparisons for a selected agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_evaluation_plots import (
    AGENT_LABELS,
    LEVELS,
    LEVEL_LABELS,
    detect_payload_mode,
    level_summary,
    setup_style,
)


def load_payload(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = detect_payload_mode(payload)
    return payload, mode


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare before-vs-after Panopticon benchmark results")
    parser.add_argument("--before", required=True, help="Evaluation JSON for the baseline / before-training run")
    parser.add_argument("--after", required=True, help="Evaluation JSON for the improved / after-training run")
    parser.add_argument("--agent", default="trained", help="Agent key to compare inside each payload")
    parser.add_argument("--before-label", default="Before training", help="Legend label for the baseline payload")
    parser.add_argument("--after-label", default="After training", help="Legend label for the improved payload")
    parser.add_argument("--output", default="plots/prepost_training_comparison.png", help="Output PNG path")
    return parser


def metric_series(payload: dict[str, Any], mode: str, agent_key: str, metric_key: str) -> list[float]:
    return [level_summary(payload, mode, agent_key, level)[metric_key] for level in LEVELS]


def main() -> None:
    args = build_cli().parse_args()
    before_payload, before_mode = load_payload(Path(args.before))
    after_payload, after_mode = load_payload(Path(args.after))
    agent_label = AGENT_LABELS.get(args.agent, args.agent.title())

    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    metric_specs = [
        ("grade_mean", "Composite grade"),
        ("reward_mean", "Episode reward"),
        ("revenue_mean", "Final revenue"),
        ("security_mean", "Final security"),
        ("sleepers_caught_mean", "Sleepers caught"),
    ]
    x = list(range(len(LEVELS)))
    x_labels = [LEVEL_LABELS[level] for level in LEVELS]

    for ax, (metric_key, title) in zip(axes_flat, metric_specs):
        before_values = metric_series(before_payload, before_mode, args.agent, metric_key)
        after_values = metric_series(after_payload, after_mode, args.agent, metric_key)
        ax.plot(x, before_values, marker="o", linewidth=2.2, linestyle="--", color="#64748B", label=args.before_label)
        ax.plot(x, after_values, marker="o", linewidth=2.4, color="#06B6D4", label=args.after_label)
        ax.set_xticks(x, x_labels, rotation=15)
        ax.set_title(f"{agent_label}: {title}")
        ax.grid(True, axis="y", alpha=0.3)

    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"Agent compared: {agent_label}",
                f"Before file: {Path(args.before).name}",
                f"After file: {Path(args.after).name}",
                "",
                "This figure is intended for judge-facing",
                "before/after analysis: did the tuned policy",
                "improve grade, reward, revenue, security,",
                "and sleeper capture across the five levels?",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
    )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Before-vs-after training comparison across Panopticon levels", y=1.02)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Wrote before/after comparison plot to {output_path}")


if __name__ == "__main__":
    main()
