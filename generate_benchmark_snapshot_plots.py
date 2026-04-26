#!/usr/bin/env python3
"""Generate README-ready evaluation plots from the April 26 benchmark snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SNAPSHOT_PATH = Path("evaluation_snapshot_apr26.json")
PLOT_DIR = Path("plots")
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
LEVEL_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "level_4": "Level 4",
    "level_5": "Level 5",
}
LEVEL_COLORS = {
    "easy": "#2E8B57",
    "medium": "#2F6BFF",
    "hard": "#E68613",
    "level_4": "#C13C37",
    "level_5": "#6B4EA0",
}
AGENT_ORDER = ["random", "heuristic", "trained"]
AGENT_LABELS = {"random": "Random", "heuristic": "Heuristic", "trained": "Trained"}
AGENT_COLORS = {"random": "#ff5a5f", "heuristic": "#f59e0b", "trained": "#06b6d4"}


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C9CDD3",
            "axes.labelcolor": "#1F2937",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "figure.titlesize": 18,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D7DCE2",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def agent_level_summary(payload: dict, agent_key: str, level: str) -> dict:
    return payload["summary"][agent_key][level]


def agent_level_episodes(payload: dict, agent_key: str, level: str) -> list[dict]:
    return [
        episode
        for episode in payload["episodes"]
        if episode["agent"] == agent_key and episode["level"] == level
    ]


def savefig(name: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOT_DIR / name, dpi=260)
    plt.close()


def build_console_lines(payload: dict) -> list[str]:
    lines = []
    rule = "-" * 76
    header = f"{'AGENT':<10} | {'LEVEL':<8} | {'GRADE (+/- STD)':<18} | {'REWARD':>8} | {'REV':>7} | {'SEC':>7} | {'CAUGHT':>7}"
    lines.append(rule)
    lines.append(header)
    lines.append(rule)
    for agent_key in AGENT_ORDER:
        for level in LEVELS:
            row = agent_level_summary(payload, agent_key, level)
            lines.append(
                f"{AGENT_LABELS[agent_key]:<10} | {level:<8} | "
                f"{row['grade_mean']:.3f} +/- {row['grade_std']:.3f} | "
                f"{row['reward_mean']:>8.2f} | {row['revenue_mean']:>7.1f} | "
                f"{row['security_mean']:>7.1f} | {row['sleepers_caught_mean']:>7.2f}"
            )
        lines.append(rule)
    return lines


def plot_benchmark_summary_table(payload: dict) -> None:
    lines = build_console_lines(payload)
    fig_height = max(7.2, 0.42 * len(lines))
    fig, ax = plt.subplots(figsize=(13.5, fig_height))
    bg = "#0d1117"
    fg = "#d1d5db"
    muted = "#9ca3af"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    y_top = 0.98
    step = 0.92 / max(len(lines), 1)
    for idx, line in enumerate(lines):
        color = muted if set(line) == {"-"} else fg
        ax.text(
            0.015,
            y_top - idx * step,
            line,
            family="DejaVu Sans Mono",
            fontsize=12.5,
            color=color,
            va="top",
            ha="left",
        )

    plt.savefig(PLOT_DIR / "benchmark_summary_table.png", dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grades(payload: dict) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.24
    x = np.arange(len(LEVELS))

    for idx, agent_key in enumerate(AGENT_ORDER):
        means = [agent_level_summary(payload, agent_key, level)["grade_mean"] for level in LEVELS]
        stds = [agent_level_summary(payload, agent_key, level)["grade_std"] for level in LEVELS]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            color=AGENT_COLORS[agent_key],
            label=AGENT_LABELS[agent_key],
            yerr=stds,
            capsize=4,
            alpha=0.9,
        )

    ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Composite grade")
    ax.set_title("Benchmark snapshot: grade by level and agent")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    savefig("comparison_grades.png")


def plot_comparison_operations(payload: dict) -> None:
    setup_style()
    metrics = [
        ("reward_mean", "Reward", "Mean episode reward"),
        ("revenue_mean", "Revenue", "Final enterprise revenue"),
        ("security_mean", "Security", "Final security score"),
        ("sleepers_caught_mean", "Caught", "Sleepers caught"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.24
    x = np.arange(len(LEVELS))

    for ax, (metric, ylabel, title) in zip(axes.flatten(), metrics):
        for idx, agent_key in enumerate(AGENT_ORDER):
            values = [agent_level_summary(payload, agent_key, level)[metric] for level in LEVELS]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                color=AGENT_COLORS[agent_key],
                label=AGENT_LABELS[agent_key],
                alpha=0.9,
            )
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS], rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Reward, revenue, security, and sleepers caught", y=1.03)
    fig.tight_layout()
    savefig("comparison_operations.png")


def plot_comparison_radar(payload: dict) -> None:
    setup_style()
    metric_names = ["grade_mean", "reward_mean", "revenue_mean", "security_mean", "sleepers_caught_mean"]
    metric_labels = ["Grade", "Reward", "Revenue", "Security", "Caught"]
    overall = {}
    for agent_key in AGENT_ORDER:
        overall[agent_key] = {}
        for metric in metric_names:
            values = [agent_level_summary(payload, agent_key, level)[metric] for level in LEVELS]
            overall[agent_key][metric] = float(np.mean(values))

    normalized = {}
    for metric in metric_names:
        column = np.array([overall[agent_key][metric] for agent_key in AGENT_ORDER], dtype=float)
        lo = column.min()
        hi = column.max()
        if np.isclose(lo, hi):
            for agent_key in AGENT_ORDER:
                normalized.setdefault(agent_key, {})[metric] = 0.5
        else:
            for agent_key, value in zip(AGENT_ORDER, column):
                normalized.setdefault(agent_key, {})[metric] = float((value - lo) / (hi - lo))

    angles = [idx / float(len(metric_names)) * 2 * np.pi for idx in range(len(metric_names))]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for agent_key in AGENT_ORDER:
        values = [normalized[agent_key][metric] * 100 for metric in metric_names]
        values += values[:1]
        ax.plot(angles, values, color=AGENT_COLORS[agent_key], linewidth=2, label=AGENT_LABELS[agent_key])
        ax.fill(angles, values, color=AGENT_COLORS[agent_key], alpha=0.14)

    ax.set_xticks(angles[:-1], metric_labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Normalized benchmark metric radar", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.08))
    fig.tight_layout()
    savefig("comparison_radar.png")


def plot_reward_distributions(payload: dict) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2.2, 1.2])
    rng = np.random.default_rng(7)
    width = 0.24
    x = np.arange(len(LEVELS))

    for idx, agent_key in enumerate(AGENT_ORDER):
        offset = (idx - 1) * width
        for level_idx, level in enumerate(LEVELS):
            rewards = [episode["reward"] for episode in agent_level_episodes(payload, agent_key, level)]
            position = level_idx + offset
            box = axes[0].boxplot(
                [rewards],
                positions=[position],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(AGENT_COLORS[agent_key])
                patch.set_alpha(0.45)
                patch.set_edgecolor(AGENT_COLORS[agent_key])
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.3)
            jitter = rng.normal(position, 0.016, size=len(rewards))
            axes[0].scatter(
                jitter,
                rewards,
                s=26,
                color=AGENT_COLORS[agent_key],
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )

        means = [agent_level_summary(payload, agent_key, level)["reward_mean"] for level in LEVELS]
        axes[1].plot(x, means, marker="o", linewidth=2.2, color=AGENT_COLORS[agent_key], label=AGENT_LABELS[agent_key])

    axes[0].set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward distributions from the April 26 benchmark snapshot")
    axes[0].grid(axis="y")

    axes[1].set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("Reward trend by difficulty")
    axes[1].grid(axis="y")
    axes[1].legend(ncol=3, loc="upper left")

    fig.tight_layout()
    savefig("reward_distributions.png")


def plot_reward_frontier(payload: dict) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharex=True, sharey=True)
    legend_handles = {}

    for ax, agent_key in zip(axes, AGENT_ORDER):
        for level in LEVELS:
            episodes = agent_level_episodes(payload, agent_key, level)
            x_values = [episode["security"] for episode in episodes]
            y_values = [episode["reward"] for episode in episodes]
            sizes = [max(40.0, episode["revenue"] * 0.14) for episode in episodes]
            scatter = ax.scatter(
                x_values,
                y_values,
                s=sizes,
                color=LEVEL_COLORS[level],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.6,
                label=LEVEL_LABELS[level],
            )
            legend_handles[level] = scatter
        ax.axvline(50.0, color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
        ax.set_title(AGENT_LABELS[agent_key])
        ax.set_xlabel("Final security score")
        ax.grid(True)

    axes[0].set_ylabel("Episode reward")
    fig.suptitle("Reward-security frontier (marker size proportional to revenue)", y=1.03)
    fig.legend(
        [legend_handles[level] for level in LEVELS],
        [LEVEL_LABELS[level] for level in LEVELS],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    fig.tight_layout()
    savefig("reward_frontier.png")


def plot_reward_turn_dynamics(payload: dict) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(LEVELS))

    for agent_key in AGENT_ORDER:
        reward_means = [agent_level_summary(payload, agent_key, level)["reward_mean"] for level in LEVELS]
        security_means = [agent_level_summary(payload, agent_key, level)["security_mean"] for level in LEVELS]
        axes[0].plot(x, reward_means, marker="o", linewidth=2.2, color=AGENT_COLORS[agent_key], label=AGENT_LABELS[agent_key])
        axes[1].plot(x, security_means, marker="o", linewidth=2.2, color=AGENT_COLORS[agent_key], label=AGENT_LABELS[agent_key])

    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("Reward response across escalating difficulty")
    axes[0].grid(True)
    axes[0].legend(ncol=3, loc="upper left")

    axes[1].set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    axes[1].set_ylabel("Mean security")
    axes[1].set_title("Security retention across escalating difficulty")
    axes[1].grid(True)

    fig.tight_layout()
    savefig("reward_turn_dynamics.png")


def plot_scenario_timeline(payload: dict) -> None:
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    metrics = [("reward", "Episode reward"), ("revenue", "Final revenue"), ("security", "Final security")]
    x_labels = []
    x_positions = []
    cursor = 0

    for level in LEVELS:
        for agent_key in AGENT_ORDER:
            episodes = agent_level_episodes(payload, agent_key, level)
            for episode in episodes:
                x_positions.append(cursor)
                x_labels.append(f"{LEVEL_LABELS[level]}\n{AGENT_LABELS[agent_key][0]}{episode['episode']}")
                cursor += 1

    for ax, (metric_key, title) in zip(axes, metrics):
        cursor = 0
        for level in LEVELS:
            for agent_key in AGENT_ORDER:
                episodes = agent_level_episodes(payload, agent_key, level)
                values = [episode[metric_key] for episode in episodes]
                positions = list(range(cursor, cursor + len(episodes)))
                ax.plot(
                    positions,
                    values,
                    marker="o",
                    linewidth=1.4,
                    color=AGENT_COLORS[agent_key],
                    alpha=0.95,
                )
                cursor += len(episodes)
        ax.set_title(title)
        ax.grid(True, axis="y")

    axes[-1].set_xticks(x_positions, x_labels, rotation=60, ha="right")
    fig.suptitle("Episode-outcome panorama across all logged benchmark runs", y=0.98)
    fig.tight_layout()
    savefig("scenario_timeline.png")


def main() -> None:
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    plot_benchmark_summary_table(payload)
    plot_comparison_grades(payload)
    plot_comparison_operations(payload)
    plot_comparison_radar(payload)
    plot_reward_distributions(payload)
    plot_reward_frontier(payload)
    plot_reward_turn_dynamics(payload)
    plot_scenario_timeline(payload)

    print("Generated benchmark snapshot plots:")
    for filename in [
        "benchmark_summary_table.png",
        "comparison_grades.png",
        "comparison_operations.png",
        "comparison_radar.png",
        "reward_distributions.png",
        "reward_frontier.png",
        "reward_turn_dynamics.png",
        "scenario_timeline.png",
    ]:
        print(f"  - {PLOT_DIR / filename}")


if __name__ == "__main__":
    main()
