#!/usr/bin/env python3
"""Generate publication-style evaluation and reward plots from evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
DIMENSIONS = ["security", "revenue", "intelligence", "adaptability", "efficiency"]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_builtin(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


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


def ensure_plot_dir(plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)


def level_summary(payload: dict[str, Any], agent_key: str, level: str) -> dict[str, Any]:
    return payload["agents"][agent_key]["summary"][level]


def level_episodes(payload: dict[str, Any], agent_key: str, level: str) -> list[dict[str, Any]]:
    return payload["agents"][agent_key]["episodes"][level]


def representative_episode(payload: dict[str, Any], agent_key: str, level: str) -> dict[str, Any]:
    episodes = level_episodes(payload, agent_key, level)
    if not episodes:
        return {}
    return max(episodes, key=lambda episode: episode["grade"]["score"])


def mean_std_band(series_collection: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack(series_collection)
    return matrix.mean(axis=0), matrix.std(axis=0)


def interpolate_series(values: list[float], target_points: int = 100) -> np.ndarray:
    if not values:
        return np.zeros(target_points, dtype=float)
    if len(values) == 1:
        return np.full(target_points, values[0], dtype=float)
    x_original = np.linspace(0.0, 1.0, num=len(values))
    x_target = np.linspace(0.0, 1.0, num=target_points)
    return np.interp(x_target, x_original, np.asarray(values, dtype=float))


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def build_plot_manifest(plot_dir: Path) -> dict[str, str]:
    return {
        "comparison_grades": str(plot_dir / "comparison_grades.png"),
        "comparison_operations": str(plot_dir / "comparison_operations.png"),
        "comparison_radar": str(plot_dir / "comparison_radar.png"),
        "scenario_timeline": str(plot_dir / "scenario_timeline.png"),
        "reward_distributions": str(plot_dir / "reward_distributions.png"),
        "reward_frontier": str(plot_dir / "reward_frontier.png"),
        "reward_turn_dynamics": str(plot_dir / "reward_turn_dynamics.png"),
    }


def plot_grade_comparison(payload: dict[str, Any], output_path: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.24
    x_positions = list(range(len(LEVELS)))

    for idx, agent_key in enumerate(AGENT_ORDER):
        means = [level_summary(payload, agent_key, level)["grade_mean"] for level in LEVELS]
        stds = [level_summary(payload, agent_key, level)["grade_std"] for level in LEVELS]
        shifted = [x + (idx - 1) * width for x in x_positions]
        ax.bar(
            shifted,
            means,
            width=width,
            color=AGENT_COLORS[agent_key],
            label=AGENT_LABELS[agent_key],
            yerr=stds,
            capsize=4,
            alpha=0.9,
        )

    ax.set_xticks(x_positions, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Composite grade")
    ax.set_title("Panopticon ARGUS evaluation: grade by level and agent")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_operations_comparison(payload: dict[str, Any], output_path: Path) -> None:
    setup_style()
    metrics = [
        ("reward_mean", "Reward", "Mean episode reward"),
        ("revenue_mean", "Revenue", "Final enterprise revenue"),
        ("security_mean", "Security", "Final security score"),
        ("sleepers_caught_mean", "Caught", "Sleepers caught"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()
    width = 0.24
    x_positions = list(range(len(LEVELS)))

    for ax, (metric_key, short_label, title) in zip(axes_flat, metrics):
        for idx, agent_key in enumerate(AGENT_ORDER):
            means = [level_summary(payload, agent_key, level)[metric_key] for level in LEVELS]
            shifted = [x + (idx - 1) * width for x in x_positions]
            ax.bar(
                shifted,
                means,
                width=width,
                color=AGENT_COLORS[agent_key],
                label=AGENT_LABELS[agent_key],
                alpha=0.9,
            )
        ax.set_xticks(x_positions, [LEVEL_LABELS[level] for level in LEVELS], rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(short_label)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Operational comparison across all five Panopticon levels", y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_radar_comparison(payload: dict[str, Any], output_path: Path) -> None:
    setup_style()
    angles = [idx / float(len(DIMENSIONS)) * 2 * np.pi for idx in range(len(DIMENSIONS))]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for agent_key in AGENT_ORDER:
        overall = payload["agents"][agent_key]["overall"]["grader_dimensions"]
        values = [overall[dimension]["mean"] * 100 for dimension in DIMENSIONS]
        values += values[:1]
        ax.plot(angles, values, color=AGENT_COLORS[agent_key], linewidth=2, label=AGENT_LABELS[agent_key])
        ax.fill(angles, values, color=AGENT_COLORS[agent_key], alpha=0.14)

    ax.set_xticks(angles[:-1], [dimension.title() for dimension in DIMENSIONS])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_title("Average grader dimensions across all evaluated episodes", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_timeline_comparison(payload: dict[str, Any], output_path: Path, level: str) -> None:
    setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    metric_titles = [
        "Per-turn reward",
        "Cumulative reward",
        "Enterprise revenue",
        "Security score",
    ]

    for agent_key in AGENT_ORDER:
        episode = representative_episode(payload, agent_key, level)
        timeline = episode.get("timeline", [])
        x_values = [item["turn"] for item in timeline]
        reward_values = [item["reward"] for item in timeline]
        cumulative_values = np.cumsum(reward_values) if reward_values else []
        revenue_values = [item["metrics_after"]["enterprise_revenue"] for item in timeline]
        security_values = [item["metrics_after"]["security_score"] for item in timeline]
        series = [reward_values, cumulative_values, revenue_values, security_values]

        for ax, values, title in zip(axes, series, metric_titles):
            ax.plot(
                x_values,
                values,
                label=AGENT_LABELS[agent_key],
                color=AGENT_COLORS[agent_key],
                linewidth=2,
            )
            ax.set_title(title)
            ax.grid(alpha=0.25)

    axes[-1].set_xlabel(f"Turn ({LEVEL_LABELS[level]})")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f"Representative turn-by-turn comparison on {LEVEL_LABELS[level]}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_reward_distributions(payload: dict[str, Any], output_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2.2, 1.2])

    rng = np.random.default_rng(7)
    width = 0.24
    x_positions = np.arange(len(LEVELS))

    for idx, agent_key in enumerate(AGENT_ORDER):
        agent_offset = (idx - 1) * width
        for level_idx, level in enumerate(LEVELS):
            rewards = [episode["total_reward"] for episode in level_episodes(payload, agent_key, level)]
            position = level_idx + agent_offset
            if rewards:
                box = axes[0].boxplot(
                    [rewards],
                    positions=[position],
                    widths=0.18,
                    patch_artist=True,
                    showcaps=True,
                    showfliers=False,
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(AGENT_COLORS[agent_key])
                    patch.set_alpha(0.45)
                    patch.set_edgecolor(AGENT_COLORS[agent_key])
                for median in box["medians"]:
                    median.set_color("#111827")
                    median.set_linewidth(1.3)
                jitter = rng.normal(position, 0.018, size=len(rewards))
                axes[0].scatter(
                    jitter,
                    rewards,
                    s=28,
                    color=AGENT_COLORS[agent_key],
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )

        means = [level_summary(payload, agent_key, level)["reward_mean"] for level in LEVELS]
        stds = [level_summary(payload, agent_key, level)["reward_std"] for level in LEVELS]
        axes[1].plot(
            x_positions,
            means,
            marker="o",
            linewidth=2.3,
            color=AGENT_COLORS[agent_key],
            label=AGENT_LABELS[agent_key],
        )
        axes[1].fill_between(
            x_positions,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=AGENT_COLORS[agent_key],
            alpha=0.12,
        )

    axes[0].set_xticks(x_positions, [LEVEL_LABELS[level] for level in LEVELS])
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward distributions by level and agent")
    axes[0].grid(True, axis="y")

    axes[1].set_xticks(x_positions, [LEVEL_LABELS[level] for level in LEVELS])
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("Reward trend with one-standard-deviation envelope")
    axes[1].grid(True, axis="y")
    axes[1].legend(ncol=3, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_reward_frontier(payload: dict[str, Any], output_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    legend_handles: dict[str, Any] = {}

    for ax, agent_key in zip(axes, AGENT_ORDER):
        for level in LEVELS:
            episodes = level_episodes(payload, agent_key, level)
            x_values = [episode["final_state"]["security_score"] for episode in episodes]
            y_values = [episode["total_reward"] for episode in episodes]
            sizes = [max(40.0, episode["final_state"]["enterprise_revenue"] * 0.12) for episode in episodes]
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
    fig.suptitle("Reward-security frontier (marker size encodes final revenue)", y=1.02)
    fig.legend(
        [legend_handles[level] for level in LEVELS],
        [LEVEL_LABELS[level] for level in LEVELS],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_reward_turn_dynamics(payload: dict[str, Any], output_path: Path, level: str) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x_axis = np.linspace(0.0, 100.0, num=100)

    for agent_key in AGENT_ORDER:
        reward_curves = []
        cumulative_curves = []
        for episode in level_episodes(payload, agent_key, level):
            reward_series = interpolate_series(episode.get("reward_history", []), target_points=100)
            reward_curves.append(rolling_mean(reward_series, 7))
            cumulative_curves.append(np.cumsum(reward_series))

        if not reward_curves:
            continue

        reward_mean, reward_std = mean_std_band(reward_curves)
        cumulative_mean, cumulative_std = mean_std_band(cumulative_curves)

        axes[0].plot(x_axis, reward_mean, color=AGENT_COLORS[agent_key], linewidth=2.2, label=AGENT_LABELS[agent_key])
        axes[0].fill_between(
            x_axis,
            reward_mean - reward_std,
            reward_mean + reward_std,
            color=AGENT_COLORS[agent_key],
            alpha=0.12,
        )

        axes[1].plot(
            x_axis,
            cumulative_mean,
            color=AGENT_COLORS[agent_key],
            linewidth=2.2,
            label=AGENT_LABELS[agent_key],
        )
        axes[1].fill_between(
            x_axis,
            cumulative_mean - cumulative_std,
            cumulative_mean + cumulative_std,
            color=AGENT_COLORS[agent_key],
            alpha=0.12,
        )

    axes[0].set_title(f"Instantaneous reward profile on {LEVEL_LABELS[level]}")
    axes[0].set_ylabel("Per-turn reward")
    axes[0].grid(True)
    axes[0].legend(ncol=3, loc="upper left")

    axes[1].set_title(f"Cumulative reward accumulation on {LEVEL_LABELS[level]}")
    axes[1].set_xlabel("Episode progress (%)")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def render_evaluation_plots(payload: dict[str, Any], plot_dir: Path, timeline_level: str) -> dict[str, str]:
    ensure_plot_dir(plot_dir)
    plot_files = build_plot_manifest(plot_dir)
    plot_grade_comparison(payload, Path(plot_files["comparison_grades"]))
    plot_operations_comparison(payload, Path(plot_files["comparison_operations"]))
    plot_radar_comparison(payload, Path(plot_files["comparison_radar"]))
    plot_timeline_comparison(payload, Path(plot_files["scenario_timeline"]), timeline_level)
    plot_reward_distributions(payload, Path(plot_files["reward_distributions"]))
    plot_reward_frontier(payload, Path(plot_files["reward_frontier"]))
    plot_reward_turn_dynamics(payload, Path(plot_files["reward_turn_dynamics"]), timeline_level)
    return plot_files


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation plots from a Panopticon evaluation JSON file")
    parser.add_argument("--input", default="evaluationResults.json", help="Path to evaluation JSON produced by full_evaluation.py")
    parser.add_argument("--plot-dir", default="plots", help="Directory for generated plots")
    parser.add_argument("--timeline-level", default="level_5", choices=LEVELS, help="Level used for reward/timeline profiles")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    payload = to_builtin(payload)
    plot_files = render_evaluation_plots(payload, Path(args.plot_dir), args.timeline_level)
    print("Generated evaluation plot set:")
    for path in plot_files.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
