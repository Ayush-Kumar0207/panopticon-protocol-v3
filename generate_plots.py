#!/usr/bin/env python3
"""Generate publication-style training analysis plots from output_logs.txt."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOG_CANDIDATES = [Path("output_logs.txt"), Path("output_log.txt")]
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

LEVEL_ORDER = ["easy", "medium", "hard", "level_4", "level_5"]
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


@dataclass
class LevelData:
    name: str
    episodes: int
    examples: int
    token_min: int
    token_max: int
    token_avg: int
    expert_steps: list[int]
    expert_reward: list[float]
    expert_revenue: list[int]
    expert_security: list[int]
    expert_caught: list[int]
    expert_grade: list[float]
    loss: list[float]
    grad_norm: list[float]
    learning_rate: list[float]
    epoch: list[float]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0]
        return value, value
    center = statistics.mean(values)
    spread = statistics.stdev(values) / math.sqrt(len(values))
    radius = 1.96 * spread
    return center - radius, center + radius


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def to_percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def parse_levels(log_text: str) -> dict[str, LevelData]:
    header_pattern = re.compile(r"TRAINING:\s+(\w+)\s+\| Episodes:\s+(\d+)")
    expert_pattern = re.compile(
        r"\[Expert Ep \d+/\d+\]\s+Steps=(\d+)\s+Reward=([-\d.]+)\s+Rev=(\d+)\s+Sec=(\d+)\s+Caught=(\d+)\s+DAs=(\d+)\s+Grade=([\d.]+)"
    )
    metric_pattern = re.compile(
        r"'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.eE+-]+),\s*'learning_rate':\s*([\d.eE+-]+),\s*'epoch':\s*([\d.]+)"
    )
    token_pattern = re.compile(r"\[DATA\] Max:\s*(\d+), Min:\s*(\d+), Avg:\s*(\d+)")
    saved_pattern = re.compile(r"Saved\s+(\d+)\s+examples")

    headers = list(header_pattern.finditer(log_text))
    if not headers:
        raise ValueError("No training blocks found in the provided training log.")

    parsed: dict[str, LevelData] = {}

    for idx, match in enumerate(headers):
        level = match.group(1)
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(log_text)
        block = log_text[match.start() : end]

        token_match = token_pattern.search(block)
        saved_match = saved_pattern.search(block)
        if token_match is None or saved_match is None:
            raise ValueError(f"Missing token/example metadata for level {level}")

        expert_rows = [tuple(row) for row in expert_pattern.findall(block)]
        metric_rows = [tuple(row) for row in metric_pattern.findall(block)]
        if not expert_rows or not metric_rows:
            raise ValueError(f"Missing expert or training metrics for level {level}")

        parsed[level] = LevelData(
            name=level,
            episodes=int(match.group(2)),
            examples=int(saved_match.group(1)),
            token_min=int(token_match.group(2)),
            token_max=int(token_match.group(1)),
            token_avg=int(token_match.group(3)),
            expert_steps=[int(row[0]) for row in expert_rows],
            expert_reward=[float(row[1]) for row in expert_rows],
            expert_revenue=[int(row[2]) for row in expert_rows],
            expert_security=[int(row[3]) for row in expert_rows],
            expert_caught=[int(row[4]) for row in expert_rows],
            expert_grade=[float(row[6]) for row in expert_rows],
            loss=[float(row[0]) for row in metric_rows],
            grad_norm=[float(row[1]) for row in metric_rows],
            learning_rate=[float(row[2]) for row in metric_rows],
            epoch=[float(row[3]) for row in metric_rows],
        )

    return parsed


def resolve_log_file() -> Path:
    for candidate in LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    names = ", ".join(str(path) for path in LOG_CANDIDATES)
    raise FileNotFoundError(f"Could not find any training log file. Checked: {names}")


def build_summary(levels: dict[str, LevelData]) -> dict[str, object]:
    per_level: dict[str, dict[str, object]] = {}

    total_examples = 0
    total_tokens = 0
    total_steps = 0

    for level_name in LEVEL_ORDER:
        if level_name not in levels:
            continue
        level = levels[level_name]
        total_examples += level.examples
        total_tokens += level.examples * level.token_avg
        total_steps += len(level.loss)

        start_loss = level.loss[0]
        final_loss = level.loss[-1]
        loss_delta = start_loss - final_loss
        half_target = final_loss + 0.5 * loss_delta
        near_final_target = final_loss + 0.1 * loss_delta
        half_step = next((idx + 1 for idx, value in enumerate(level.loss) if value <= half_target), len(level.loss))
        near_final_step = next(
            (idx + 1 for idx, value in enumerate(level.loss) if value <= near_final_target),
            len(level.loss),
        )
        grade_interval = ci95(level.expert_grade)

        per_level[level_name] = {
            "label": LEVEL_LABELS[level_name],
            "episodes": level.episodes,
            "examples": level.examples,
            "tokens": {
                "min": level.token_min,
                "avg": level.token_avg,
                "max": level.token_max,
            },
            "expert": {
                "grade_mean": mean(level.expert_grade),
                "grade_std": sample_std(level.expert_grade),
                "grade_ci95_low": grade_interval[0],
                "grade_ci95_high": grade_interval[1],
                "grade_min": min(level.expert_grade),
                "grade_max": max(level.expert_grade),
                "reward_mean": mean(level.expert_reward),
                "reward_std": sample_std(level.expert_reward),
                "reward_min": min(level.expert_reward),
                "reward_max": max(level.expert_reward),
                "revenue_mean": mean(level.expert_revenue),
                "revenue_std": sample_std(level.expert_revenue),
                "security_mean": mean(level.expert_security),
                "caught_mean": mean(level.expert_caught),
                "caught_max": max(level.expert_caught),
                "episode_steps_mean": mean(level.expert_steps),
            },
            "optimization": {
                "train_steps": len(level.loss),
                "epochs": level.epoch[-1],
                "loss_start": start_loss,
                "loss_final": final_loss,
                "loss_reduction_pct": 100.0 * loss_delta / start_loss,
                "half_loss_step": half_step,
                "near_final_step": near_final_step,
                "grad_mean": mean(level.grad_norm),
                "grad_p95": to_percentile(level.grad_norm, 95),
                "grad_max": max(level.grad_norm),
                "learning_rate_peak": max(level.learning_rate),
            },
        }

    all_final_losses = [per_level[level]["optimization"]["loss_final"] for level in per_level]

    return {
        "levels": per_level,
        "global": {
            "level_count": len(per_level),
            "total_examples": total_examples,
            "approx_total_tokens": total_tokens,
            "total_train_steps": total_steps,
            "best_final_loss": min(all_final_losses),
            "worst_final_loss": max(all_final_losses),
        },
    }


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


def save_figure(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=260)
    plt.close()
    return path


def plot_curriculum_loss_overview(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    cursor = 0
    for level_name in LEVEL_ORDER:
        if level_name not in levels:
            continue
        level = levels[level_name]
        steps = np.arange(cursor, cursor + len(level.loss))
        smooth = rolling_mean(level.loss, max(7, len(level.loss) // 20))
        color = LEVEL_COLORS[level_name]

        ax.plot(steps, level.loss, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(steps, smooth, color=color, linewidth=2.3, label=LEVEL_LABELS[level_name])
        ax.axvspan(cursor, cursor + len(level.loss), color=color, alpha=0.04)
        midpoint = cursor + len(level.loss) / 2
        ax.text(midpoint, ax.get_ylim()[1] if ax.lines else max(level.loss), LEVEL_LABELS[level_name], color=color)
        cursor += len(level.loss)

    ax.set_title("Curriculum Training Loss Across All Five Levels")
    ax.set_xlabel("Global optimization step")
    ax.set_ylabel("SFT training loss")
    ax.grid(True, axis="both")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_xlim(left=0)

    global_stats = summary["global"]
    ax.text(
        0.01,
        0.97,
        (
            f"29k examples total | "
            f"~{global_stats['approx_total_tokens'] / 1_000_000:.1f}M tokens | "
            f"{global_stats['total_train_steps']} logged updates"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F7FA", "edgecolor": "#D7DCE2"},
    )

    save_figure("curriculum_loss_overview.png")


def plot_per_level_convergence(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()

    for idx, level_name in enumerate(LEVEL_ORDER):
        ax = axes[idx]
        if level_name not in levels:
            ax.axis("off")
            continue
        level = levels[level_name]
        stats = summary["levels"][level_name]["optimization"]
        x = np.asarray(level.epoch)
        y = np.asarray(level.loss)
        smooth = rolling_mean(level.loss, max(5, len(level.loss) // 18))
        color = LEVEL_COLORS[level_name]

        ax.plot(x, y, color=color, alpha=0.2, linewidth=1.0)
        ax.plot(x, smooth, color=color, linewidth=2.2)
        ax.axhline(stats["loss_final"], color="#6B7280", linewidth=1.0, linestyle=":")

        ax.set_title(f"{LEVEL_LABELS[level_name]} ({len(level.loss)} steps)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.text(
            0.04,
            0.96,
            (
                f"start={stats['loss_start']:.3f}\n"
                f"final={stats['loss_final']:.3f}\n"
                f"drop={stats['loss_reduction_pct']:.1f}%\n"
                f"half-loss step={stats['half_loss_step']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FBFCFE", "edgecolor": "#D7DCE2"},
        )

    axes[-1].axis("off")
    fig.suptitle("Per-Level Convergence Profiles", y=1.02)
    fig.tight_layout()
    save_figure("per_level_convergence.png")


def plot_expert_grade_distribution(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2.2, 1.2])

    names = [level for level in LEVEL_ORDER if level in levels]
    labels = [LEVEL_LABELS[name] for name in names]
    colors = [LEVEL_COLORS[name] for name in names]
    distributions = [levels[name].expert_grade for name in names]

    violin = axes[0].violinplot(distributions, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)

    box = axes[0].boxplot(distributions, patch_artist=True, widths=0.25)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(7)
    for idx, (values, color) in enumerate(zip(distributions, colors), start=1):
        x = rng.normal(idx, 0.04, size=len(values))
        axes[0].scatter(x, values, s=18, color=color, alpha=0.55, edgecolors="white", linewidths=0.4)

    axes[0].set_xticks(range(1, len(labels) + 1), labels)
    axes[0].set_ylabel("Composite expert grade")
    axes[0].set_title("Distribution of Expert Demonstration Quality")
    axes[0].grid(True, axis="y")

    means = [summary["levels"][name]["expert"]["grade_mean"] for name in names]
    lows = [summary["levels"][name]["expert"]["grade_ci95_low"] for name in names]
    highs = [summary["levels"][name]["expert"]["grade_ci95_high"] for name in names]
    lower_err = np.asarray(means) - np.asarray(lows)
    upper_err = np.asarray(highs) - np.asarray(means)

    axes[1].errorbar(
        labels,
        means,
        yerr=[lower_err, upper_err],
        fmt="o",
        color="#111827",
        ecolor="#4B5563",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
    )
    axes[1].bar(labels, means, color=colors, alpha=0.6)
    axes[1].set_ylabel("Mean grade")
    axes[1].set_ylim(0.4, 0.8)
    axes[1].set_title("Mean grade with 95% confidence interval")
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    save_figure("expert_grade_distribution.png")


def plot_expert_operational_metrics(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    names = [level for level in LEVEL_ORDER if level in levels]
    labels = [LEVEL_LABELS[name] for name in names]
    colors = [LEVEL_COLORS[name] for name in names]

    revenue_data = [levels[name].expert_revenue for name in names]
    revenue_box = axes[0, 0].boxplot(revenue_data, patch_artist=True)
    for patch, color in zip(revenue_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    axes[0, 0].set_xticks(range(1, len(labels) + 1), labels)
    axes[0, 0].set_title("Expert revenue distribution")
    axes[0, 0].set_ylabel("Episode revenue")
    axes[0, 0].grid(True, axis="y")

    security_means = [summary["levels"][name]["expert"]["security_mean"] for name in names]
    axes[0, 1].bar(labels, security_means, color=colors, alpha=0.8)
    for idx, value in enumerate(security_means):
        axes[0, 1].text(idx, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)
    axes[0, 1].set_title("Mean security retention")
    axes[0, 1].set_ylabel("Security score")
    axes[0, 1].set_ylim(0, max(105, max(security_means) + 10))
    axes[0, 1].grid(True, axis="y")

    caught_means = [summary["levels"][name]["expert"]["caught_mean"] for name in names]
    axes[1, 0].bar(labels, caught_means, color=colors, alpha=0.8)
    for idx, value in enumerate(caught_means):
        axes[1, 0].text(idx, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)
    axes[1, 0].set_title("Mean sleepers caught")
    axes[1, 0].set_ylabel("Caught sleepers")
    axes[1, 0].set_ylim(0, max(caught_means) + 0.4)
    axes[1, 0].grid(True, axis="y")

    for name in names:
        axes[1, 1].scatter(
            levels[name].expert_revenue,
            levels[name].expert_grade,
            s=36,
            alpha=0.75,
            color=LEVEL_COLORS[name],
            label=LEVEL_LABELS[name],
            edgecolors="white",
            linewidths=0.45,
        )
    axes[1, 1].set_title("Revenue-grade tradeoff across expert episodes")
    axes[1, 1].set_xlabel("Revenue")
    axes[1, 1].set_ylabel("Grade")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc="lower left", ncol=2)

    fig.tight_layout()
    save_figure("expert_operational_metrics.png")


def plot_expert_reward_progression(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2.0, 1.2])

    for level_name in LEVEL_ORDER:
        if level_name not in levels:
            continue
        level = levels[level_name]
        rewards = np.asarray(level.expert_reward, dtype=float)
        episodes = np.arange(1, len(rewards) + 1)
        smooth = rolling_mean(level.expert_reward, max(3, len(rewards) // 8))
        color = LEVEL_COLORS[level_name]

        axes[0].plot(episodes, rewards, color=color, alpha=0.25, linewidth=1.2)
        axes[0].scatter(episodes, rewards, color=color, alpha=0.45, s=22)
        axes[0].plot(episodes, smooth, color=color, linewidth=2.4, label=LEVEL_LABELS[level_name])

    axes[0].set_title("Training-time expert reward traces across curriculum episodes")
    axes[0].set_xlabel("Episode index within level")
    axes[0].set_ylabel("Episode reward")
    axes[0].grid(True, axis="both")
    axes[0].legend(ncol=3, loc="upper left")
    axes[0].text(
        0.01,
        0.97,
        "Fast proxy during training: these rewards come from expert-trajectory generation.\n"
        "Use full_evaluation.py for the official trained-model benchmark.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F7FA", "edgecolor": "#D7DCE2"},
    )

    labels = [LEVEL_LABELS[name] for name in LEVEL_ORDER if name in levels]
    colors = [LEVEL_COLORS[name] for name in LEVEL_ORDER if name in levels]
    reward_means = [summary["levels"][name]["expert"]["reward_mean"] for name in LEVEL_ORDER if name in levels]
    reward_stds = [summary["levels"][name]["expert"]["reward_std"] for name in LEVEL_ORDER if name in levels]
    x = np.arange(len(labels))
    axes[1].bar(x, reward_means, color=colors, alpha=0.8)
    axes[1].errorbar(x, reward_means, yerr=reward_stds, fmt="none", ecolor="#111827", capsize=4, linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Mean training-time reward by difficulty")
    axes[1].set_ylabel("Reward mean +/- std")
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    save_figure("expert_reward_progression.png")


def plot_optimization_diagnostics(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    cursor = 0
    all_gradients: list[float] = []
    for level_name in LEVEL_ORDER:
        if level_name not in levels:
            continue
        level = levels[level_name]
        steps = np.arange(cursor, cursor + len(level.grad_norm))
        color = LEVEL_COLORS[level_name]
        grad_smooth = rolling_mean(level.grad_norm, max(7, len(level.grad_norm) // 20))
        lr_scaled = np.asarray(level.learning_rate) * 1e5

        axes[0, 0].plot(steps, level.grad_norm, color=color, alpha=0.14, linewidth=0.9)
        axes[0, 0].plot(steps, grad_smooth, color=color, linewidth=2.0, label=LEVEL_LABELS[level_name])
        axes[0, 1].plot(steps, lr_scaled, color=color, linewidth=1.8, label=LEVEL_LABELS[level_name])
        all_gradients.extend(level.grad_norm)
        cursor += len(level.grad_norm)

    axes[0, 0].set_title("Gradient norm stability")
    axes[0, 0].set_xlabel("Global optimization step")
    axes[0, 0].set_ylabel("Gradient norm")
    axes[0, 0].grid(True)
    axes[0, 0].legend(ncol=3, loc="upper right")

    axes[0, 1].set_title("Learning rate schedule")
    axes[0, 1].set_xlabel("Global optimization step")
    axes[0, 1].set_ylabel("Learning rate (x1e-5)")
    axes[0, 1].grid(True)

    grad_p95 = [summary["levels"][name]["optimization"]["grad_p95"] for name in LEVEL_ORDER if name in levels]
    grad_mean = [summary["levels"][name]["optimization"]["grad_mean"] for name in LEVEL_ORDER if name in levels]
    labels = [LEVEL_LABELS[name] for name in LEVEL_ORDER if name in levels]
    colors = [LEVEL_COLORS[name] for name in LEVEL_ORDER if name in levels]

    x = np.arange(len(labels))
    width = 0.36
    axes[1, 0].bar(x - width / 2, grad_mean, width, color=colors, alpha=0.75, label="Mean")
    axes[1, 0].bar(x + width / 2, grad_p95, width, color="#374151", alpha=0.75, label="P95")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_title("Gradient statistics by level")
    axes[1, 0].set_ylabel("Gradient norm")
    axes[1, 0].grid(True, axis="y")
    axes[1, 0].legend()

    axes[1, 1].hist(all_gradients, bins=36, color="#4B5563", alpha=0.8, edgecolor="white")
    axes[1, 1].axvline(np.mean(all_gradients), color="#C13C37", linestyle="--", linewidth=1.4, label="Global mean")
    axes[1, 1].axvline(np.percentile(all_gradients, 95), color="#2F6BFF", linestyle=":", linewidth=1.4, label="Global P95")
    axes[1, 1].set_title("Aggregate gradient distribution")
    axes[1, 1].set_xlabel("Gradient norm")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, axis="y")
    axes[1, 1].legend()

    fig.tight_layout()
    save_figure("optimization_diagnostics.png")


def plot_dataset_scaling(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    names = [name for name in LEVEL_ORDER if name in levels]
    labels = [LEVEL_LABELS[name] for name in names]
    colors = [LEVEL_COLORS[name] for name in names]
    examples = [levels[name].examples for name in names]
    avg_tokens = [levels[name].token_avg for name in names]
    min_tokens = [levels[name].token_min for name in names]
    max_tokens = [levels[name].token_max for name in names]

    x = np.arange(len(labels))
    axes[0].bar(x, examples, color=colors, alpha=0.85)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Curriculum dataset growth")
    axes[0].set_ylabel("Supervised examples")
    axes[0].grid(True, axis="y")

    twin = axes[0].twinx()
    twin.plot(x, avg_tokens, color="#111827", marker="o", linewidth=2.0)
    twin.set_ylabel("Average tokens per example")
    twin.set_ylim(min(avg_tokens) - 10, max(avg_tokens) + 20)

    axes[1].fill_between(x, min_tokens, max_tokens, color="#CBD5E1", alpha=0.5, label="Token range")
    axes[1].plot(x, avg_tokens, color="#111827", marker="o", linewidth=2.1, label="Average length")
    axes[1].plot(x, min_tokens, color="#64748B", linestyle="--", linewidth=1.2, label="Minimum")
    axes[1].plot(x, max_tokens, color="#0F172A", linestyle=":", linewidth=1.2, label="Maximum")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Sequence length scaling by difficulty")
    axes[1].set_ylabel("Tokens")
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    save_figure("dataset_scaling.png")


def plot_curriculum_heatmap(levels: dict[str, LevelData], summary: dict[str, object]) -> None:
    setup_style()
    names = [name for name in LEVEL_ORDER if name in levels]
    stats = summary["levels"]

    metric_names = [
        "Examples",
        "Avg tokens",
        "Expert grade",
        "Grade std",
        "Reward",
        "Revenue",
        "Security",
        "Caught",
        "Loss reduction",
        "Final loss",
        "Half-loss step",
    ]

    raw_matrix = np.array(
        [
            [stats[name]["examples"] for name in names],
            [stats[name]["tokens"]["avg"] for name in names],
            [stats[name]["expert"]["grade_mean"] for name in names],
            [stats[name]["expert"]["grade_std"] for name in names],
            [stats[name]["expert"]["reward_mean"] for name in names],
            [stats[name]["expert"]["revenue_mean"] for name in names],
            [stats[name]["expert"]["security_mean"] for name in names],
            [stats[name]["expert"]["caught_mean"] for name in names],
            [stats[name]["optimization"]["loss_reduction_pct"] for name in names],
            [stats[name]["optimization"]["loss_final"] for name in names],
            [stats[name]["optimization"]["half_loss_step"] for name in names],
        ],
        dtype=float,
    )

    norm_matrix = np.zeros_like(raw_matrix)
    reverse_rows = {3, 9, 10}
    for row_idx in range(raw_matrix.shape[0]):
        row = raw_matrix[row_idx]
        lo = row.min()
        hi = row.max()
        if math.isclose(lo, hi):
            norm_matrix[row_idx] = 0.5
        else:
            normalized = (row - lo) / (hi - lo)
            norm_matrix[row_idx] = 1.0 - normalized if row_idx in reverse_rows else normalized

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    heatmap = ax.imshow(norm_matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(names)), [LEVEL_LABELS[name] for name in names])
    ax.set_yticks(np.arange(len(metric_names)), metric_names)
    ax.set_title("Normalized Curriculum Metric Heatmap")

    for row_idx in range(norm_matrix.shape[0]):
        for col_idx in range(norm_matrix.shape[1]):
            raw_value = raw_matrix[row_idx, col_idx]
            if row_idx in (2, 3, 4, 7, 8, 9):
                label = f"{raw_value:.2f}"
            elif row_idx == 10:
                label = f"{raw_value:.0f}"
            else:
                label = f"{raw_value:.0f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="#0F172A", fontsize=8.5)

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.024, pad=0.03)
    cbar.set_label("Within-metric normalized score")

    fig.tight_layout()
    save_figure("curriculum_heatmap.png")


def write_statistics_files(summary: dict[str, object]) -> None:
    json_path = OUTPUT_DIR / "training_statistics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Training Statistics",
        "",
        "| Level | Examples | Avg Tokens | Expert Grade (mean +/- std) | Reward Mean | Revenue Mean | Security Mean | Caught Mean | Loss Start -> Final | Loss Reduction | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for level_name in LEVEL_ORDER:
        if level_name not in summary["levels"]:
            continue
        stats = summary["levels"][level_name]
        lines.append(
            "| "
            f"{stats['label']} | "
            f"{stats['examples']} | "
            f"{stats['tokens']['avg']} | "
            f"{stats['expert']['grade_mean']:.3f} +/- {stats['expert']['grade_std']:.3f} | "
            f"{stats['expert']['reward_mean']:.2f} | "
            f"{stats['expert']['revenue_mean']:.1f} | "
            f"{stats['expert']['security_mean']:.1f} | "
            f"{stats['expert']['caught_mean']:.2f} | "
            f"{stats['optimization']['loss_start']:.3f} -> {stats['optimization']['loss_final']:.3f} | "
            f"{stats['optimization']['loss_reduction_pct']:.1f}% | "
            f"{stats['optimization']['train_steps']} |"
        )

    lines.extend(
        [
            "",
            f"Total examples: {summary['global']['total_examples']}",
            f"Approximate token budget: {summary['global']['approx_total_tokens'] / 1_000_000:.2f}M tokens",
            f"Total logged training updates: {summary['global']['total_train_steps']}",
            "",
            f"Source log: `{summary['source_log']}`",
        ]
    )

    (OUTPUT_DIR / "training_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    log_file = resolve_log_file()
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    levels = parse_levels(log_text)
    summary = build_summary(levels)
    summary["source_log"] = str(log_file)

    plot_curriculum_loss_overview(levels, summary)
    plot_per_level_convergence(levels, summary)
    plot_expert_grade_distribution(levels, summary)
    plot_expert_operational_metrics(levels, summary)
    plot_expert_reward_progression(levels, summary)
    plot_optimization_diagnostics(levels, summary)
    plot_dataset_scaling(levels, summary)
    plot_curriculum_heatmap(levels, summary)
    write_statistics_files(summary)

    print("Generated plot set:")
    for filename in [
        "curriculum_loss_overview.png",
        "per_level_convergence.png",
        "expert_grade_distribution.png",
        "expert_operational_metrics.png",
        "expert_reward_progression.png",
        "optimization_diagnostics.png",
        "dataset_scaling.png",
        "curriculum_heatmap.png",
        "training_statistics.json",
        "training_statistics.md",
    ]:
        print(f"  - {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
