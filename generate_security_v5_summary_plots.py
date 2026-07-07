#!/usr/bin/env python3
"""Write the latest Security-First V5 benchmark summary and plots.

The full Drive evaluation JSON files are multi-GB artifacts, so this script keeps
the checked-in repository lightweight by storing the verified level summaries and
acceptance-gate outcomes that were read from the completed Drive run.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_evaluation_plots import LEVEL_LABELS, LEVELS, setup_style

OUTPUT_JSON = Path("evaluation_comparison_latest.json")
PLOT_DIR = Path("plots")

AGENT_ORDER = [
    "base_untrained",
    "raw_v5_trained",
    "security_first_supervisor",
    "heuristic",
    "random",
]

AGENT_STYLE = {
    "base_untrained": ("Base untrained", "#64748B", "--", "o"),
    "raw_v5_trained": ("Raw V5 trained", "#06B6D4", "-", "o"),
    "security_first_supervisor": ("Security-first supervisor", "#16A34A", "-", "D"),
    "heuristic": ("Heuristic", "#F59E0B", ":", "s"),
    "random": ("Random", "#EF4444", "-.", "^"),
}

AGENTS: dict[str, dict[str, Any]] = {
    "base_untrained": {
        "label": "Base untrained",
        "role": "Untrained Qwen/Qwen2.5-1.5B-Instruct evaluated in the trained slot.",
        "grade": [0.630, 0.671, 0.616, 0.657, 0.631],
        "grade_std": [0.042, 0.018, 0.000, 0.051, 0.053],
        "reward": [1.87, 4.21, 8.57, 11.36, 12.21],
        "revenue": [236.1, 313.9, 433.9, 574.3, 682.6],
        "security": [100.0, 100.0, 100.0, 95.0, 84.8],
        "sleepers_caught": [1.0, 2.0, 3.0, 3.85, 4.5],
    },
    "raw_v5_trained": {
        "label": "Raw V5 trained",
        "role": "Merged V5 LoRA model, no deterministic security supervisor.",
        "grade": [0.728, 0.731, 0.671, 0.722, 0.656],
        "grade_std": [0.011, 0.000, 0.000, 0.040, 0.044],
        "reward": [3.30, 6.47, 10.66, 12.32, 8.27],
        "revenue": [230.7, 337.3, 476.7, 640.8, 734.0],
        "security": [100.0, 100.0, 100.0, 85.85, 60.47],
        "sleepers_caught": [1.0, 2.0, 3.0, 3.5, 3.9],
        "pass_rate": [1.0, 1.0, 1.0, 0.5, 0.05],
    },
    "security_first_supervisor": {
        "label": "Security-first supervisor",
        "role": "Deterministic security-first policy evaluated in the trained slot.",
        "grade": [0.720045, 0.735125, 0.679445, 0.900675, 0.917065],
        "grade_std": [0.004, 0.005, 0.006, 0.003, 0.005],
        "reward": [5.67, 8.57, 12.95, 18.43, 71.02],
        "revenue": [308.2, 405.2, 550.8, 736.4, 857.3],
        "security": [100.0, 100.0, 100.0, 100.0, 100.0],
        "sleepers_caught": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pass_rate": [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "heuristic": {
        "label": "Heuristic",
        "role": "Legacy deterministic heuristic baseline.",
        "grade": [0.725, 0.727, 0.680, 0.689, 0.626],
        "grade_std": [0.006, 0.007, 0.012, 0.018, 0.046],
        "reward": [6.10, 9.99, 14.43, 13.30, 8.81],
        "revenue": [324.8, 456.2, 612.9, 771.0, 896.9],
        "security": [100.0, 100.0, 100.0, 73.4, 44.3],
        "sleepers_caught": [1.0, 2.0, 3.0, 3.05, 3.30],
    },
    "random": {
        "label": "Random",
        "role": "Random action baseline.",
        "grade": [0.631, 0.666, 0.650, 0.636, 0.654],
        "grade_std": [0.080, 0.074, 0.089, 0.105, 0.076],
        "reward": [-14.95, -24.19, -32.55, -47.70, -6.82],
        "revenue": [122.0, 173.8, 208.7, 234.6, 343.4],
        "security": [92.4, 83.5, 66.1, 59.3, 45.0],
        "sleepers_caught": [0.90, 1.85, 2.75, 3.70, 4.55],
    },
}

TRAINING_DATASETS = {
    "easy": {"episodes": 50, "examples": 7430, "max_seq_length": 512},
    "medium": {"episodes": 50, "examples": 13166, "max_seq_length": 512},
    "hard": {"episodes": 50, "examples": 18414, "max_seq_length": 512},
    "level_4": {"episodes": 50, "examples": 23889, "max_seq_length": 512},
    "level_5": {"episodes": 50, "examples": 25997, "max_seq_length": 512},
}

GRADE_MACROS = {
    "base_untrained": 0.64111,
    "raw_v5_trained": 0.701627,
    "security_first_supervisor": 0.7904709999999999,
    "heuristic": mean(AGENTS["heuristic"]["grade"]),
    "random": mean(AGENTS["random"]["grade"]),
}

FAILED_RAW_CHECKS = [
    {"name": "level_4.pass_rate_100pct", "actual": 0.5, "required": 1.0},
    {"name": "level_4.security_not_worse", "actual": 85.85, "required": ">= 95.05"},
    {"name": "level_4.caught_not_worse", "actual": 3.5, "required": ">= 3.85"},
    {"name": "level_4.zero_missed", "actual": 0.5, "required": 0.0},
    {"name": "level_5.pass_rate_100pct", "actual": 0.05, "required": 1.0},
    {"name": "level_5.security_not_worse", "actual": 60.47, "required": ">= 84.81"},
    {"name": "level_5.caught_not_worse", "actual": 3.9, "required": ">= 4.5"},
    {"name": "level_5.zero_missed", "actual": 1.1, "required": 0.0},
    {"name": "level_5.zero_false_accusations", "actual": 0.1, "required": 0.0},
]


def macro(agent_key: str, metric: str) -> float:
    if metric == "grade":
        return float(GRADE_MACROS[agent_key])
    return float(mean(AGENTS[agent_key][metric]))


def diff(lhs: str, rhs: str) -> dict[str, float]:
    return {
        metric: round(macro(lhs, metric) - macro(rhs, metric), 6)
        for metric in ["grade", "reward", "revenue", "security", "sleepers_caught"]
    }


def build_payload() -> dict[str, Any]:
    return {
        "schema_version": "panopticon-security-v5-evaluation-comparison-v1",
        "created_at": "2026-07-07",
        "drive_folder": {
            "name": "panopticon-security-v5-ep50",
            "id": "1OXQgw1OA-cAAXxLp9t8YbOsjv9dIBB_-",
            "url": "https://drive.google.com/drive/folders/1OXQgw1OA-cAAXxLp9t8YbOsjv9dIBB_-",
        },
        "benchmark": {
            "levels": LEVELS,
            "episodes_per_agent_level": 20,
            "completed_episode_checkpoints_per_run": 300,
            "seed": 42,
            "reward_schema_version": "security-first-v2",
            "grader_schema_version": "security-gated-v2",
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "trained_model": "/content/drive/MyDrive/panopticon-security-v5-ep50/merged_model",
            "training_episodes_per_level": 50,
            "training_total_episodes": sum(item["episodes"] for item in TRAINING_DATASETS.values()),
            "training_total_examples": sum(item["examples"] for item in TRAINING_DATASETS.values()),
            "training_source_commit": "4c1f2db6a7a3d3c88136d5b51fbcdfe4058ad818",
            "evaluation_source_commit": "58572637f2ee8bf78406380e286dd86f50455027",
            "note": (
                "Values are transcribed from completed Drive console logs, progress JSONs, "
                "and acceptance reports. The raw evaluation JSON artifacts are multi-GB."
            ),
        },
        "training_datasets": TRAINING_DATASETS,
        "agents": AGENTS,
        "level_macro_averages": {
            agent_key: {
                "grade": macro(agent_key, "grade"),
                "reward": macro(agent_key, "reward"),
                "revenue": macro(agent_key, "revenue"),
                "security": macro(agent_key, "security"),
                "sleepers_caught": macro(agent_key, "sleepers_caught"),
            }
            for agent_key in AGENT_ORDER
        },
        "comparisons": {
            "raw_v5_minus_base": diff("raw_v5_trained", "base_untrained"),
            "raw_v5_minus_heuristic": diff("raw_v5_trained", "heuristic"),
            "supervisor_minus_base": diff("security_first_supervisor", "base_untrained"),
            "supervisor_minus_raw_v5": diff("security_first_supervisor", "raw_v5_trained"),
            "supervisor_minus_heuristic": diff("security_first_supervisor", "heuristic"),
        },
        "acceptance": {
            "raw_v5_trained": {
                "accepted": False,
                "base_macro_grade": 0.64111,
                "candidate_macro_grade": 0.701627,
                "failed_checks": FAILED_RAW_CHECKS,
            },
            "security_first_supervisor": {
                "accepted": True,
                "base_macro_grade": 0.64111,
                "candidate_macro_grade": 0.7904709999999999,
                "failed_checks": [],
                "note": (
                    "This validates a deterministic security-first supervisor, not the raw "
                    "fine-tuned model by itself."
                ),
            },
        },
    }


def write_summary() -> dict[str, Any]:
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_direct_comparison(payload: dict[str, Any]) -> None:
    agents = payload["agents"]
    averages = payload["level_macro_averages"]
    x = np.arange(len(LEVELS))
    labels = [LEVEL_LABELS[level] for level in LEVELS]
    metrics = [
        ("grade", "Composite Grade"),
        ("reward", "Episode Reward"),
        ("revenue", "Final Revenue"),
        ("security", "Final Security"),
        ("sleepers_caught", "Sleepers Caught"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    flat = axes.flatten()
    for ax, (metric, title) in zip(flat, metrics):
        for key in ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]:
            label, color, linestyle, marker = AGENT_STYLE[key]
            ax.plot(
                x,
                agents[key][metric],
                marker=marker,
                linewidth=2.5,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        ax.set_xticks(x, labels, rotation=15)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.35)
        if metric == "security":
            ax.set_ylim(35, 105)
        if metric == "grade":
            ax.set_ylim(0.55, 0.95)

    summary = flat[5]
    summary.axis("off")
    lines = [
        "Security-First V5 benchmark verdict",
        "",
        f"Base macro grade: {averages['base_untrained']['grade']:.5f}",
        f"Raw V5 macro grade: {averages['raw_v5_trained']['grade']:.6f}",
        f"Supervisor macro grade: {averages['security_first_supervisor']['grade']:.6f}",
        f"Heuristic macro grade: {averages['heuristic']['grade']:.4f}",
        "",
        "Raw V5: improved grade, failed acceptance.",
        "Supervisor: accepted, all gate checks passed.",
        "",
        "Key raw V5 failures:",
        "  Level 4 pass rate: 50%",
        "  Level 5 pass rate: 5%",
        "  Level 5 security: 60.47 vs base 84.81",
    ]
    summary.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    handles, legend_labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Latest Security-First V5 benchmark summary", y=1.02)
    fig.tight_layout()
    save(fig, "base_vs_fixed_comparison.png")


def plot_summary_table(payload: dict[str, Any]) -> None:
    averages = payload["level_macro_averages"]
    rows = []
    for key in ["security_first_supervisor", "raw_v5_trained", "base_untrained", "heuristic", "random"]:
        row = averages[key]
        accepted = ""
        if key == "raw_v5_trained":
            accepted = "No"
        if key == "security_first_supervisor":
            accepted = "Yes"
        rows.append(
            [
                AGENT_STYLE[key][0],
                f"{row['grade']:.4f}",
                f"{row['reward']:.2f}",
                f"{row['revenue']:.1f}",
                f"{row['security']:.1f}",
                f"{row['sleepers_caught']:.2f}",
                accepted,
            ]
        )
    columns = ["Agent", "Macro Grade", "Reward", "Revenue", "Security", "Caught", "Accepted"]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis("off")
    ax.set_title("Security-First V5 Benchmark Scoreboard", pad=20, fontsize=18, weight="bold")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_text_props(weight="bold")
        if row == 1:
            cell.set_facecolor("#DCFCE7")
        elif row == 2:
            cell.set_facecolor("#E0F2FE")
        elif row > 0:
            cell.set_facecolor("#F8FAFC")
    ax.text(
        0.0,
        -0.08,
        "Raw V5 improves over the base model but fails the strict advanced-tier security gate. "
        "The supervisor diagnostic passes all checks.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
    )
    save(fig, "benchmark_summary_table.png")


def plot_comparison_grades(payload: dict[str, Any]) -> None:
    x = np.arange(len(LEVELS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, key in enumerate(AGENT_ORDER):
        label, color, _, _ = AGENT_STYLE[key]
        offset = (idx - (len(AGENT_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            payload["agents"][key]["grade"],
            width,
            yerr=payload["agents"][key].get("grade_std"),
            label=label,
            color=color,
            alpha=0.88,
            capsize=3,
        )
    ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Composite grade")
    ax.set_title("Composite Grade by Agent and Difficulty")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="upper left")
    save(fig, "comparison_grades.png")


def plot_comparison_operations(payload: dict[str, Any]) -> None:
    metrics = [
        ("reward", "Reward"),
        ("revenue", "Revenue"),
        ("security", "Security"),
        ("sleepers_caught", "Sleepers Caught"),
    ]
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for key in ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic", "random"]:
            label, color, linestyle, marker = AGENT_STYLE[key]
            ax.plot(
                x,
                payload["agents"][key][metric],
                marker=marker,
                linewidth=2.2,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        ax.set_title(title)
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS], rotation=15)
        ax.grid(True, alpha=0.3)
        if metric == "security":
            ax.set_ylim(35, 105)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Operational Metrics Across the Matched Benchmark", y=1.04)
    fig.tight_layout()
    save(fig, "comparison_operations.png")


def plot_radar(payload: dict[str, Any]) -> None:
    metrics = ["grade", "reward", "revenue", "security", "sleepers_caught"]
    metric_labels = ["Grade", "Reward", "Revenue", "Security", "Caught"]
    averages = payload["level_macro_averages"]
    raw_values = {metric: [averages[key][metric] for key in AGENT_ORDER] for metric in metrics}
    normalized: dict[str, list[float]] = {key: [] for key in AGENT_ORDER}
    for metric in metrics:
        vals = raw_values[metric]
        low, high = min(vals), max(vals)
        for key, value in zip(AGENT_ORDER, vals):
            normalized[key].append(0.5 if high == low else (value - low) / (high - low))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for key in ["security_first_supervisor", "raw_v5_trained", "base_untrained", "heuristic", "random"]:
        label, color, linestyle, _ = AGENT_STYLE[key]
        values = normalized[key] + normalized[key][:1]
        ax.plot(angles, values, color=color, linewidth=2.5, linestyle=linestyle, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], metric_labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized Balanced-Performance Radar", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    save(fig, "comparison_radar.png")


def plot_reward_distributions(payload: dict[str, Any]) -> None:
    x = np.arange(len(LEVELS))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for idx, key in enumerate(["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]):
        label, color, _, _ = AGENT_STYLE[key]
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, payload["agents"][key]["reward"], width, label=label, color=color, alpha=0.88)
        axes[1].bar(x + offset, payload["agents"][key]["grade"], width, label=label, color=color, alpha=0.88)
    for ax, title, ylabel in [
        (axes[0], "Mean Reward by Difficulty", "Reward"),
        (axes[1], "Mean Grade by Difficulty", "Composite grade"),
    ]:
        ax.set_title(title)
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_ylim(0.55, 0.95)
    axes[0].legend(ncol=2, loc="upper left")
    fig.suptitle("Reward and Grade Summary from Completed V5 Evaluations", y=1.03)
    fig.tight_layout()
    save(fig, "reward_distributions.png")


def plot_reward_frontier(payload: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    for key in AGENT_ORDER:
        label, color, _, marker = AGENT_STYLE[key]
        security = payload["agents"][key]["security"]
        reward = payload["agents"][key]["reward"]
        revenue = payload["agents"][key]["revenue"]
        sizes = [max(60, value / 4) for value in revenue]
        ax.scatter(security, reward, s=sizes, color=color, marker=marker, alpha=0.78, label=label)
        for level, sx, ry in zip(LEVELS, security, reward):
            ax.text(sx + 0.6, ry + 0.4, LEVEL_LABELS[level].replace("Level ", "L"), fontsize=8, color=color)
    ax.axvline(90, color="#334155", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(90.5, ax.get_ylim()[1] - 5, "security gate", color="#334155")
    ax.set_xlabel("Final security")
    ax.set_ylabel("Episode reward")
    ax.set_title("Reward-Security Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    save(fig, "reward_frontier.png")


def plot_reward_turn_dynamics(payload: dict[str, Any]) -> None:
    x = np.arange(len(LEVELS))
    labels = [LEVEL_LABELS[level] for level in LEVELS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for key in ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]:
        label, color, linestyle, marker = AGENT_STYLE[key]
        axes[0].plot(x, payload["agents"][key]["reward"], marker=marker, color=color, linestyle=linestyle, linewidth=2.5, label=label)
        axes[1].plot(x, payload["agents"][key]["security"], marker=marker, color=color, linestyle=linestyle, linewidth=2.5, label=label)
    axes[0].set_title("Reward Response as Difficulty Escalates")
    axes[1].set_title("Security Retention as Difficulty Escalates")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(True, alpha=0.3)
    axes[1].set_ylim(35, 105)
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Security")
    axes[0].legend(ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "reward_turn_dynamics.png")


def plot_scenario_timeline(payload: dict[str, Any]) -> None:
    metrics = ["grade", "reward", "security", "sleepers_caught"]
    titles = ["Grade", "Reward", "Security", "Sleepers Caught"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        data = np.array([payload["agents"][key][metric] for key in AGENT_ORDER])
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(LEVELS)), [LEVEL_LABELS[level] for level in LEVELS], rotation=20)
        ax.set_yticks(np.arange(len(AGENT_ORDER)), [AGENT_STYLE[key][0] for key in AGENT_ORDER])
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                ax.text(col, row, f"{data[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Scenario-Level Outcome Panorama", y=1.02)
    fig.tight_layout()
    save(fig, "scenario_timeline.png")


def main() -> None:
    setup_style()
    payload = write_summary()
    plot_direct_comparison(payload)
    plot_summary_table(payload)
    plot_comparison_grades(payload)
    plot_comparison_operations(payload)
    plot_radar(payload)
    plot_reward_distributions(payload)
    plot_reward_frontier(payload)
    plot_reward_turn_dynamics(payload)
    plot_scenario_timeline(payload)
    print(f"[*] Wrote {OUTPUT_JSON}")
    print(f"[*] Wrote summary plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()

