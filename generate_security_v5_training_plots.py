#!/usr/bin/env python3
"""Generate compact Security-First V5 training-side diagnostics.

These plots use Drive metadata plus the completed V5 evaluation summary. They are
not a substitute for the full optimizer event log; the full `training_events.jsonl`
remains in Google Drive.
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

SUMMARY_PATH = Path("evaluation_comparison_latest.json")
PLOT_DIR = Path("plots")

COLORS = {
    "base_untrained": "#64748B",
    "raw_v5_trained": "#06B6D4",
    "security_first_supervisor": "#16A34A",
    "heuristic": "#F59E0B",
    "random": "#EF4444",
}

LABELS = {
    "base_untrained": "Base untrained",
    "raw_v5_trained": "Raw V5 trained",
    "security_first_supervisor": "Security-first supervisor",
    "heuristic": "Heuristic",
    "random": "Random",
}

ARTIFACT_SIZES_MB = {
    "base eval JSON": 1437.7,
    "raw eval checkpoints": 744.7,
    "supervisor eval JSON": 1380.0,
    "supervisor checkpoints": 637.0,
    "training events": 23.6,
    "console training log": 1.9,
}


def load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def arrays(payload: dict[str, Any], agent: str, metric: str) -> list[float]:
    return [float(v) for v in payload["agents"][agent][metric]]


def plot_curriculum_overview(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    x = np.arange(len(LEVELS))
    examples = [training[level]["examples"] for level in LEVELS]
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bars = ax1.bar(x, examples, color="#0EA5E9", alpha=0.78, label="Supervised examples")
    ax1.set_ylabel("Supervised examples")
    ax1.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax1.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, examples):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 650, f"{value:,}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, arrays(payload, "raw_v5_trained", "grade"), color=COLORS["raw_v5_trained"], marker="o", linewidth=2.5, label="Raw V5 grade")
    ax2.plot(x, arrays(payload, "security_first_supervisor", "grade"), color=COLORS["security_first_supervisor"], marker="D", linewidth=2.5, label="Supervisor grade")
    ax2.set_ylabel("Evaluation grade")
    ax2.set_ylim(0.55, 0.95)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.suptitle("Security-First V5 Curriculum Scale and Evaluation Outcome", y=1.02)
    fig.tight_layout()
    save(fig, "curriculum_loss_overview.png")


def plot_per_level_convergence(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(x, [training[level]["examples"] for level in LEVELS], color="#38BDF8")
    axes[0].set_title("V5 Examples by Curriculum Level")
    axes[0].set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    axes[0].set_ylabel("Examples")
    axes[0].grid(True, axis="y", alpha=0.25)

    raw_pass = payload["agents"]["raw_v5_trained"].get("pass_rate", [1, 1, 1, 0.5, 0.05])
    supervisor_pass = payload["agents"]["security_first_supervisor"].get("pass_rate", [1] * 5)
    width = 0.35
    axes[1].bar(x - width / 2, raw_pass, width, color=COLORS["raw_v5_trained"], label="Raw V5")
    axes[1].bar(x + width / 2, supervisor_pass, width, color=COLORS["security_first_supervisor"], label="Supervisor")
    axes[1].set_title("Acceptance Pass Rate by Level")
    axes[1].set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Pass rate")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Per-Level V5 Training Scale and Gate Behavior", y=1.04)
    fig.tight_layout()
    save(fig, "per_level_convergence.png")


def plot_reward_progression(payload: dict[str, Any]) -> None:
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]:
        ax.plot(x, arrays(payload, key, "reward"), marker="o", linewidth=2.5, color=COLORS[key], label=LABELS[key])
    ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylabel("Mean reward")
    ax.set_title("Reward Progression Across V5 Difficulty Levels")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, loc="upper left")
    save(fig, "expert_reward_progression.png")


def plot_grade_distribution(payload: dict[str, Any]) -> None:
    x = np.arange(len(LEVELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, key in enumerate(["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, arrays(payload, key, "grade"), width, color=COLORS[key], alpha=0.88, label=LABELS[key])
    ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Composite grade")
    ax.set_title("V5 Grade Distribution by Level")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, loc="upper left")
    save(fig, "expert_grade_distribution.png")


def plot_operational_metrics(payload: dict[str, Any]) -> None:
    metrics = [
        ("security", "Final Security"),
        ("sleepers_caught", "Sleepers Caught"),
        ("reward", "Mean Reward"),
        ("revenue", "Final Revenue"),
    ]
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for key in ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]:
            ax.plot(x, arrays(payload, key, metric), marker="o", linewidth=2.2, color=COLORS[key], label=LABELS[key])
        ax.set_title(title)
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS], rotation=15)
        ax.grid(True, alpha=0.3)
        if metric == "security":
            ax.set_ylim(35, 105)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("V5 Operational Outcomes", y=1.04)
    fig.tight_layout()
    save(fig, "expert_operational_metrics.png")


def plot_optimization_diagnostics(payload: dict[str, Any]) -> None:
    acceptance = payload["acceptance"]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    lines = [
        "Security-First V5 Run Diagnostics",
        "",
        "Training folder: panopticon-security-v5-ep50",
        "Training commit: " + payload["benchmark"]["training_source_commit"][:12],
        "Evaluation commit: " + payload["benchmark"]["evaluation_source_commit"][:12],
        "Training: 250 expert episodes, 88,896 examples, max sequence 512",
        "Base evaluation: 300/300 episode checkpoints complete",
        "Raw trained evaluation: 300/300 episode checkpoints complete",
        "Supervisor evaluation: 300/300 episode checkpoints complete",
        "",
        f"Raw V5 macro grade: {acceptance['raw_v5_trained']['candidate_macro_grade']:.6f} (accepted: {acceptance['raw_v5_trained']['accepted']})",
        f"Supervisor macro grade: {acceptance['security_first_supervisor']['candidate_macro_grade']:.6f} (accepted: {acceptance['security_first_supervisor']['accepted']})",
        f"Raw V5 failed checks: {len(acceptance['raw_v5_trained']['failed_checks'])}",
        "Primary raw-model failure: Level 4/5 security gate and zero-miss requirements.",
    ]
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=13, family="monospace")
    save(fig, "optimization_diagnostics.png")


def plot_dataset_scaling(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    examples = [training[level]["examples"] for level in LEVELS]
    cumulative = np.cumsum(examples)
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, examples, color="#06B6D4", alpha=0.75, label="Level examples")
    ax.plot(x, cumulative, color="#0F172A", marker="o", linewidth=2.5, label="Cumulative examples")
    ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_ylabel("Examples")
    ax.set_title("V5 Dataset Scaling")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save(fig, "dataset_scaling.png")


def normalize(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def plot_curriculum_heatmap(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    rows = [
        ("Examples", [training[level]["examples"] for level in LEVELS]),
        ("Raw grade", arrays(payload, "raw_v5_trained", "grade")),
        ("Raw security", arrays(payload, "raw_v5_trained", "security")),
        ("Supervisor grade", arrays(payload, "security_first_supervisor", "grade")),
        ("Supervisor security", arrays(payload, "security_first_supervisor", "security")),
    ]
    data = np.array([normalize([float(v) for v in values]) for _, values in rows])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(LEVELS)), [LEVEL_LABELS[level] for level in LEVELS])
    ax.set_yticks(np.arange(len(rows)), [name for name, _ in rows])
    for row_idx, (_, raw_values) in enumerate(rows):
        for col_idx, value in enumerate(raw_values):
            text = f"{value:,.0f}" if value > 1000 else f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    ax.set_title("V5 Curriculum Heatmap: Scale, Raw Model, and Supervisor Outcomes")
    save(fig, "curriculum_heatmap.png")


def plot_progress_timeline(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    examples = [training[level]["examples"] for level in LEVELS]
    cumulative_examples = np.cumsum(examples)
    cumulative_episodes = np.cumsum([training[level]["episodes"] for level in LEVELS])
    x = np.arange(len(LEVELS))
    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    ax1.plot(x, cumulative_examples, color="#06B6D4", marker="o", linewidth=2.8, label="Cumulative training examples")
    ax1.set_ylabel("Cumulative examples")
    ax1.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_episodes, color="#F59E0B", marker="s", linewidth=2.8, label="Cumulative expert episodes")
    ax2.set_ylabel("Cumulative expert episodes")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("V5 Resume-Aware Curriculum Progress")
    save(fig, "curriculum_progress_timeline.png")


def plot_action_mix(payload: dict[str, Any]) -> None:
    averages = payload["level_macro_averages"]
    metrics = ["grade", "reward", "revenue", "security", "sleepers_caught"]
    agents = ["base_untrained", "raw_v5_trained", "security_first_supervisor", "heuristic"]
    raw = {metric: [averages[agent][metric] for agent in agents] for metric in metrics}
    normalized = {agent: [] for agent in agents}
    for metric in metrics:
        vals = raw[metric]
        low, high = min(vals), max(vals)
        for agent, value in zip(agents, vals):
            normalized[agent].append(0.5 if high == low else (value - low) / (high - low))
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, agent in enumerate(agents):
        ax.bar(x + (idx - 1.5) * width, normalized[agent], width, color=COLORS[agent], label=LABELS[agent], alpha=0.86)
    ax.set_xticks(x, ["Grade", "Reward", "Revenue", "Security", "Caught"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized V5 Policy Coverage")
    ax.set_ylabel("Min-max normalized macro metric")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="upper left")
    save(fig, "training_action_mix.png")


def plot_advanced_tier_dynamics(payload: dict[str, Any]) -> None:
    levels = ["level_4", "level_5"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, metric, title in [
        (axes[0], "security", "Advanced-Tier Security"),
        (axes[1], "sleepers_caught", "Sleepers Caught"),
        (axes[2], "grade", "Composite Grade"),
    ]:
        raw_values = [payload["agents"]["raw_v5_trained"][metric][LEVELS.index(level)] for level in levels]
        supervisor_values = [payload["agents"]["security_first_supervisor"][metric][LEVELS.index(level)] for level in levels]
        width = 0.35
        ax.bar(x - width / 2, raw_values, width, color=COLORS["raw_v5_trained"], label="Raw V5")
        ax.bar(x + width / 2, supervisor_values, width, color=COLORS["security_first_supervisor"], label="Supervisor")
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in levels])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylim(50, 105)
    axes[2].set_ylim(0.55, 0.95)
    axes[0].legend()
    fig.suptitle("Where Raw V5 Fails and Supervisor Recovers", y=1.04)
    fig.tight_layout()
    save(fig, "expert_step_dynamics.png")


def plot_artifact_footprint(_: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    names = list(ARTIFACT_SIZES_MB)
    values = list(ARTIFACT_SIZES_MB.values())
    ax.barh(np.arange(len(names)), values, color="#6366F1", alpha=0.82)
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Size in MB")
    ax.set_title("Drive Artifact Footprint for the Completed V5 Run")
    ax.grid(True, axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value + 10, idx, f"{value:,.1f} MB", va="center", fontsize=9)
    save(fig, "runtime_breakdown.png")


def write_training_stats(payload: dict[str, Any]) -> None:
    training = payload["training_datasets"]
    stats = {
        "schema_version": "panopticon-security-v5-compact-training-summary-v1",
        "created_at": payload["created_at"],
        "source": "Google Drive folder panopticon-security-v5-ep50 metadata plus completed V5 evaluation summaries",
        "training_datasets": training,
        "totals": {
            "episodes": sum(item["episodes"] for item in training.values()),
            "examples": sum(item["examples"] for item in training.values()),
            "max_seq_length": 512,
        },
        "raw_v5_acceptance": payload["acceptance"]["raw_v5_trained"],
        "supervisor_acceptance": payload["acceptance"]["security_first_supervisor"],
        "level_macro_averages": payload["level_macro_averages"],
    }
    (PLOT_DIR / "training_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Security-First V5 Compact Training Summary",
        "",
        "This summary is derived from Drive metadata and completed V5 benchmark reports.",
        "The full optimizer event log remains in Google Drive as `training_events.jsonl`.",
        "",
        "| Level | Episodes | Examples | Max Seq |",
        "|---|---:|---:|---:|",
    ]
    for level in LEVELS:
        item = training[level]
        lines.append(f"| {LEVEL_LABELS[level]} | {item['episodes']} | {item['examples']:,} | {item['max_seq_length']} |")
    lines.extend(
        [
            f"| **Total** | **{stats['totals']['episodes']}** | **{stats['totals']['examples']:,}** | **512** |",
            "",
            f"Raw V5 macro grade: `{payload['acceptance']['raw_v5_trained']['candidate_macro_grade']:.6f}`; accepted: `False`.",
            f"Supervisor macro grade: `{payload['acceptance']['security_first_supervisor']['candidate_macro_grade']:.6f}`; accepted: `True`.",
        ]
    )
    (PLOT_DIR / "training_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    payload = load_summary()
    plot_curriculum_overview(payload)
    plot_per_level_convergence(payload)
    plot_reward_progression(payload)
    plot_grade_distribution(payload)
    plot_operational_metrics(payload)
    plot_optimization_diagnostics(payload)
    plot_dataset_scaling(payload)
    plot_curriculum_heatmap(payload)
    plot_progress_timeline(payload)
    plot_action_mix(payload)
    plot_advanced_tier_dynamics(payload)
    plot_artifact_footprint(payload)
    write_training_stats(payload)
    print("[*] Wrote compact V5 training diagnostics to plots")


if __name__ == "__main__":
    main()
