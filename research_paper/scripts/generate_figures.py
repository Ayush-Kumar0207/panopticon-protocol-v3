"""Generate publication-specific figures from the compact V5 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research_paper"
SOURCE = ROOT / "evaluation_comparison_latest.json"
FIGURES = PACKAGE / "assets" / "figures"

LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
LEVEL_LABELS = ["Easy", "Medium", "Hard", "Level 4", "Level 5"]
AGENTS = [
    "base_untrained",
    "raw_v5_trained",
    "security_first_supervisor",
    "heuristic",
    "random",
]
SHORT = {
    "base_untrained": "Base",
    "raw_v5_trained": "Raw V5",
    "security_first_supervisor": "Supervisor",
    "heuristic": "Heuristic",
    "random": "Random",
}
COLORS = {
    "base_untrained": "#6B7280",
    "raw_v5_trained": "#2563EB",
    "security_first_supervisor": "#059669",
    "heuristic": "#D97706",
    "random": "#7C3AED",
}
MARKERS = {
    "base_untrained": "o",
    "raw_v5_trained": "s",
    "security_first_supervisor": "D",
    "heuristic": "^",
    "random": "X",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def load() -> dict:
    with SOURCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def macro_grade(data: dict) -> None:
    values = [data["level_macro_averages"][a]["grade"] for a in AGENTS]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(AGENTS))
    bars = ax.bar(x, values, color=[COLORS[a] for a in AGENTS], width=0.68)
    base = data["level_macro_averages"]["base_untrained"]["grade"]
    ax.axhline(base, color=COLORS["base_untrained"], linestyle="--", linewidth=1.2, label="Base macro grade")
    ax.set_xticks(x, [SHORT[a] for a in AGENTS])
    ax.set_ylim(0.55, 0.84)
    ax.set_ylabel("Level-macro composite grade")
    ax.set_title("Average grade and release-gate outcome are different signals")
    for bar, value, agent in zip(bars, values, AGENTS):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom")
        status = data.get("acceptance", {}).get(agent, {}).get("accepted")
        if status is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.565,
                "gate: pass" if status else "gate: fail",
                ha="center",
                va="bottom",
                color="#047857" if status else "#B91C1C",
                fontsize=8,
                fontweight="bold",
            )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "macro_grade")


def per_level_grade(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(LEVELS))
    for agent in AGENTS:
        record = data["agents"][agent]
        ax.errorbar(
            x,
            record["grade"],
            yerr=record["grade_std"],
            label=SHORT[agent],
            color=COLORS[agent],
            marker=MARKERS[agent],
            linewidth=1.8,
            markersize=5,
            capsize=2,
        )
    ax.set_xticks(x, LEVEL_LABELS)
    ax.set_ylabel("Composite grade (mean ± stored SD)")
    ax.set_title("Policy grade by difficulty level (20 episodes per cell)")
    ax.legend(ncol=3, frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "per_level_grade")


def tradeoff(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for agent in AGENTS:
        macro = data["level_macro_averages"][agent]
        ax.scatter(
            macro["revenue"],
            macro["security"],
            s=90 + 320 * macro["grade"],
            color=COLORS[agent],
            marker=MARKERS[agent],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(SHORT[agent], (macro["revenue"], macro["security"]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(90, color="#B91C1C", linestyle="--", linewidth=1, label="Advanced minimum security")
    ax.set_xlabel("Level-macro final revenue")
    ax.set_ylabel("Level-macro final security")
    ax.set_title("Security–revenue trade-off (marker area tracks grade)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "security_revenue_tradeoff")


def advanced_pass(data: dict) -> None:
    agents = ["raw_v5_trained", "security_first_supervisor"]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for offset, agent in enumerate(agents):
        record = data["agents"][agent]
        values = [record["pass_rate"][3], record["pass_rate"][4]]
        bars = ax.bar(x + (offset - 0.5) * width, values, width, color=COLORS[agent], label=SHORT[agent])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100*value:.0f}%", ha="center")
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="Required 100%")
    ax.set_xticks(x, ["Level 4", "Level 5"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Episode pass rate")
    ax.set_title("Advanced operational pass rate")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    save(fig, "advanced_pass_rate")


def training_data(data: dict) -> None:
    datasets = data["training_datasets"]
    examples = [datasets[level]["examples"] for level in LEVELS]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(LEVELS))
    bars = ax.bar(x, examples, color=["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#2563EB"])
    ax.set_xticks(x, LEVEL_LABELS)
    ax.set_ylabel("Weighted training examples")
    ax.set_title("V5 curriculum data increases with horizon and difficulty")
    for bar, value in zip(bars, examples):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 500, f"{value:,}", ha="center", fontsize=8)
    ax.set_ylim(0, max(examples) * 1.16)
    fig.tight_layout()
    save(fig, "training_data_by_level")


def main() -> None:
    configure()
    data = load()
    macro_grade(data)
    per_level_grade(data)
    tradeoff(data)
    advanced_pass(data)
    training_data(data)
    print("Generated 5 paper figures in PNG and PDF formats")


if __name__ == "__main__":
    main()
