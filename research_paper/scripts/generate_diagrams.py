"""Generate vector-friendly architecture and research-to-product diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_paper" / "assets" / "figures"


def box(ax, x, y, w, h, title, detail, color="#E8F0FE", edge="#1D4ED8"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#111827")
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=7.6, color="#374151", linespacing=1.2)


def arrow(ax, start, end, label=None, color="#4B5563", bend=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color=color,
        connectionstyle=f"arc3,rad={bend}",
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + (0.035 if bend >= 0 else -0.035)
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.2, color=color, backgroundcolor="white")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def research_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "Panopticon research architecture: control, evaluation, and attribution", ha="center", va="top", fontsize=14, fontweight="bold")

    box(ax, 0.04, 0.58, 0.20, 0.22, "Typed latent state", "Workers, leaks, phases,\nHYDRA memory, economy", "#F3F4F6", "#4B5563")
    box(ax, 0.30, 0.58, 0.18, 0.22, "Redacted observation", "Hidden identity removed\nStructured or 136 floats", "#ECFDF5", "#059669")
    box(ax, 0.54, 0.58, 0.18, 0.22, "Policy stratum", "PPO | raw LLM | repair\nsupervisor | hybrid", "#EFF6FF", "#2563EB")
    box(ax, 0.78, 0.58, 0.18, 0.22, "AgentAction", "8 action types\ntarget + sub-action", "#FFF7ED", "#D97706")

    arrow(ax, (0.24, 0.69), (0.30, 0.69), "redact")
    arrow(ax, (0.48, 0.69), (0.54, 0.69), "decide")
    arrow(ax, (0.72, 0.69), (0.78, 0.69), "validate")
    arrow(ax, (0.87, 0.58), (0.14, 0.58), "transition + adversary", bend=-0.24)

    box(ax, 0.09, 0.17, 0.22, 0.20, "Dense training reward", "Revenue/security deltas\nthreat and false-accusation terms", "#FEF3C7", "#B45309")
    box(ax, 0.39, 0.17, 0.22, 0.20, "Independent grader + gate", "5 dimensions plus mandatory\nLevel-4/5 security constraints", "#FEE2E2", "#B91C1C")
    box(ax, 0.69, 0.17, 0.22, 0.20, "Provenance + metrics", "Raw/repair/supervisor labels\nCMG, intervention dependence", "#EDE9FE", "#7C3AED")

    arrow(ax, (0.18, 0.58), (0.20, 0.37), "step signal", bend=0.08)
    arrow(ax, (0.22, 0.17), (0.61, 0.58), "optimize", bend=0.18)
    arrow(ax, (0.20, 0.58), (0.50, 0.37), "episode state", bend=-0.06)
    arrow(ax, (0.61, 0.27), (0.69, 0.27), "pass/fail")
    arrow(ax, (0.63, 0.58), (0.80, 0.37), "interventions", bend=-0.05)

    ax.text(0.5, 0.06, "Key separation: learning signal ≠ composite grade ≠ operational release gate ≠ supervisor attribution", ha="center", fontsize=9.5, fontweight="bold", color="#111827")
    save(fig, "research_architecture")


def product_roadmap():
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Evidence-gated path from research environment to enterprise product", ha="center", va="top", fontsize=14, fontweight="bold")

    stages = [
        (0.03, "1. Research benchmark", "Scenario DSL, mock targets,\nmetrics, open artifacts", "#EFF6FF", "#2563EB"),
        (0.27, "2. Evaluation CI", "Authorized app adapters,\nregression gates, reports", "#ECFDF5", "#059669"),
        (0.51, "3. Runtime gateway", "Session risk, DLP, approvals,\nRBAC, tenant isolation", "#FFF7ED", "#D97706"),
        (0.75, "4. Agentic SOC", "Cross-tool temporal intent,\ncorrelation, human escalation", "#F5F3FF", "#7C3AED"),
    ]
    for x, title, detail, color, edge in stages:
        box(ax, x, 0.48, 0.20, 0.25, title, detail, color, edge)
    for x in (0.23, 0.47, 0.71):
        arrow(ax, (x, 0.605), (x + 0.04, 0.605), color="#374151")

    gates = [
        (0.25, "Gate A", "Reproducible temporal failure\nmissed by single-turn baseline"),
        (0.49, "Gate B", "Paid pilots + precision/recall,\nlatency, utility evidence"),
        (0.73, "Gate C", "Security program + production\nreliability and audit evidence"),
    ]
    for x, title, detail in gates:
        ax.text(x, 0.35, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#991B1B")
        ax.text(x, 0.25, detail, ha="center", va="center", fontsize=7.4, color="#4B5563")

    ax.text(0.5, 0.08, "Do not advance a stage because the demo looks convincing; advance when the previous evidence gate is met.", ha="center", fontsize=9.5, fontweight="bold", color="#111827")
    save(fig, "research_to_product_roadmap")


def main():
    research_architecture()
    product_roadmap()
    print("Generated research architecture and product-roadmap diagrams in PNG and PDF")


if __name__ == "__main__":
    main()
