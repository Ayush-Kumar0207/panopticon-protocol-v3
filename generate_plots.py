#!/usr/bin/env python3
"""Parse training logs from output_logs.txt and generate publication-quality plots."""

import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

LOG_FILE = Path("output_logs.txt")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Parse training metrics from logs ──
def parse_training_logs(log_path):
    """Extract loss, grad_norm, learning_rate, epoch from log lines, grouped by level."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    
    # Find level headers
    level_pattern = re.compile(r"TRAINING:\s+(\w+)\s+\|")
    metric_pattern = re.compile(r"\{'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.e+-]+),\s*'learning_rate':\s*([\d.e+-]+),\s*'epoch':\s*([\d.]+)\}")
    
    levels_data = {}
    current_level = None
    
    for line in text.split("\n"):
        level_match = level_pattern.search(line)
        if level_match:
            current_level = level_match.group(1)
            levels_data[current_level] = {"loss": [], "grad_norm": [], "lr": [], "epoch": []}
        
        metric_match = metric_pattern.search(line)
        if metric_match and current_level:
            loss = float(metric_match.group(1))
            grad_norm = float(metric_match.group(2))
            lr = float(metric_match.group(3))
            epoch = float(metric_match.group(4))
            levels_data[current_level]["loss"].append(loss)
            levels_data[current_level]["grad_norm"].append(grad_norm)
            levels_data[current_level]["lr"].append(lr)
            levels_data[current_level]["epoch"].append(epoch)
    
    return levels_data

# ── Parse expert trajectory grades ──
def parse_expert_grades(log_path):
    """Extract expert episode grades grouped by level."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    
    level_pattern = re.compile(r"TRAINING:\s+(\w+)\s+\|")
    grade_pattern = re.compile(r"\[Expert Ep (\d+)/(\d+)\].*Grade=([\d.]+)")
    
    levels_grades = {}
    current_level = None
    
    for line in text.split("\n"):
        level_match = level_pattern.search(line)
        if level_match:
            current_level = level_match.group(1)
            levels_grades[current_level] = []
        
        grade_match = grade_pattern.search(line)
        if grade_match and current_level:
            levels_grades[current_level].append(float(grade_match.group(3)))
    
    return levels_grades

# ── Color palette ──
COLORS = {
    "easy": "#22c55e",
    "medium": "#3b82f6", 
    "hard": "#f59e0b",
    "level_4": "#ef4444",
    "level_5": "#8b5cf6",
}

LEVEL_LABELS = {
    "easy": "Level 1: Easy",
    "medium": "Level 2: Medium",
    "hard": "Level 3: Hard",
    "level_4": "Level 4: Advanced",
    "level_5": "Level 5: Expert",
}

def setup_style():
    """Set dark premium style."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def plot_combined_loss(levels_data):
    """Plot 1: Combined loss curve across all curriculum levels."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    global_step = 0
    boundaries = []
    
    for level_name in ["easy", "medium", "hard", "level_4", "level_5"]:
        if level_name not in levels_data:
            continue
        data = levels_data[level_name]
        steps = list(range(global_step, global_step + len(data["loss"])))
        
        ax.plot(steps, data["loss"], color=COLORS[level_name], 
                label=LEVEL_LABELS[level_name], linewidth=1.2, alpha=0.85)
        
        boundaries.append(global_step)
        global_step += len(data["loss"])
    
    # Draw vertical lines at level boundaries
    for b in boundaries[1:]:
        ax.axvline(x=b, color='#484f58', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel("Training Steps (Global)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("Panopticon ARGUS — Curriculum Training Loss", fontweight='bold', fontsize=16, pad=15)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss_curve.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] training_loss_curve.png")


def plot_per_level_loss(levels_data):
    """Plot 2: Individual loss curves per level (subplots)."""
    setup_style()
    level_names = [l for l in ["easy", "medium", "hard", "level_4", "level_5"] if l in levels_data]
    n = len(level_names)
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    
    for ax, level_name in zip(axes, level_names):
        data = levels_data[level_name]
        ax.plot(data["epoch"], data["loss"], color=COLORS[level_name], linewidth=1.2)
        ax.fill_between(data["epoch"], data["loss"], alpha=0.15, color=COLORS[level_name])
        ax.set_title(LEVEL_LABELS[level_name], fontweight='bold', fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Panopticon ARGUS — Per-Level Training Loss", fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_level_loss.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] per_level_loss.png")


def plot_grad_norm(levels_data):
    """Plot 3: Gradient norm across training."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    
    global_step = 0
    
    for level_name in ["easy", "medium", "hard", "level_4", "level_5"]:
        if level_name not in levels_data:
            continue
        data = levels_data[level_name]
        steps = list(range(global_step, global_step + len(data["grad_norm"])))
        ax.plot(steps, data["grad_norm"], color=COLORS[level_name],
                label=LEVEL_LABELS[level_name], linewidth=0.8, alpha=0.7)
        global_step += len(data["grad_norm"])
    
    ax.set_xlabel("Training Steps (Global)", fontweight='bold')
    ax.set_ylabel("Gradient Norm", fontweight='bold')
    ax.set_title("Panopticon ARGUS — Gradient Norm During Training", fontweight='bold', fontsize=16, pad=15)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_norm.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] gradient_norm.png")


def plot_learning_rate(levels_data):
    """Plot 4: Learning rate schedule."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    
    global_step = 0
    
    for level_name in ["easy", "medium", "hard", "level_4", "level_5"]:
        if level_name not in levels_data:
            continue
        data = levels_data[level_name]
        steps = list(range(global_step, global_step + len(data["lr"])))
        ax.plot(steps, [lr * 1e5 for lr in data["lr"]], color=COLORS[level_name],
                label=LEVEL_LABELS[level_name], linewidth=1.2)
        global_step += len(data["lr"])
    
    ax.set_xlabel("Training Steps (Global)", fontweight='bold')
    ax.set_ylabel("Learning Rate (×1e-5)", fontweight='bold')
    ax.set_title("Panopticon ARGUS — Learning Rate Schedule", fontweight='bold', fontsize=16, pad=15)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_rate_schedule.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] learning_rate_schedule.png")


def plot_expert_grades(levels_grades):
    """Plot 5: Expert trajectory grades bar chart."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    level_names = [l for l in ["easy", "medium", "hard", "level_4", "level_5"] if l in levels_grades]
    avg_grades = [np.mean(levels_grades[l]) for l in level_names]
    std_grades = [np.std(levels_grades[l]) for l in level_names]
    colors = [COLORS[l] for l in level_names]
    labels = [LEVEL_LABELS[l] for l in level_names]
    
    bars = ax.bar(labels, avg_grades, color=colors, alpha=0.85, edgecolor='#30363d', linewidth=1.2)
    ax.errorbar(labels, avg_grades, yerr=std_grades, fmt='none', ecolor='#c9d1d9', capsize=5, capthick=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_grades):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#c9d1d9')
    
    ax.set_ylabel("Average Grade", fontweight='bold')
    ax.set_title("Panopticon ARGUS — Expert Trajectory Quality per Level", fontweight='bold', fontsize=16, pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expert_grades.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] expert_grades.png")


def plot_training_summary(levels_data):
    """Plot 6: Final loss comparison across levels."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    level_names = [l for l in ["easy", "medium", "hard", "level_4", "level_5"] if l in levels_data]
    
    # Get initial and final loss for each level
    initial_losses = [levels_data[l]["loss"][0] for l in level_names]
    final_losses = [levels_data[l]["loss"][-1] for l in level_names]
    labels = [LEVEL_LABELS[l] for l in level_names]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, initial_losses, width, label='Initial Loss', 
                   color='#f87171', alpha=0.8, edgecolor='#30363d')
    bars2 = ax.bar(x + width/2, final_losses, width, label='Final Loss',
                   color='#34d399', alpha=0.8, edgecolor='#30363d')
    
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("Panopticon ARGUS — Loss Reduction per Level", fontweight='bold', fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_reduction.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] loss_reduction.png")


if __name__ == "__main__":
    print("\n[*] Parsing training logs...")
    levels_data = parse_training_logs(LOG_FILE)
    levels_grades = parse_expert_grades(LOG_FILE)
    
    print(f"  Found {len(levels_data)} levels: {list(levels_data.keys())}")
    for level, data in levels_data.items():
        print(f"  {level}: {len(data['loss'])} metric points, loss {data['loss'][0]:.4f} -> {data['loss'][-1]:.4f}")
    
    print("\n[*] Generating plots...")
    plot_combined_loss(levels_data)
    plot_per_level_loss(levels_data)
    plot_grad_norm(levels_data)
    plot_learning_rate(levels_data)
    plot_expert_grades(levels_grades)
    plot_training_summary(levels_data)
    
    print(f"\n[*] All plots saved to {OUTPUT_DIR}/")
