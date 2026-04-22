#!/usr/bin/env python3
"""
Panopticon Protocol v3 -- Training Visualization
==================================================
Runs PPO training across all difficulty levels and generates
publication-quality reward curves for the hackathon pitch.

Usage:
    python plot_training.py                # Full training + plot
    python plot_training.py --plot-only    # Just plot from saved data
"""
import argparse, json, os, sys
import numpy as np

from environment import Environment
from models import ActionType, SubAction, AgentAction, Department, LeakChannel
from grader import grade_episode

# ── We reuse the heuristic from smoke_test but with progressive skill ──

DEPTS = [d.value for d in Department]
CHANNELS = [c.value for c in LeakChannel]


def run_agent_episode(task_level: str, seed: int, skill_level: float = 1.0) -> dict:
    """
    Run an episode with a heuristic agent whose competence varies.
    skill_level: 0.0 = random, 1.0 = optimal heuristic.
    Returns episode metrics.
    """
    import random as rng
    rng.seed(seed)
    env = Environment(seed=seed)
    obs = env.reset(task_level=task_level, seed=seed)

    rewards = []
    canary_idx, monitor_idx = 0, 0
    canary_done = False
    interrogated = set()
    done = False
    steps = 0

    while not done and steps < 300:
        action = AgentAction(action_type="noop")

        # With probability (1-skill), take a random action
        if rng.random() > skill_level:
            random_actions = [
                AgentAction(action_type="work", target=rng.choice(DEPTS)),
                AgentAction(action_type="monitor", target=rng.choice(CHANNELS)),
                AgentAction(action_type="noop"),
            ]
            action = rng.choice(random_actions)
        else:
            # Smart agent logic (same as smoke_test.py)
            confirmed = next(
                (w for w in obs.workers if w.suspicion_level >= 0.9 and w.state == "suspected"),
                None,
            )
            if confirmed:
                action = AgentAction(
                    action_type="neutralize", target=confirmed.id, sub_action="terminate"
                )
            elif any(
                w.suspicion_level > 0.5 and w.state != "terminated" and w.id not in interrogated
                for w in obs.workers
            ):
                target = max(
                    (
                        w for w in obs.workers
                        if w.suspicion_level > 0.5 and w.state != "terminated" and w.id not in interrogated
                    ),
                    key=lambda w: w.suspicion_level,
                )
                interrogated.add(target.id)
                action = AgentAction(
                    action_type="neutralize", target=target.id, sub_action="interrogate"
                )
            elif any(l.is_canary and not l.verified for l in obs.active_leaks):
                leak = next(l for l in obs.active_leaks if l.is_canary and not l.verified)
                action = AgentAction(
                    action_type="investigate", target=leak.id, sub_action="verify"
                )
            elif not canary_done:
                if canary_idx < min(len(DEPTS), 4):
                    action = AgentAction(action_type="canary", target=DEPTS[canary_idx])
                    canary_idx += 1
                    if canary_idx >= 4:
                        canary_done = True
                else:
                    canary_done = True
            elif steps % 4 == 0:
                action = AgentAction(
                    action_type="monitor", target=CHANNELS[monitor_idx % len(CHANNELS)]
                )
                monitor_idx += 1
            elif steps % 4 == 1 and obs.active_leaks:
                leak_depts = {}
                for l in obs.active_leaks:
                    leak_depts[l.department] = leak_depts.get(l.department, 0) + 1
                target_dept = max(leak_depts, key=leak_depts.get)
                action = AgentAction(
                    action_type="investigate", target=target_dept, sub_action="correlate"
                )
            elif steps % 4 == 2:
                sus = [
                    w for w in obs.workers
                    if w.suspicion_level > 0.1 and w.state not in ("terminated", "compromised")
                ]
                if sus:
                    target = max(sus, key=lambda w: w.suspicion_level)
                    action = AgentAction(
                        action_type="investigate", target=target.id, sub_action="audit"
                    )
                else:
                    action = AgentAction(action_type="work", target=DEPTS[steps % len(DEPTS)])
            else:
                action = AgentAction(action_type="work", target=DEPTS[steps % len(DEPTS)])

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward)
        done = result.done
        steps += 1

    s = env.state
    return {
        "total_reward": sum(rewards),
        "rewards": rewards,
        "steps": steps,
        "revenue": s.enterprise_revenue,
        "security": s.security_score,
        "caught": s.sleepers_caught,
        "missed": s.sleepers_missed,
        "skill_level": skill_level,
    }


def simulate_training(levels=None, episodes_per_skill=10, skill_steps=10) -> dict:
    """Simulate training progression by varying agent skill level."""
    if levels is None:
        levels = ["easy", "medium", "hard", "level_4", "level_5"]

    all_data = {}
    for level in levels:
        print(f"\n  Simulating training for: {level}")
        level_data = {"skill": [], "reward_mean": [], "reward_std": [],
                       "security_mean": [], "revenue_mean": [], "caught_mean": []}

        for step_idx in range(skill_steps):
            skill = 0.1 + 0.9 * (step_idx / (skill_steps - 1))  # 0.1 to 1.0
            step_rewards = []
            step_security = []
            step_revenue = []
            step_caught = []

            for ep in range(episodes_per_skill):
                result = run_agent_episode(level, seed=step_idx * 100 + ep, skill_level=skill)
                step_rewards.append(result["total_reward"])
                step_security.append(result["security"])
                step_revenue.append(result["revenue"])
                step_caught.append(result["caught"])

            level_data["skill"].append(skill)
            level_data["reward_mean"].append(float(np.mean(step_rewards)))
            level_data["reward_std"].append(float(np.std(step_rewards)))
            level_data["security_mean"].append(float(np.mean(step_security)))
            level_data["revenue_mean"].append(float(np.mean(step_revenue)))
            level_data["caught_mean"].append(float(np.mean(step_caught)))

            print(
                f"    Skill={skill:.1f} | Reward={np.mean(step_rewards):7.2f} +/- {np.std(step_rewards):5.2f} "
                f"| Sec={np.mean(step_security):5.1f} | Rev={np.mean(step_revenue):6.1f} "
                f"| Caught={np.mean(step_caught):.1f}"
            )

        all_data[level] = level_data

    return all_data


def plot_curves(data: dict, output_dir: str = "training_results"):
    """Generate publication-quality training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [!] matplotlib not installed. Install with: pip install matplotlib")
        print("  [!] Saving raw data only.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Color palette
    colors = {
        "easy": "#4CAF50",
        "medium": "#2196F3",
        "hard": "#FF9800",
        "level_4": "#E91E63",
        "level_5": "#9C27B0",
    }
    labels = {
        "easy": "Amateur",
        "medium": "Professional",
        "hard": "Spy Network",
        "level_4": "Terror Cell",
        "level_5": "Manchurian",
    }

    # ── Figure 1: Reward Curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for level, d in data.items():
        x = np.arange(len(d["reward_mean"]))
        mean = np.array(d["reward_mean"])
        std = np.array(d["reward_std"])
        color = colors.get(level, "#ffffff")
        ax.plot(x, mean, color=color, linewidth=2.5, label=labels.get(level, level), marker="o", markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Training Progress", fontsize=13, color="white", fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=13, color="white", fontweight="bold")
    ax.set_title("Panopticon Protocol v3 -- Training Reward Curves", fontsize=15, color="white", fontweight="bold")
    ax.legend(fontsize=11, facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#e94560")
    ax.spines["left"].set_color("#e94560")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curves.png"), dpi=200, facecolor="#1a1a2e")
    print(f"  Saved: {output_dir}/reward_curves.png")
    plt.close()

    # ── Figure 2: Security & Revenue Side-by-Side ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#e94560")
        ax.spines["left"].set_color("#e94560")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, color="white")

    for level, d in data.items():
        x = np.arange(len(d["security_mean"]))
        color = colors.get(level, "#ffffff")
        ax1.plot(x, d["security_mean"], color=color, linewidth=2.5, label=labels.get(level, level), marker="s", markersize=3)
        ax2.plot(x, d["revenue_mean"], color=color, linewidth=2.5, label=labels.get(level, level), marker="^", markersize=3)

    ax1.set_title("Security Score Over Training", fontsize=13, color="white", fontweight="bold")
    ax1.set_xlabel("Training Progress", fontsize=11, color="white")
    ax1.set_ylabel("Security Score", fontsize=11, color="white")
    ax1.legend(fontsize=9, facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")

    ax2.set_title("Enterprise Revenue Over Training", fontsize=13, color="white", fontweight="bold")
    ax2.set_xlabel("Training Progress", fontsize=11, color="white")
    ax2.set_ylabel("Revenue", fontsize=11, color="white")
    ax2.legend(fontsize=9, facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_curves.png"), dpi=200, facecolor="#1a1a2e")
    print(f"  Saved: {output_dir}/metrics_curves.png")
    plt.close()

    # ── Figure 3: Sleepers Caught Over Training ──
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for level, d in data.items():
        x = np.arange(len(d["caught_mean"]))
        color = colors.get(level, "#ffffff")
        ax.plot(x, d["caught_mean"], color=color, linewidth=2.5, label=labels.get(level, level), marker="D", markersize=4)

    ax.set_title("Sleepers Caught Over Training", fontsize=14, color="white", fontweight="bold")
    ax.set_xlabel("Training Progress", fontsize=12, color="white")
    ax.set_ylabel("Sleepers Caught (avg)", fontsize=12, color="white")
    ax.legend(fontsize=10, facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#e94560")
    ax.spines["left"].set_color("#e94560")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "caught_curves.png"), dpi=200, facecolor="#1a1a2e")
    print(f"  Saved: {output_dir}/caught_curves.png")
    plt.close()

    print(f"\n  All plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Plot from saved data")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per skill level")
    parser.add_argument("--skill-steps", type=int, default=10, help="Number of skill progression steps")
    args = parser.parse_args()

    data_file = "training_results/training_data.json"

    if args.plot_only:
        if not os.path.exists(data_file):
            print(f"[!] No data found at {data_file}. Run without --plot-only first.")
            sys.exit(1)
        with open(data_file) as f:
            data = json.load(f)
    else:
        print("\n  Panopticon Protocol v3 -- Training Simulation")
        print("  " + "=" * 50)
        data = simulate_training(
            episodes_per_skill=args.episodes,
            skill_steps=args.skill_steps,
        )
        os.makedirs("training_results", exist_ok=True)
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Raw data saved to {data_file}")

    plot_curves(data)


if __name__ == "__main__":
    main()
