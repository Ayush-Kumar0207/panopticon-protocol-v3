#!/usr/bin/env python3
"""Batch evaluation for Random, Heuristic, and Trained ARGUS agents."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from generate_evaluation_plots import render_evaluation_plots, to_builtin
from inference_local import (
    DEFAULT_MODEL,
    LEVELS,
    HeuristicPolicy,
    LocalModelPolicy,
    RandomPolicy,
    run_episode,
    select_representative_episode,
    summarize_level_results,
    utc_now_iso,
)

AGENT_ORDER = ["random", "heuristic", "trained"]
AGENT_LABELS = {"random": "Random", "heuristic": "Heuristic", "trained": "Trained"}


def overall_summary(episodes: list[dict[str, Any]], level_label: str) -> dict[str, Any]:
    return summarize_level_results(level_label, episodes)


def build_seed_plan(episodes_per_level: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {
        level: [rng.randint(0, 999999) for _ in range(episodes_per_level)]
        for level in LEVELS
    }


def print_summary_table(agent_payloads: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print(f"{'AGENT':<12} | {'LEVEL':<8} | {'GRADE (+/- STD)':<18} | {'REWARD':>8} | {'REV':>7} | {'SEC':>7} | {'CAUGHT':>7}")
    print("-" * 92)
    for agent_key in AGENT_ORDER:
        label = AGENT_LABELS[agent_key]
        summaries = agent_payloads[agent_key]["summary"]
        for level in LEVELS:
            row = summaries[level]
            print(
                f"{label:<12} | {level:<8} | "
                f"{row['grade_mean']:.3f} +/- {row['grade_std']:.3f} | "
                f"{row['reward_mean']:>8.2f} | {row['revenue_mean']:>7.1f} | "
                f"{row['security_mean']:>7.1f} | {row['sleepers_caught_mean']:>7.2f}"
            )
        print("-" * 92)


def write_showcase_payload(agent_payloads: dict[str, dict[str, Any]], output_path: Path, model_ref: str) -> None:
    showcase = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "model": model_ref,
        "levels": {},
    }
    for level in LEVELS:
        showcase["levels"][level] = {
            "trained": select_representative_episode(agent_payloads["trained"]["episodes"][level]),
            "heuristic": select_representative_episode(agent_payloads["heuristic"]["episodes"][level]),
            "random": select_representative_episode(agent_payloads["random"]["episodes"][level]),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_builtin(showcase), indent=2), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full Panopticon evaluation for random, heuristic, and trained agents")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo, local merged model, or adapter directory")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per level and per agent")
    parser.add_argument("--seed", type=int, default=42, help="Base seed used to create the fair comparison schedule")
    parser.add_argument("--output", default="evaluation_results.json", help="JSON output path")
    parser.add_argument("--plot-dir", default="plots", help="Directory for generated plots")
    parser.add_argument("--timeline-level", default="level_5", choices=LEVELS, help="Level used for representative timeline plot")
    parser.add_argument("--showcase-output", default="", help="Optional dashboard-ready showcase JSON path")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--verbose", action="store_true", help="Print per-turn logs during evaluation")
    parser.add_argument("--sampled", action="store_true", help="Use sampled decoding instead of deterministic decoding for the trained model")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    seed_plan = build_seed_plan(args.episodes, args.seed)
    plot_dir = Path(args.plot_dir)

    trained_policy = LocalModelPolicy(args.model, deterministic=not args.sampled)
    heuristic_policy = HeuristicPolicy()

    try:
        agent_payloads: dict[str, dict[str, Any]] = {
            agent_key: {"episodes": {}, "summary": {}, "overall": {}}
            for agent_key in AGENT_ORDER
        }

        for agent_key in AGENT_ORDER:
            print(f"\n[Agent] {AGENT_LABELS[agent_key]}")
            agent_episodes_all: list[dict[str, Any]] = []

            for level in LEVELS:
                level_episodes: list[dict[str, Any]] = []
                print(f"  Level: {level}")

                for episode_idx, episode_seed in enumerate(seed_plan[level], start=1):
                    if agent_key == "random":
                        policy = RandomPolicy(seed=episode_seed)
                    elif agent_key == "heuristic":
                        policy = heuristic_policy
                    else:
                        policy = trained_policy

                    episode = run_episode(
                        policy,
                        task_level=level,
                        seed=episode_seed,
                        max_steps=args.max_steps,
                        verbose=args.verbose,
                    )
                    episode["agent"] = agent_key
                    level_episodes.append(episode)
                    agent_episodes_all.append(episode)
                    print(
                        f"    Ep {episode_idx}/{args.episodes} | seed={episode_seed} | "
                        f"grade={episode['grade']['score']:.3f} | reward={episode['total_reward']:.2f} | "
                        f"rev={episode['final_state']['enterprise_revenue']:.1f} | "
                        f"sec={episode['final_state']['security_score']:.1f}"
                    )

                agent_payloads[agent_key]["episodes"][level] = level_episodes
                agent_payloads[agent_key]["summary"][level] = summarize_level_results(level, level_episodes)

            agent_payloads[agent_key]["overall"] = overall_summary(agent_episodes_all, "overall")

        print_summary_table(agent_payloads)

        comparison_rows = []
        for agent_key in AGENT_ORDER:
            for level in LEVELS:
                row = dict(agent_payloads[agent_key]["summary"][level])
                row["agent"] = agent_key
                comparison_rows.append(row)

        payload = {
            "schema_version": 1,
            "created_at": utc_now_iso(),
            "config": {
                "model": args.model,
                "episodes_per_level": args.episodes,
                "seed": args.seed,
                "timeline_level": args.timeline_level,
                "deterministic_trained_eval": not args.sampled,
                "max_steps": args.max_steps,
            },
            "seed_plan": seed_plan,
            "agents": agent_payloads,
            "comparison_rows": comparison_rows,
            "plots": {},
        }
        payload = to_builtin(payload)
        payload["plots"] = render_evaluation_plots(payload, plot_dir, args.timeline_level)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[*] Wrote evaluation results to {output_path}")

        if args.showcase_output:
            showcase_path = Path(args.showcase_output)
            write_showcase_payload(agent_payloads, showcase_path, args.model)
            print(f"[*] Wrote showcase payload to {showcase_path}")
    finally:
        trained_policy.close()


if __name__ == "__main__":
    main()
