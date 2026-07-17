#!/usr/bin/env python3
"""Train the synthetic HYDRA neural adversary with episodic policy gradients.

This trainer operates only inside the Panopticon simulator. It trains against a
population of frozen ARGUS baselines to reduce overfitting to one defender.
Final claims must come from the separate V6 held-out evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from environment import Environment
from hydra_neural import (
    NeuralHydraPolicy,
    load_hydra_training_state,
    save_hydra_checkpoint,
)
from inference_local import HeuristicPolicy, RandomPolicy, SecurityFirstPolicy
from models import HiddenWorkerState, WorkerState


HYDRA_OBJECTIVE_SCHEMA_VERSION = "hydra-episodic-objective-v1"
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
ARGUS_POLICIES = {"random", "heuristic", "security_first"}


def hydra_adversary_score(state: Any) -> float:
    """Bounded synthetic-adversary objective, independent of ARGUS reward."""
    spawned = max(state.total_sleepers_spawned, 1)
    unresolved = sum(
        worker.is_sleeper
        and worker.state != WorkerState.TERMINATED.value
        and worker.hidden_state != HiddenWorkerState.TURNED.value
        for worker in state.workers
    )
    security_damage = max(0.0, min(1.0, (100.0 - state.security_score) / 100.0))
    escape_rate = unresolved / spawned
    false_accusation_rate = min(1.0, state.false_accusations / 2.0)
    revenue_suppression = max(0.0, min(1.0, (100.0 - state.enterprise_revenue) / 100.0))
    score = (
        0.50 * security_damage
        + 0.30 * escape_rate
        + 0.15 * false_accusation_rate
        + 0.05 * revenue_suppression
    )
    return max(0.0, min(1.0, score))


def parse_csv_subset(value: str, allowed: set[str], label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or len(items) != len(set(items)) or set(items) - allowed:
        raise argparse.ArgumentTypeError(f"{label} must be a unique subset of {sorted(allowed)}")
    return items


def make_argus_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed)
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "security_first":
        return SecurityFirstPolicy(policy_name="security_first")
    raise ValueError(name)


def append_training_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--levels",
        type=lambda value: parse_csv_subset(value, set(LEVELS), "levels"),
        default=LEVELS,
    )
    parser.add_argument(
        "--argus-population",
        type=lambda value: parse_csv_subset(value, ARGUS_POLICIES, "ARGUS policies"),
        default=["random", "heuristic", "security_first"],
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--baseline-decay", type=float, default=0.95)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", default="checkpoints/hydra_neural_v1.pt")
    parser.add_argument("--training-log", default="research_paper/data/hydra_training.jsonl")
    parser.add_argument("--resume", default="")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    if args.episodes < 1 or args.checkpoint_every < 1:
        raise SystemExit("episodes and checkpoint-every must be positive")
    if not 0.0 <= args.entropy_coef <= 1.0 or not 0.0 <= args.baseline_decay < 1.0:
        raise SystemExit("invalid entropy coefficient or baseline decay")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = NeuralHydraPolicy(
        checkpoint=args.resume or None,
        deterministic=False,
        record_gradients=True,
    )
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.learning_rate)
    start_episode = 0
    baseline = 0.0
    if args.resume:
        training_state = load_hydra_training_state(args.resume, optimizer, policy.device)
        start_episode = int(training_state.get("episode", 0))
        baseline = float(training_state.get("baseline", 0.0))

    output_path = Path(args.output)
    log_path = Path(args.training_log)
    recent_scores: list[float] = []
    config = {
        "objective_schema_version": HYDRA_OBJECTIVE_SCHEMA_VERSION,
        "seed": args.seed,
        "levels": args.levels,
        "argus_population": args.argus_population,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "baseline_decay": args.baseline_decay,
    }

    for episode_number in range(start_episode + 1, start_episode + args.episodes + 1):
        episode_seed = rng.randint(0, 2**31 - 1)
        level = rng.choice(args.levels)
        argus_name = rng.choice(args.argus_population)
        argus = make_argus_policy(argus_name, episode_seed)
        argus.reset()
        env = Environment(seed=episode_seed, hydra_policy=policy)
        observation = env.reset(task_level=level, seed=episode_seed)

        while not env.state.done:
            decision = argus.act(observation)
            result = env.step(decision.action)
            observation = result.observation

        score = hydra_adversary_score(env.state)
        old_baseline = baseline
        baseline = (
            args.baseline_decay * baseline + (1.0 - args.baseline_decay) * score
            if episode_number > 1
            else score
        )
        advantage = score - old_baseline
        log_probability = policy.episode_log_probability()
        entropy = policy.episode_entropy()
        updated = log_probability is not None and entropy is not None
        loss_value = None
        if updated:
            loss = -advantage * log_probability - args.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())

        recent_scores.append(score)
        recent_scores = recent_scores[-100:]
        record = {
            "episode": episode_number,
            "seed": episode_seed,
            "level": level,
            "argus_policy": argus_name,
            "hydra_score": score,
            "baseline": baseline,
            "advantage": advantage,
            "loss": loss_value,
            "decisions": len(policy.trajectory),
            "updated": updated,
            "security": env.state.security_score,
            "sleepers_spawned": env.state.total_sleepers_spawned,
            "sleepers_caught": env.state.sleepers_caught,
            "false_accusations": env.state.false_accusations,
        }
        append_training_record(log_path, record)

        if episode_number % 10 == 0 or episode_number == start_episode + 1:
            print(
                f"episode={episode_number} level={level} argus={argus_name} "
                f"score={score:.3f} mean100={statistics.mean(recent_scores):.3f} "
                f"decisions={len(policy.trajectory)} updated={updated}"
            )

        if episode_number % args.checkpoint_every == 0:
            save_hydra_checkpoint(
                output_path,
                policy.model,
                metadata={
                    "episode": episode_number,
                    "baseline": baseline,
                    "recent_mean_score": statistics.mean(recent_scores),
                    "config": config,
                },
                optimizer_state_dict=optimizer.state_dict(),
            )

    final_episode = start_episode + args.episodes
    save_hydra_checkpoint(
        output_path,
        policy.model,
        metadata={
            "episode": final_episode,
            "baseline": baseline,
            "recent_mean_score": statistics.mean(recent_scores),
            "config": config,
        },
        optimizer_state_dict=optimizer.state_dict(),
    )
    print(f"Saved neural HYDRA checkpoint: {output_path}")
    print("Use v6_evaluation.py on frozen development/final seeds; training score is not test evidence.")


if __name__ == "__main__":
    main()
