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
from panopticon_bench.seed_plan import canonical_sha256, load_seed_plan


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


def prepare_training_log(
    path: Path,
    *,
    start_episode: int,
    config_sha256: str,
    resuming: bool,
) -> None:
    """Keep the append-only log aligned with the last durable checkpoint."""
    if not path.exists():
        if resuming and start_episode > 0:
            raise RuntimeError(
                f"Resume checkpoint is at episode {start_episode}, but its training log "
                f"is missing: {path}"
            )
        return
    if not resuming:
        raise RuntimeError(
            f"Training log already exists: {path}. Pass --resume with its checkpoint "
            "or choose a new output/log path."
        )
    retained: list[str] = []
    retained_episodes: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("config_sha256") != config_sha256:
            raise RuntimeError(f"Training log configuration mismatch at line {line_number}")
        episode = int(record["episode"])
        if episode <= start_episode:
            retained.append(json.dumps(record, sort_keys=True))
            retained_episodes.append(episode)
    expected_episodes = list(range(1, start_episode + 1))
    if retained_episodes != expected_episodes:
        raise RuntimeError(
            "Training log is not a complete, ordered record through the resume checkpoint: "
            f"expected episodes 1..{start_episode}, found {retained_episodes[:5]}..."
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    temporary.replace(path)


def training_episode_spec(
    *,
    training_seed: int,
    episode_number: int,
    levels: list[str],
    argus_population: list[str],
    development_split: dict[str, list[int]],
) -> tuple[str, int, str]:
    """Return the resume-independent level, environment seed, and defender."""
    schedule_rng = random.Random(f"{training_seed}:{episode_number}")
    level = schedule_rng.choice(levels)
    episode_seed = schedule_rng.choice(development_split[level])
    argus_name = schedule_rng.choice(argus_population)
    return level, episode_seed, argus_name


def checkpoint_metadata(
    *,
    episode: int,
    baseline: float,
    recent_scores: list[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "episode": episode,
        "baseline": baseline,
        "recent_mean_score": statistics.mean(recent_scores) if recent_scores else 0.0,
        "recent_scores": list(recent_scores),
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        metadata["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return metadata


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=2_000,
        help="Target total; rerunning with --resume never exceeds this episode count",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--seed-plan",
        default="research_paper/data/seed_plans/v6_seed_plan.json",
        help="Frozen plan; HYDRA training uses only its development split",
    )
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

    seed_plan = load_seed_plan(args.seed_plan)
    missing_levels = set(args.levels) - set(seed_plan["levels"])
    if missing_levels:
        raise SystemExit(f"levels missing from seed plan: {sorted(missing_levels)}")
    training_split = seed_plan["splits"]["development"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = {
        "objective_schema_version": HYDRA_OBJECTIVE_SCHEMA_VERSION,
        "seed": args.seed,
        "levels": args.levels,
        "argus_population": args.argus_population,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "baseline_decay": args.baseline_decay,
        "seed_plan_sha256": seed_plan["seed_plan_sha256"],
        "training_seed_split": "development",
    }
    config_sha256 = canonical_sha256(config)
    policy = NeuralHydraPolicy(
        checkpoint=args.resume or None,
        deterministic=False,
        record_gradients=True,
    )
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.learning_rate)
    start_episode = 0
    baseline = 0.0
    restored_recent_scores: list[float] = []
    if args.resume:
        training_state = load_hydra_training_state(args.resume, optimizer, policy.device)
        if training_state.get("config") != config:
            raise RuntimeError("Resume checkpoint configuration differs from requested training run")
        start_episode = int(training_state.get("episode", 0))
        baseline = float(training_state.get("baseline", 0.0))
        restored_recent_scores = [
            float(score) for score in training_state.get("recent_scores", [])
        ][-100:]
        torch_rng_state = training_state.get("torch_rng_state")
        if torch_rng_state is None:
            raise RuntimeError("Resume checkpoint is missing the PyTorch RNG state")
        torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_states = training_state.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])

    output_path = Path(args.output)
    log_path = Path(args.training_log)
    prepare_training_log(
        log_path,
        start_episode=start_episode,
        config_sha256=config_sha256,
        resuming=bool(args.resume),
    )
    if start_episode >= args.episodes:
        print(
            f"Checkpoint already reached episode {start_episode}, "
            f"which satisfies target {args.episodes}; nothing to do."
        )
        return
    recent_scores: list[float] = restored_recent_scores

    for episode_number in range(start_episode + 1, args.episodes + 1):
        level, episode_seed, argus_name = training_episode_spec(
            training_seed=args.seed,
            episode_number=episode_number,
            levels=args.levels,
            argus_population=args.argus_population,
            development_split=training_split,
        )
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
            "config_sha256": config_sha256,
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
                metadata=checkpoint_metadata(
                    episode=episode_number,
                    baseline=baseline,
                    recent_scores=recent_scores,
                    config=config,
                ),
                optimizer_state_dict=optimizer.state_dict(),
            )

    final_episode = args.episodes
    save_hydra_checkpoint(
        output_path,
        policy.model,
        metadata=checkpoint_metadata(
            episode=final_episode,
            baseline=baseline,
            recent_scores=recent_scores,
            config=config,
        ),
        optimizer_state_dict=optimizer.state_dict(),
    )
    print(f"Saved neural HYDRA checkpoint: {output_path}")
    print("Use v6_evaluation.py on frozen development/final seeds; training score is not test evidence.")


if __name__ == "__main__":
    main()
