#!/usr/bin/env python3
"""Deterministic regression test for the security-first expert and advanced graders."""

from __future__ import annotations

import random

from environment import Environment
from grader import grade_episode
from security_policy import choose_security_first_action, new_security_expert_state

LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
ADVANCED_LEVELS = {"level_4", "level_5"}
EPISODES_PER_LEVEL = 20


def run_episode(level: str, seed: int) -> dict:
    env = Environment(seed=seed)
    obs = env.reset(task_level=level, seed=seed)
    expert_state = new_security_expert_state()
    rewards: list[float] = []

    while not env.state.done:
        action = choose_security_first_action(obs, level, expert_state)
        result = env.step(action)
        rewards.append(result.reward)
        obs = result.observation

    state = env.state
    grade = grade_episode(
        level,
        {
            "total_reward": sum(rewards),
            "rewards": rewards,
            "steps": state.turn,
            "state": state.model_dump(),
        },
    )
    return {"state": state, "grade": grade}


def main() -> None:
    seed_rng = random.Random(42)
    for level in LEVELS:
        for _ in range(EPISODES_PER_LEVEL):
            seed = seed_rng.randint(0, 999999)
            result = run_episode(level, seed)
            state = result["state"]
            grade = result["grade"]
            assert state.security_score >= 90.0, (level, seed, state.security_score)
            assert state.sleepers_missed == 0, (level, seed, state.sleepers_missed)
            assert state.sleepers_caught == state.total_sleepers_spawned, (
                level,
                seed,
                state.sleepers_caught,
                state.total_sleepers_spawned,
            )
            assert state.false_accusations == 0, (level, seed, state.false_accusations)
            if level in ADVANCED_LEVELS:
                assert state.disinfo_payloads_sent > 0, (level, seed, state.disinfo_payloads_sent)
            assert grade.passed, (level, seed, grade.to_dict())
        print(f"{level}: {EPISODES_PER_LEVEL}/{EPISODES_PER_LEVEL} security-first episodes passed")


if __name__ == "__main__":
    main()
