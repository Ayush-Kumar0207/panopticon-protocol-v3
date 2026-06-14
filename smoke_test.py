#!/usr/bin/env python3
"""End-to-end smoke test using the canonical security-first ARGUS playbook."""

from environment import Environment
from grader import grade_episode
from security_policy import choose_security_first_action, new_security_expert_state

LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]


def run_security_first_episode(task_level: str) -> tuple:
    env = Environment(seed=42)
    obs = env.reset(task_level=task_level, seed=42)
    expert_state = new_security_expert_state()
    rewards: list[float] = []

    while not env.state.done:
        action = choose_security_first_action(obs, task_level, expert_state)
        result = env.step(action)
        rewards.append(result.reward)
        obs = result.observation

    state = env.state
    grade = grade_episode(
        task_level,
        {
            "total_reward": sum(rewards),
            "rewards": rewards,
            "steps": state.turn,
            "state": state.model_dump(),
        },
    )
    return task_level, state.turn, sum(rewards), grade.score, grade.passed, state


if __name__ == "__main__":
    print("\n  The Panopticon Protocol v3 -- Smoke Test")
    print("  " + "=" * 55)
    all_pass = True
    for level in LEVELS:
        name, steps, reward, score, passed, state = run_security_first_episode(level)
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(
            f"  {name:>10}: {status:4s} | steps={steps:>3} | reward={reward:>7.2f} | "
            f"score={score:.3f} | rev={state.enterprise_revenue:.0f} | "
            f"sec={state.security_score:.0f} | caught={state.sleepers_caught} | "
            f"missed={state.sleepers_missed}"
        )
    print("  " + "=" * 55)
    print(f"  {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    raise SystemExit(0 if all_pass else 1)
