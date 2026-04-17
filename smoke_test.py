#!/usr/bin/env python3
"""Quick smoke test — verifies the environment works end-to-end."""
import sys, time
from environment import Environment
from models import ActionType, AgentAction
from grader import grade_episode

def run_test(task_level: str, max_steps: int = 30):
    env = Environment(seed=42)
    obs = env.reset(task_level=task_level, seed=42)
    rewards, done, steps = [], False, 0

    while not done and steps < max_steps:
        # Simple heuristic: process first incomplete task target
        action = AgentAction(action_type=ActionType.NOOP)
        for task in obs.tasks:
            if not task.completed and task.target_entities:
                action = AgentAction(action_type=ActionType.PROCESS, target=task.target_entities[0])
                break

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward)
        done = result.done
        steps += 1

    success = len(obs.tasks) == 0
    episode_data = {"total_reward": sum(rewards), "rewards": rewards, "success": success,
                    "steps": steps, "state": env.state.model_dump(), "cascade_failures": 0, "invalid_actions": 0}
    grade = grade_episode(task_level, episode_data)
    return task_level, steps, sum(rewards), success, grade.score, grade.passed

if __name__ == "__main__":
    print("\n  OpenEnv Starter Kit — Smoke Test")
    print("  " + "=" * 50)
    all_pass = True
    for level in ["easy", "medium", "hard", "level_4", "level_5"]:
        name, steps, reward, success, score, passed = run_test(level)
        status = "PASS" if success else "FAIL"
        if not success: all_pass = False
        print(f"  {name:>10}: {status} | steps={steps:>3} | reward={reward:>7.2f} | score={score:.3f}")
    print("  " + "=" * 50)
    print(f"  {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
