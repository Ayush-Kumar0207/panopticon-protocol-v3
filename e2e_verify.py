"""
End-to-End Pipeline Verification — Panopticon v3
==================================================
Mocks environment.py with stub_env.py and runs a full
train/eval/resume cycle to verify the RL pipeline.
"""

import sys, os, torch
from unittest.mock import patch

# 1. Setup Mock Environment
import stub_env
import gym_wrapper

# Monkeypatch gym_wrapper's Environment with stub_env's Environment
gym_wrapper.Environment = stub_env.Environment

from train_rl import train, TOTAL_TIMESTEPS, NUM_STEPS
from gym_wrapper import OBS_SIZE
import train_rl

# 2. Configure for fast test
train_rl.TOTAL_TIMESTEPS = 256
train_rl.NUM_STEPS = 128


def verify_pipeline():
    print("\n🚀 [PANOPTICON v3] End-to-End Pipeline Validation")

    task_level = "easy"
    checkpoint_file = f"best_ppo_{task_level}.pt"
    final_file = f"final_ppo_{task_level}.pt"

    # Cleanup previous artifacts
    for f in [checkpoint_file, final_file]:
        if os.path.exists(f):
            os.remove(f)

    # TEST A: Initial Training
    print("\n── Test A: Initial Training ──")
    try:
        train(task_level=task_level)
    except Exception as e:
        print(f"❌ Initial Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    if not os.path.exists(final_file):
        print(f"❌ Final model {final_file} was not saved.")
        return False
    print(f"✅ Training completed. {final_file} exists.")

    # TEST B: Resume from Checkpoint
    print("\n── Test B: Resume from Checkpoint ──")
    try:
        train(task_level=task_level, checkpoint_path=final_file)
    except Exception as e:
        print(f"❌ Resume Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print("✅ Resume completed successfully.")

    # TEST C: Observation Shape Check
    print("\n── Test C: Observation Shape Consistency ──")
    from gym_wrapper import OpenEnvGymWrapper
    env = OpenEnvGymWrapper(task_level=task_level)
    obs, _ = env.reset()
    if obs.shape[0] != OBS_SIZE:
        print(f"❌ Observation size mismatch: {obs.shape[0]} != {OBS_SIZE}")
        return False
    print(f"✅ Observation vector: {obs.shape[0]} features (expected {OBS_SIZE}).")

    # TEST D: Action Space Check
    print("\n── Test D: Action Space ──")
    action = env.action_space.sample()
    if len(action) != 3:
        print(f"❌ Action space should be 3-dim MultiDiscrete, got {len(action)}")
        return False
    print(f"✅ Action space: MultiDiscrete({list(env.action_space.nvec)}) — 3 heads verified.")

    # TEST E: Step produces valid output
    print("\n── Test E: Step Cycle ──")
    obs2, reward, done, truncated, info = env.step(action)
    if obs2.shape[0] != OBS_SIZE:
        print(f"❌ Post-step obs shape mismatch: {obs2.shape[0]}")
        return False
    print(f"✅ Step produced valid output. Reward: {reward:.2f}")

    print("\n💯 [SUCCESS] All Panopticon v3 Pipeline Checks Passed!")
    return True


if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
