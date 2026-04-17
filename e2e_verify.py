"""
End-to-End Pipeline Verification Script.
Mocks environment.py with stub_env.py and runs a full train/eval/resume cycle.
"""

import sys, os, torch, shutil
from unittest.mock import MagicMock, patch

# 1. Setup Mock Environment
import stub_env
import gym_wrapper

# Monkeypatch gym_wrapper's Environment with stub_env's Environment
gym_wrapper.Environment = stub_env.Environment

from train_rl import train, TOTAL_TIMESTEPS, NUM_STEPS
import train_rl

# 2. Configure for fast test
train_rl.TOTAL_TIMESTEPS = 256 # Exactly 2 updates (NUM_STEPS=128)
train_rl.NUM_STEPS = 128

def verify_pipeline():
    print("\n🚀 [PHASE 3] Starting End-to-End Validation...")
    
    task_level = "easy"
    checkpoint_file = f"best_ppo_{task_level}.pt"
    final_file = f"final_ppo_{task_level}.pt"
    
    # Cleanup previous artifacts
    for f in [checkpoint_file, final_file]:
        if os.path.exists(f): os.remove(f)

    # TEST A: Start Training
    print("\n--- Test A: Initial Training ---")
    try:
        train(task_level=task_level)
    except Exception as e:
        print(f"❌ Initial Training Failed: {e}")
        return False
    
    if not os.path.exists(final_file):
        print(f"❌ Final model {final_file} was not saved.")
        return False
    print(f"✅ Training completed. {final_file} exists.")

    # TEST B: Resume from Checkpoint
    print("\n--- Test B: Resume from Checkpoint ---")
    # We'll use the final file as a checkpoint for testing the loader
    try:
        train(task_level=task_level, checkpoint_path=final_file)
    except Exception as e:
        print(f"❌ Resume Failed: {e}")
        return False
    print(f"✅ Resume completed successfully.")

    # TEST C: Grader & Server Wiring (Check if models.py is compatible)
    print("\n--- Test C: Shape Consistency Check ---")
    from gym_wrapper import OpenEnvGymWrapper
    env = OpenEnvGymWrapper(task_level=task_level)
    obs, _ = env.reset()
    if obs.shape[0] != 132: # Based on our OBS_SIZE constant
        print(f"❌ Observation size mismatch: {obs.shape[0]} != 132")
        return False
    print(f"✅ Matrix shapes are consistent ({obs.shape[0]} items).")

    print("\n💯 [SUCCESS] All Phase 3 Validation Checks Passed!")
    return True

if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
