"""Validity tests for action coverage, information boundaries, and HYDRA policies."""

from __future__ import annotations

import numpy as np
import torch

from environment import Environment
from gym_wrapper import NUM_SUB_ACTIONS, NUM_TARGETS, OpenEnvGymWrapper
from hydra_neural import NeuralHydraPolicy
from hydra_policy import ScriptedHydraPolicy
from inference_local import HeuristicPolicy
from models import HiddenWorkerState, validate_action
from train_rl import PanopticonAgent


LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]


def test_joint_action_mask_contains_only_semantically_valid_actions() -> None:
    for level in LEVELS:
        env = OpenEnvGymWrapper(level)
        env.reset(seed=77)
        mask = env.get_action_mask()
        assert mask.shape == (8, NUM_TARGETS, NUM_SUB_ACTIONS)
        assert mask.any()
        for encoded in np.argwhere(mask):
            decoded = env._decode_action(encoded, strict=True)
            assert decoded is not None
            assert validate_action(decoded, env._last_obs)[0]


def test_hardest_level_can_address_workers_beyond_legacy_target_cap() -> None:
    env = OpenEnvGymWrapper("level_5")
    env.reset(seed=77)
    mask = env.get_action_mask()
    assert len(env._last_obs.workers) == 10
    assert mask[:, 8, :].any()
    assert mask[:, 9, :].any()


def test_masked_ppo_sampling_never_emits_invalid_action() -> None:
    env = OpenEnvGymWrapper("level_5")
    flattened, _ = env.reset(seed=91)
    model = PanopticonAgent()
    observation = torch.tensor(flattened)
    mask = torch.tensor(env.get_action_mask())
    for _ in range(50):
        action, log_probability, _, _ = model.get_action_and_value(
            observation, action_mask=mask
        )
        decoded = env._decode_action(action.numpy(), strict=True)
        assert decoded is not None and validate_action(decoded, env._last_obs)[0]
        _, recomputed, _, _ = model.get_action_and_value(
            observation, action=action, action_mask=mask
        )
        assert torch.allclose(log_probability, recomputed)


def test_argus_observation_never_contains_hydra_hidden_truth() -> None:
    env = Environment(seed=12)
    observation = env.reset("level_5", seed=12)
    policy = HeuristicPolicy()
    for _ in range(12):
        result = env.step(policy.act(observation).action)
        observation = result.observation
    assert env.state.total_sleepers_spawned >= 1
    assert any(worker.is_sleeper for worker in env.state.workers)
    assert all(worker.is_sleeper is False for worker in observation.workers)
    assert all(worker.generation == 0 for worker in observation.workers)
    assert all(worker.hidden_state == HiddenWorkerState.CLEAN.value for worker in observation.workers)
    assert "hydra_memory" not in observation.model_dump()


def test_scripted_hydra_is_explicitly_rule_based_and_reproducible() -> None:
    assert ScriptedHydraPolicy.policy_name == "scripted_memory_v1"
    first = Environment(seed=222, hydra_policy=ScriptedHydraPolicy())
    second = Environment(seed=222, hydra_policy=ScriptedHydraPolicy())
    policy_a, policy_b = HeuristicPolicy(), HeuristicPolicy()
    obs_a = first.reset("hard", seed=222)
    obs_b = second.reset("hard", seed=222)
    for _ in range(50):
        obs_a = first.step(policy_a.act(obs_a).action).observation
        obs_b = second.step(policy_b.act(obs_b).action).observation
    assert first.state.model_dump() == second.state.model_dump()


def test_neural_hydra_uses_declared_boundary_and_records_trainable_decisions() -> None:
    hydra = NeuralHydraPolicy(deterministic=False, record_gradients=True, device="cpu")
    env = Environment(seed=333, hydra_policy=hydra)
    argus = HeuristicPolicy()
    observation = env.reset("easy", seed=333)
    while not env.state.done:
        observation = env.step(argus.act(observation).action).observation
    assert hydra.trajectory
    log_probability = hydra.episode_log_probability()
    assert log_probability is not None and log_probability.requires_grad
    assert env.hydra_policy_name == "neural_hydra_v1"
