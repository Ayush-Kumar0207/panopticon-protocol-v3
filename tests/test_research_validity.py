"""Validity tests for action coverage, information boundaries, and HYDRA policies."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

import v6_evaluation
from environment import Environment
from gym_wrapper import NUM_SUB_ACTIONS, NUM_TARGETS, OpenEnvGymWrapper
from hydra_neural import HydraPolicyNetwork, NeuralHydraPolicy, save_hydra_checkpoint
from hydra_policy import ScriptedHydraPolicy
from inference_local import HeuristicPolicy, RandomPolicy, run_episode
from models import HiddenWorkerState, validate_action
from panopticon_bench.seed_plan import canonical_sha256, load_seed_plan
from train_hydra import training_episode_spec
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


def test_neural_hydra_checkpoint_save_is_atomic_and_rolling() -> None:
    tmp_path = Path("research_paper") / f".hydra-checkpoint-test-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=False, exist_ok=False)
    checkpoint = tmp_path / "hydra.pt"
    try:
        model = HydraPolicyNetwork()
        save_hydra_checkpoint(checkpoint, model, metadata={"episode": 1})
        first_size = checkpoint.stat().st_size
        save_hydra_checkpoint(checkpoint, model, metadata={"episode": 2})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert checkpoint.stat().st_size > 0
        assert first_size > 0
        assert payload["metadata"]["episode"] == 2
        assert not checkpoint.with_suffix(".pt.tmp").exists()
        assert list(tmp_path.glob("*.pt")) == [checkpoint]
    finally:
        checkpoint.unlink(missing_ok=True)
        checkpoint.with_suffix(".pt.tmp").unlink(missing_ok=True)
        tmp_path.rmdir()


def test_neural_hydra_training_schedule_uses_only_development_seeds() -> None:
    plan = load_seed_plan("research_paper/data/seed_plans/v6_seed_plan.json")
    development = plan["splits"]["development"]
    prohibited = {
        seed
        for split_name in ("pilot", "final")
        for seeds in plan["splits"][split_name].values()
        for seed in seeds
    }
    first = []
    second = []
    for episode_number in range(1, 101):
        spec = training_episode_spec(
            training_seed=2026071701,
            episode_number=episode_number,
            levels=LEVELS,
            argus_population=["random", "heuristic", "security_first"],
            development_split=development,
        )
        level, environment_seed, argus_name = spec
        assert environment_seed in development[level]
        assert environment_seed not in prohibited
        assert argus_name in {"random", "heuristic", "security_first"}
        first.append(spec)
        second.append(
            training_episode_spec(
                training_seed=2026071701,
                episode_number=episode_number,
                levels=LEVELS,
                argus_population=["random", "heuristic", "security_first"],
                development_split=development,
            )
        )
    assert first == second


def test_v6_evaluator_persists_failure_manifest(monkeypatch) -> None:
    output_dir = Path("research_paper") / f".v6-failure-test-{uuid.uuid4().hex}"

    def fail_model_load(*_args, **_kwargs):
        raise RuntimeError("synthetic model-load failure")

    monkeypatch.setattr(v6_evaluation, "build_policy", fail_model_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v6_evaluation.py",
            "--split",
            "pilot",
            "--policies",
            "model_raw",
            "--model",
            "unused-local-model",
            "--max-episodes-per-level",
            "1",
            "--output-dir",
            str(output_dir),
            "--resume",
        ],
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic model-load failure"):
            v6_evaluation.main()
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["failed_policy"] == "model_raw"
        assert manifest["completed_episodes"] == 0
        assert manifest["model_prompt_max_tokens"] == 512
        assert manifest["model_max_new_tokens"] == 128
        assert manifest["prompt_fitting"] == "structured_compaction_no_token_truncation"
        assert manifest["failure"]["type"] == "RuntimeError"
        assert "synthetic model-load failure" in manifest["failure"]["message"]
    finally:
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
            output_dir.rmdir()


def test_level_5_random_episode_is_canonically_json_serializable() -> None:
    """Regression for the exact pilot episode that stopped the first Colab run."""
    episode = run_episode(
        RandomPolicy(seed=922094758),
        task_level="level_5",
        seed=922094758,
        max_steps=300,
        verbose=False,
    )
    compact = v6_evaluation.compact_trace(episode)
    assert type(compact["grade"]["passed"]) is bool
    assert compact["provenance_summary"]["model_context"]["model_turns"] == 0
    assert compact["provenance_summary"]["model_context"]["token_truncated_turns"] == 0
    assert len(canonical_sha256(compact)) == 64


def test_structured_prompt_fitting_compacts_without_token_truncation() -> None:
    from argus_llm import LocalArgusModel

    class WordTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return " ".join(message["content"] for message in messages)

        def __call__(self, text, *, add_special_tokens=False, verbose=False):
            assert add_special_tokens is False
            assert verbose is False
            return {"input_ids": text.split()}

    runtime = LocalArgusModel.__new__(LocalArgusModel)
    runtime.max_seq_length = 220
    runtime.tokenizer = WordTokenizer()
    observation = Environment(seed=1).reset(task_level="level_5", seed=1)
    prompt = runtime.build_prompt(observation)
    assert prompt.original_token_count > runtime.max_seq_length
    assert prompt.final_token_count <= runtime.max_seq_length
    assert prompt.compaction_level > 0


def test_v6_imports_only_compatible_non_model_baselines() -> None:
    root = Path("research_paper") / f".v6-reuse-test-{uuid.uuid4().hex}"
    source_dir = root / "legacy"
    source_dir.mkdir(parents=True)
    target_path = root / "fresh" / "episodes.jsonl"
    try:
        frozen_config = {
            "seed_plan_sha256": "seed-plan",
            "split": "pilot",
            "episodes_per_level": 1,
            "partial_split": True,
            "policies": ["random", "model_raw"],
            "hydra_policy": "scripted_memory_v1",
            "hydra_checkpoint_sha256": None,
            "max_steps": 300,
            "trace_level": "compact",
            "reward_schema_version": "reward-v1",
            "grader_schema_version": "grader-v1",
        }
        (source_dir / "manifest.json").write_text(
            json.dumps(frozen_config), encoding="utf-8"
        )

        def source_record(key: str, policy: str) -> dict:
            episode = {"policy": policy, "grade": {"score": 0.5, "passed": False}}
            return {
                "record_schema_version": "panopticon-evaluation-v6",
                "config_sha256": "legacy-config",
                "episode_key": key,
                "policy_stratum": policy,
                "episode": episode,
                "episode_sha256": canonical_sha256(episode),
            }

        baseline_key = "random|easy|1|123"
        provisional_key = "model_raw|easy|1|123"
        source_records = [
            source_record(baseline_key, "random"),
            source_record(provisional_key, "model_raw"),
        ]
        (source_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in source_records),
            encoding="utf-8",
        )
        result = v6_evaluation.import_reusable_baselines(
            source_dir / "episodes.jsonl",
            target_path,
            config_sha256="new-config",
            frozen_config=frozen_config,
            expected_keys={baseline_key},
        )
        imported = [
            json.loads(line)
            for line in target_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert result["imported_records"] == 1
        assert result["ignored_provisional_model_records"] == 1
        assert [record["episode_key"] for record in imported] == [baseline_key]
        assert imported[0]["config_sha256"] == "new-config"
        assert imported[0]["record_schema_version"] == v6_evaluation.V6_SCHEMA_VERSION
        assert (
            imported[0]["baseline_reuse_provenance"]["source_config_sha256"]
            == "legacy-config"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
