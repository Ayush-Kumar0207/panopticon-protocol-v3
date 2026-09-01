from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from benchmark_acceptance import evaluate_acceptance
from inference_local import summarize_level_results
from research_repro import (
    ReproducibilityError,
    assert_metadata_compatible,
    build_run_lock,
    capture_provenance,
    compute_run_fingerprint,
    evaluation_seed_plan,
    git_metadata,
    load_spec,
    sha256_file,
    spec_sha256,
    training_seed_plan,
    verify_seed_separation,
)
from tools.build_submission_bundle import build_bundle
from tools.freeze_model_artifact import create_or_verify_model_manifest
from tools.verify_training_artifacts import validate_evaluation, verify_run


SPEC_PATH = Path("training_specs/security_first_v5.json")
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_run_fingerprint_is_stable_and_sensitive():
    spec, _ = load_spec(SPEC_PATH)
    first = compute_run_fingerprint(spec, "a" * 40)
    assert first == compute_run_fingerprint(spec, "a" * 40)
    changed = json.loads(json.dumps(spec))
    changed["trajectory"]["training_seed"] += 1
    assert first != compute_run_fingerprint(changed, "a" * 40)
    assert first != compute_run_fingerprint(spec, "b" * 40)


def test_seed_partitions_are_exact_unique_and_disjoint():
    spec, _ = load_spec(SPEC_PATH)
    report = verify_seed_separation(spec)
    assert report["status"] == "verified-disjoint"
    plans = {"training": training_seed_plan(spec)}
    plans.update({split: evaluation_seed_plan(spec, split) for split in ("development", "canonical", "confirmation")})
    sets = {name: {seed for values in plan.values() for seed in values} for name, plan in plans.items()}
    assert max(sets["training"]) < min(sets["development"])
    assert max(sets["development"]) < min(sets["canonical"])
    assert max(sets["canonical"]) < min(sets["confirmation"])
    assert plans["canonical"] == evaluation_seed_plan(spec, "canonical")
    assert all(len(values) == len(set(values)) for values in sets.values())


@pytest.mark.parametrize("field,new_value", [
    ("seed", 43),
    ("base_model", {"id": "wrong/model", "revision": "bad"}),
    ("spec_sha256", "bad"),
])
def test_resume_rejects_changed_critical_metadata(field, new_value):
    expected = {"seed": 42, "base_model": {"id": "model", "revision": "rev"}, "spec_sha256": "ok"}
    actual = dict(expected)
    actual[field] = new_value
    with pytest.raises(ReproducibilityError):
        assert_metadata_compatible(actual, expected, expected.keys())


def test_resume_accepts_identical_metadata():
    value = {"seed": 42, "base_model": {"id": "model", "revision": "rev"}, "spec_sha256": "ok"}
    assert_metadata_compatible(dict(value), value, value.keys())


def test_provenance_excludes_secret_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_not_appear_anywhere")
    monkeypatch.setenv("TRAIN_ROOT", str(tmp_path))
    spec, resolved = load_spec(SPEC_PATH)
    payload = capture_provenance(spec, resolved, tmp_path)
    rendered = json.dumps(payload)
    assert "hf_this_must_not_appear_anywhere" not in rendered
    assert "HF_TOKEN" not in rendered
    assert payload["run_fingerprint"]


def _evaluation(
    spec: dict, fingerprint: str, source_commit: str, split: str, model_kind: str,
    root: Path, *, accepted: bool = True,
) -> dict:
    count = spec["evaluation"]["episodes_per_level"]
    seed_plan = evaluation_seed_plan(spec, split)
    model = spec["base_model"]["id"] if model_kind == "base" else spec["outputs"]["merged_model_dir"]
    frozen = (
        json.loads((root / spec["outputs"]["model_manifest"]).read_text(encoding="utf-8"))
        if model_kind == "candidate" else None
    )
    config = {
        "model": model,
        "episodes_per_level": count,
        "seed": spec["evaluation"][f"{split}_seed"],
        "timeline_level": "level_5",
        "deterministic_trained_eval": True,
        "max_steps": spec["evaluation"]["max_steps"],
        "trained_policy": spec["evaluation"]["trained_policy"],
        "model_revision": spec["base_model"]["revision"],
        "model_precision": spec["evaluation"]["model_precision"],
        "model_prompt_max_tokens": spec["evaluation"]["model_prompt_max_tokens"],
        "model_max_new_tokens": spec["evaluation"]["model_max_new_tokens"],
        "timeline_level": spec["evaluation"]["timeline_level"],
        "episode_evidence_schema": spec["evaluation"]["episode_evidence_schema"],
        "model_artifact_identity": (
            f"hf:{spec['base_model']['id']}@{spec['base_model']['revision']}"
            if model_kind == "base" else f"sha256:{frozen['aggregate_sha256']}"
        ),
        "reward_schema_version": spec["evaluation"]["reward_schema_version"],
        "grader_schema_version": spec["evaluation"]["grader_schema_version"],
        "evaluation_split": split,
        "run_fingerprint": fingerprint,
        "source_commit": source_commit,
        "spec_sha256": spec_sha256(spec),
    }
    agents = {}
    for agent in spec["evaluation"]["required_agents"]:
        agents[agent] = {"episodes": {}, "summary": {}, "overall": {}}
        all_agent_episodes = []
        for level, seeds in seed_plan.items():
            rejected_row = model_kind == "candidate" and not accepted and level == "level_5"
            score = 0.70 if model_kind == "candidate" else 0.60
            passed = not rejected_row
            security = 80.0 if rejected_row else (95.0 if model_kind == "candidate" else 90.0)
            caught = 2.0 if model_kind == "candidate" else 1.0
            episodes = []
            for item in seeds:
                model_provenance = (
                    {"prompt_tokens": 256, "token_truncated": False}
                    if agent == "trained" else {}
                )
                timeline = [{
                    "step_index": index,
                    "turn": index,
                    "phase": "fixture",
                    "reward": score / 2,
                    "metrics_after": {"enterprise_revenue": 100.0, "security_score": security},
                    "intervention_applied": False,
                    "model_provenance": model_provenance,
                    "prompt_sha256": "0" * 64,
                    "prompt_bytes": 1,
                    "messages_sha256": "1" * 64,
                    "messages_bytes": 1,
                    "observation_before_sha256": "2" * 64,
                    "observation_before_bytes": 1,
                    "observation_after_sha256": "3" * 64,
                    "observation_after_bytes": 1,
                } for index in range(2)]
                episode = {
                    "agent": agent,
                    "level": level,
                    "seed": item,
                    "steps": 2,
                    "total_reward": score,
                    "reward_history": [score / 2, score / 2],
                    "grade": {
                        "score": score,
                        "passed": passed,
                        "dimensions": {
                            name: score
                            for name in ("security", "revenue", "intelligence", "adaptability", "efficiency")
                        },
                    },
                    "final_state": {
                        "enterprise_revenue": 100.0,
                        "security_score": security,
                        "sleepers_caught": caught,
                        "sleepers_missed": 0.0,
                        "false_accusations": 0.0,
                        "invalid_actions": 0,
                    },
                    "evidence_schema_version": spec["evaluation"]["episode_evidence_schema"],
                    "timeline": timeline,
                    "provenance_summary": {
                        "interventions": 0,
                        "model_context": {
                            "model_turns": 2 if agent == "trained" else 0,
                            "prompt_tokens_max": 256 if agent == "trained" else 0,
                            "token_truncated_turns": 0,
                        },
                    },
                }
                episodes.append(episode)
            agents[agent]["episodes"][level] = episodes
            agents[agent]["summary"][level] = summarize_level_results(level, episodes)
            all_agent_episodes.extend(episodes)
        agents[agent]["overall"] = summarize_level_results("overall", all_agent_episodes)
    plot_names = (
        "benchmark_summary_table", "comparison_grades", "comparison_operations",
        "comparison_radar", "scenario_timeline", "reward_distributions",
        "reward_frontier", "reward_turn_dynamics",
    )
    plot_dir = root / f"plots_{split}_{model_kind}"
    plot_dir.mkdir(exist_ok=True)
    plots = {}
    for name in plot_names:
        (plot_dir / f"{name}.png").write_bytes(PNG_HEADER + b"fixture")
        plots[name] = f"plots_{split}_{model_kind}/{name}.png"
    total = len(spec["evaluation"]["required_agents"]) * len(spec["trajectory"]["levels"]) * count
    return {
        "schema_version": 1,
        "status": "complete",
        "config": config,
        "seed_plan": seed_plan,
        "agents": agents,
        "comparison_rows": [
            {**agents[agent]["summary"][level], "agent": agent}
            for agent in spec["evaluation"]["required_agents"]
            for level in spec["trajectory"]["levels"]
        ],
        "plots": plots,
        "evaluation_progress": {"completed_episodes": total, "total_episodes": total, "checkpointed": True},
    }


def _write_evaluation(path: Path, payload: dict) -> None:
    _write_json(path, payload)
    records = []
    config = payload["config"]
    for agent, agent_payload in payload["agents"].items():
        for level, episodes in agent_payload["episodes"].items():
            for index, episode in enumerate(episodes, start=1):
                records.append({
                    "checkpoint_schema_version": 1,
                    "config": config,
                    "agent": agent,
                    "level": level,
                    "episode_idx": index,
                    "seed": episode["seed"],
                    "episode": episode,
                })
    path.with_suffix(path.suffix + ".episodes.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8"
    )
    _write_json(path.with_suffix(path.suffix + ".progress.json"), {
        "status": "complete", "completed_episodes": len(records), "total_episodes": len(records),
    })


def _valid_run(tmp_path: Path, *, accepted: bool = True) -> tuple[Path, dict]:
    spec, _ = load_spec(SPEC_PATH)
    commit = git_metadata()["commit"]
    lock = build_run_lock(spec, commit)
    outputs = spec["outputs"]
    fingerprint = lock["run_fingerprint"]
    _write_json(tmp_path / outputs["run_lock"], lock)
    _write_json(tmp_path / outputs["provenance"], {
        "run_fingerprint": fingerprint, "git": {"commit": commit, "dirty": False},
        "spec_sha256": lock["spec_sha256"],
        "runtime": {
            "python": "3.12.0 fixture",
            "gpu": {"cuda_available": True, "name": "Fixture GPU", "vram_gb": 24.0},
            "packages": spec["dependencies"], "all_packages": {"fixture": "1.0"},
        },
        "deterministic_environment": spec["runtime"]["deterministic_environment"],
    })
    (tmp_path / outputs["provenance_markdown"]).write_text("# Fixture provenance\n", encoding="utf-8")
    _write_json(tmp_path / outputs["preflight_report"], {
        "passed": True,
        "run_fingerprint": fingerprint,
        "source_commit": commit,
        "spec_sha256": lock["spec_sha256"],
        "dependencies": spec["dependencies"],
        "runtime": {
            "python": "3.12.0 fixture",
            "gpu": {"cuda_available": True, "name": "Fixture GPU", "vram_gb": 24.0},
        },
        "cpu_smoke_only": False,
        "deterministic_environment": spec["runtime"]["deterministic_environment"],
        "seed_separation": verify_seed_separation(spec),
        "validations": [{"report": {
            "passed": True,
            "precision": spec["training"]["precision"],
            "assistant_only_loss": True,
            "valid_label_tokens": 8,
            "optimizer_seed": spec["training"]["optimization_seed_by_level"]["easy"],
            "gpu": "Fixture GPU",
            "loss": 1.0,
            "gradient_norm": 1.0,
            "peak_allocated_vram_gb": 3.0,
        }}],
    })
    _write_json(tmp_path / outputs["curriculum_state"], {
        "run_fingerprint": fingerprint,
        "source_commit": commit,
        "completed_levels": spec["trajectory"]["levels"],
        "seed": spec["trajectory"]["training_seed"],
        "episodes_per_level": spec["trajectory"]["episodes_per_level"],
    })

    trainer = {
        "max_sequence_length": spec["training"]["max_sequence_length"],
        "epochs": spec["training"]["epochs"],
        "per_device_batch_size": spec["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": spec["training"]["gradient_accumulation_steps"],
        "checkpoint_interval_steps": spec["training"]["checkpoint_interval_steps"],
        "checkpoint_retention": spec["training"]["checkpoint_retention"],
        "lora_r": spec["training"]["lora_r"],
        "lora_alpha": spec["training"]["lora_alpha"],
        "lora_dropout": spec["training"]["lora_dropout"],
        "learning_rate": spec["training"]["learning_rate"],
        "warmup_ratio": spec["training"]["warmup_ratio"],
        "weight_decay": spec["training"]["weight_decay"],
        "max_grad_norm": spec["training"]["max_grad_norm"],
        "optimizer": spec["training"]["optimizer"],
        "lr_scheduler_type": spec["training"]["lr_scheduler_type"],
        "dataloader_num_workers": spec["training"]["dataloader_num_workers"],
        "full_determinism": spec["training"]["full_determinism"],
        "completion_only_loss": spec["training"]["completion_only_loss"],
        "gradient_checkpointing": spec["training"]["gradient_checkpointing"],
        "precision": spec["training"]["precision"],
    }
    previous_model = spec["base_model"]["id"]
    seed_plan = training_seed_plan(spec)
    events = [{
        "event": "run_start", "run_fingerprint": fingerprint, "source_commit": commit, "stage": "training",
        "curriculum": True, "requested_levels": spec["trajectory"]["levels"],
        "episodes": spec["trajectory"]["episodes_per_level"], "model": spec["base_model"]["id"],
        "runtime_profile": spec["runtime"]["profile"], "max_seq_length": spec["training"]["max_sequence_length"],
        "epochs": spec["training"]["epochs"], "batch_size": spec["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": spec["training"]["gradient_accumulation_steps"],
        "seed": spec["trajectory"]["training_seed"], "merge": True,
    }]
    for level in spec["trajectory"]["levels"]:
        stage_dir = tmp_path / f"trl_model_{level}"
        stage_dir.mkdir()
        _write_json(stage_dir / "run_metadata.json", {
            "run_fingerprint": fingerprint, "source_commit": commit, "spec_sha256": lock["spec_sha256"],
            "status": "completed", "level": level, "input_model": previous_model, "base_model": spec["base_model"],
            "episodes_per_level": spec["trajectory"]["episodes_per_level"], "seed": spec["trajectory"]["training_seed"],
            "trajectory_schema_version": spec["trajectory"]["schema_version"],
            "trainer": {**trainer, "optimizer_seed": spec["training"]["optimization_seed_by_level"][level]},
        })
        _write_json(stage_dir / "adapter_config.json", {"r": spec["training"]["lora_r"]})
        _write_safetensors(stage_dir / "adapter_model.safetensors")
        data_path = tmp_path / f"training_data_{level}.jsonl"
        metrics_path = tmp_path / f"expert_metrics_{level}.json"
        data_path.write_text(json.dumps({"text": "<|im_start|>assistant\nlabelled example"}) + "\n", encoding="utf-8")
        _write_json(metrics_path, [{"seed": seed} for seed in seed_plan[level]])
        _write_json(tmp_path / f"training_data_{level}.meta.json", {
            "task_level": level, "num_episodes": spec["trajectory"]["episodes_per_level"],
            "num_examples": 1, "max_seq_length": spec["training"]["max_sequence_length"],
            "trajectory_schema_version": spec["trajectory"]["schema_version"], "model_name": previous_model,
            "seed": spec["trajectory"]["training_seed"], "runtime_profile": spec["runtime"]["profile"],
            "run_fingerprint": fingerprint, "source_commit": commit, "spec_sha256": lock["spec_sha256"],
            "episode_seeds": seed_plan[level],
            "episode_seed_plan_sha256": __import__("hashlib").sha256(
                json.dumps(seed_plan[level], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "training_data_sha256": sha256_file(data_path),
            "expert_metrics_sha256": sha256_file(metrics_path),
        })
        common = {"run_fingerprint": fingerprint, "source_commit": commit, "stage": "training", "level": level}
        events.extend([
            {**common, "event": "train_sanity", "valid_label_tokens": 8},
            {**common, "event": "train_progress", "loss": 0.5, "grad_norm": 1.0, "learning_rate": 0.00002},
            {**common, "event": "train_complete", "metrics": {"train_loss": 0.4}},
            {**common, "event": "level_complete"},
        ])
        previous_model = f"trl_model_{level}"
    events.extend([
        {"event": "merge_complete", "run_fingerprint": fingerprint, "source_commit": commit, "stage": "training"},
        {"event": "run_complete", "run_fingerprint": fingerprint, "source_commit": commit, "stage": "training"},
    ])
    (tmp_path / outputs["events"]).write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8"
    )

    model = tmp_path / outputs["merged_model_dir"]
    model.mkdir()
    _write_json(model / "run_metadata.json", {
        "run_fingerprint": fingerprint, "source_commit": commit, "spec_sha256": lock["spec_sha256"],
        "status": "merged", "adapter_path": f"trl_model_{spec['trajectory']['levels'][-1]}",
        "base_model": spec["base_model"],
    })
    _write_json(model / "config.json", {})
    _write_json(model / "tokenizer.json", {})
    _write_safetensors(model / "model.safetensors")
    create_or_verify_model_manifest(tmp_path, SPEC_PATH, create=True)
    plots = tmp_path / outputs["training_plots_dir"]
    plots.mkdir()
    (plots / "optimizer_diagnostics.png").write_bytes(PNG_HEADER + b"fixture")
    _write_json(plots / "optimizer_diagnostics.json", {
        "run_fingerprint": fingerprint, "levels": {level: {"points": 1} for level in spec["trajectory"]["levels"]},
    })

    for key, split, model_kind in (
        ("canonical_base_evaluation", "canonical", "base"),
        ("canonical_candidate_evaluation", "canonical", "candidate"),
        ("confirmation_base_evaluation", "confirmation", "base"),
        ("confirmation_evaluation", "confirmation", "candidate"),
    ):
        payload = _evaluation(spec, fingerprint, commit, split, model_kind, tmp_path, accepted=accepted)
        _write_evaluation(tmp_path / outputs[key], payload)
        for agent in spec["evaluation"]["required_agents"]:
            for level, seeds in payload["seed_plan"].items():
                for index, episode_seed in enumerate(seeds, start=1):
                    events.append({
                        "event": "evaluation_episode_completed",
                        "run_fingerprint": fingerprint,
                        "source_commit": commit,
                        "stage": split,
                        "output": outputs[key],
                        "model_artifact_identity": payload["config"]["model_artifact_identity"],
                        "agent": agent,
                        "level": level,
                        "episode_idx": index,
                        "seed": episode_seed,
                    })
        events.append({
            "event": "evaluation_completed",
            "run_fingerprint": fingerprint,
            "source_commit": commit,
            "stage": split,
            "output": outputs[key],
            "model_artifact_identity": payload["config"]["model_artifact_identity"],
            "completed_episodes": 300,
        })
    (tmp_path / outputs["events"]).write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8"
    )
    return tmp_path, spec


def test_artifact_validator_accepts_complete_run(tmp_path):
    run, _ = _valid_run(tmp_path)
    result = verify_run(run, SPEC_PATH, enforce_checkout=False)
    assert result["integrity_verified"] is True
    assert result["accepted"] is True
    assert result["scientific_status"] == "accepted"
    assert (run / "SHA256SUMS.txt").is_file()


def test_high_metric_cannot_override_security_rejection(tmp_path):
    run, _ = _valid_run(tmp_path, accepted=False)
    result = verify_run(run, SPEC_PATH, enforce_checkout=False)
    assert result["integrity_verified"] is True
    assert result["accepted"] is False
    assert result["scientific_status"] == "rejected"
    assert result["acceptance"]["canonical"]["failed_checks"]


def test_noisy_mean_improvement_fails_paired_confidence_gate(tmp_path):
    run, spec = _valid_run(tmp_path)
    base = json.loads((run / spec["outputs"]["canonical_base_evaluation"]).read_text(encoding="utf-8"))
    candidate = json.loads((run / spec["outputs"]["canonical_candidate_evaluation"]).read_text(encoding="utf-8"))
    for level in spec["trajectory"]["levels"]:
        episodes = candidate["agents"]["trained"]["episodes"][level]
        for index, episode in enumerate(episodes):
            episode["grade"]["score"] = 0.70 if index < 11 else 0.51
        candidate["agents"]["trained"]["summary"][level]["grade_mean"] = sum(
            episode["grade"]["score"] for episode in episodes
        ) / len(episodes)
    report = evaluate_acceptance(base, candidate)
    check = next(item for item in report["checks"] if item["name"] == "paired_grade_ci95_lower_gt_zero")
    assert report["candidate_macro_grade"] > report["base_macro_grade"]
    assert check["passed"] is False
    assert report["accepted"] is False


@pytest.mark.parametrize("target", ["model", "model_tamper", "results", "data", "event"])
def test_artifact_validator_rejects_missing_or_corrupt_artifacts(tmp_path, target):
    run, spec = _valid_run(tmp_path)
    if target == "model":
        (run / spec["outputs"]["merged_model_dir"] / "model.safetensors").unlink()
    elif target == "model_tamper":
        with (run / spec["outputs"]["merged_model_dir"] / "model.safetensors").open("ab") as handle:
            handle.write(b"tampered")
    elif target == "results":
        (run / spec["outputs"]["canonical_candidate_evaluation"]).unlink()
    elif target == "data":
        (run / "training_data_easy.jsonl").write_text("tampered\n", encoding="utf-8")
    else:
        events_path = run / spec["outputs"]["events"]
        first_evaluation = next(
            line for line in events_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "evaluation_episode_completed"
        )
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(first_evaluation + "\n")
    with pytest.raises(ReproducibilityError):
        verify_run(run, SPEC_PATH, enforce_checkout=False)


def test_artifact_validator_rejects_foreign_checkpoint(tmp_path):
    run, _ = _valid_run(tmp_path)
    checkpoint = run / "trl_model_easy" / "checkpoint-25"
    checkpoint.mkdir()
    _write_json(checkpoint / "run_metadata.json", {"run_fingerprint": "foreign", "level": "easy"})
    with pytest.raises(ReproducibilityError, match="checkpoint belongs to another experiment"):
        verify_run(run, SPEC_PATH, enforce_checkout=False)


def test_evaluation_rejects_wrong_fingerprint_and_incomplete_matrix(tmp_path):
    spec, _ = load_spec(SPEC_PATH)
    commit = "a" * 40
    payload = _evaluation(spec, "right", commit, "canonical", "base", tmp_path)
    path = tmp_path / "evaluation.json"
    payload["config"]["run_fingerprint"] = "wrong"
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError):
        validate_evaluation(path, spec, "right", "canonical", source_commit=commit, model_kind="base", require_sidecars=False)
    payload["config"]["run_fingerprint"] = "right"
    payload["agents"]["trained"]["episodes"]["easy"].pop()
    payload["evaluation_progress"]["completed_episodes"] -= 1
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError, match="incomplete"):
        validate_evaluation(path, spec, "right", "canonical", source_commit=commit, model_kind="base", require_sidecars=False)


def test_historical_registry_is_explicitly_noncanonical_and_safeguarded():
    registry = json.loads(Path("training_specs/historical_attempts.json").read_text(encoding="utf-8"))
    assert "no entry below is canonical" in registry["canonical_status"]
    ids = {item["id"] for item in registry["attempts"]}
    assert {
        "2026-04-loss-zero-debug-series", "2026-04-panopticon-ep50-v2-invocation-1",
        "2026-06-fixed-v3-ep20-logical-run", "2026-07-security-first-v5-raw-model",
        "2026-07-v6-pilot-r0-r3",
    } <= ids
    assert all(item.get("status") and item.get("safeguards") for item in registry["attempts"])


def test_submission_bundle_labels_status_and_external_weights(tmp_path):
    run, spec = _valid_run(tmp_path)
    bundle, index = build_bundle(run, SPEC_PATH, enforce_checkout=False)
    assert index["accepted"] is True
    assert any(item["path"].endswith(".safetensors") for item in index["external_required"])
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert "SUBMISSION_INDEX.json" in names
        assert spec["outputs"]["artifact_manifest"] in names
        assert not any(name.endswith(".safetensors") for name in names)
