#!/usr/bin/env python3
"""Fail-closed integrity and scientific-acceptance verifier for Panopticon."""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_acceptance import evaluate_acceptance
from tools.freeze_model_artifact import expected_model_manifest
from research_repro import (
    ReproducibilityError,
    atomic_write_json,
    build_run_lock,
    canonical_json,
    evaluation_seed_plan,
    git_metadata,
    load_spec,
    sha256_bytes,
    sha256_file,
    spec_sha256,
    training_seed_plan,
    verify_seed_separation,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"corrupt or missing JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReproducibilityError(f"expected a JSON object: {path}")
    return value


def verify_safetensors_header(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ReproducibilityError(f"truncated safetensors header: {path}")
            header_length = int.from_bytes(raw, "little")
            if header_length <= 2 or header_length > path.stat().st_size - 8:
                raise ReproducibilityError(f"invalid safetensors header length: {path}")
            json.loads(handle.read(header_length).decode("utf-8"))
    except ReproducibilityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"invalid safetensors artifact: {path}") from exc


def _assert_close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
        raise ReproducibilityError(f"non-finite or missing summary metric: {label}")
    if abs(float(actual) - expected) > 1e-9:
        raise ReproducibilityError(f"summary metric does not match raw episodes: {label}")


@functools.lru_cache(maxsize=512)
def _bootstrap_ci_cached(values: tuple[float, ...], samples: int) -> tuple[float, float]:
    rng = random.Random(0x50414E4F50544943 + len(values))
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def _bootstrap_ci(values: list[float], samples: int = 2000) -> tuple[float, float]:
    return _bootstrap_ci_cached(tuple(values), samples)


def _verify_raw_summary(
    episodes: list[dict[str, Any]], summary: dict[str, Any], label: str, expected_level: str,
) -> None:
    try:
        values = {
            "grade_mean": [float(item["grade"]["score"]) for item in episodes],
            "reward_mean": [float(item["total_reward"]) for item in episodes],
            "revenue_mean": [float(item["final_state"]["enterprise_revenue"]) for item in episodes],
            "security_mean": [float(item["final_state"]["security_score"]) for item in episodes],
            "sleepers_caught_mean": [float(item["final_state"]["sleepers_caught"]) for item in episodes],
            "sleepers_missed_mean": [float(item["final_state"]["sleepers_missed"]) for item in episodes],
            "false_accusations_mean": [float(item["final_state"]["false_accusations"]) for item in episodes],
            "invalid_actions_mean": [float(item["final_state"]["invalid_actions"]) for item in episodes],
            "steps_mean": [float(item["steps"]) for item in episodes],
        }
        pass_rate = sum(bool(item["grade"]["passed"]) for item in episodes) / len(episodes)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ReproducibilityError(f"raw episode schema is incomplete: {label}") from exc
    for key, series in values.items():
        if not all(math.isfinite(value) for value in series):
            raise ReproducibilityError(f"non-finite raw episode metric: {label}/{key}")
        _assert_close(summary.get(key), statistics.mean(series), f"{label}/{key}")
        std_key = key.replace("_mean", "_std")
        _assert_close(summary.get(std_key), statistics.pstdev(series), f"{label}/{std_key}")
    _assert_close(summary.get("pass_rate"), pass_rate, f"{label}/pass_rate")
    _assert_close(summary.get("grade_median"), statistics.median(values["grade_mean"]), f"{label}/grade_median")
    grade_ci = _bootstrap_ci(values["grade_mean"])
    _assert_close(summary.get("grade_ci95_low"), grade_ci[0], f"{label}/grade_ci95_low")
    _assert_close(summary.get("grade_ci95_high"), grade_ci[1], f"{label}/grade_ci95_high")
    dimensions = ("security", "revenue", "intelligence", "adaptability", "efficiency")
    for dimension in dimensions:
        try:
            series = [float(item["grade"]["dimensions"][dimension]) for item in episodes]
            actual = summary["grader_dimensions"][dimension]
        except (KeyError, TypeError, ValueError) as exc:
            raise ReproducibilityError(f"grader dimension summary is incomplete: {label}/{dimension}") from exc
        _assert_close(actual.get("mean"), statistics.mean(series), f"{label}/{dimension}/mean")
        _assert_close(actual.get("std"), statistics.pstdev(series), f"{label}/{dimension}/std")
    if summary.get("level") != expected_level:
        raise ReproducibilityError(f"summary level mismatch: {label}")
    if summary.get("episodes") != len(episodes):
        raise ReproducibilityError(f"summary episode count mismatch: {label}")


def _verify_episode_trace(
    episode: dict[str, Any], spec: dict[str, Any], label: str, run_root: Path
) -> list[Path]:
    if episode.get("evidence_schema_version") != spec["evaluation"]["episode_evidence_schema"]:
        raise ReproducibilityError(f"episode evidence schema mismatch: {label}")
    timeline = episode.get("timeline")
    if not isinstance(timeline, list) or len(timeline) != episode.get("steps"):
        raise ReproducibilityError(f"episode timeline/step mismatch: {label}")
    hash_fields = ("prompt", "messages", "observation_before", "observation_after")
    evidence_files: list[Path] = []
    for step_index, row in enumerate(timeline):
        if any(key in row for key in hash_fields):
            raise ReproducibilityError(f"uncompacted canonical episode field: {label}/step-{step_index}")
        for key in hash_fields:
            digest = row.get(f"{key}_sha256")
            size = row.get(f"{key}_bytes")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ReproducibilityError(f"invalid evidence hash: {label}/step-{step_index}/{key}")
            if not isinstance(size, int) or size <= 0:
                raise ReproducibilityError(f"invalid evidence byte count: {label}/step-{step_index}/{key}")
            relative = row.get(f"{key}_artifact")
            if not isinstance(relative, str):
                raise ReproducibilityError(f"missing raw-evidence artifact: {label}/step-{step_index}/{key}")
            expected_parent = Path(spec["outputs"]["raw_evidence_dir"]) / "sha256" / digest[:2]
            raw_relative = Path(relative)
            if raw_relative.parent != expected_parent or raw_relative.name != f"{digest}.json.gz":
                raise ReproducibilityError(f"noncanonical raw-evidence path: {label}/step-{step_index}/{key}")
            artifact = (run_root / raw_relative).resolve()
            try:
                artifact.relative_to(run_root.resolve())
            except ValueError as exc:
                raise ReproducibilityError(f"raw-evidence path escapes run directory: {relative}") from exc
            try:
                replayed = gzip.decompress(artifact.read_bytes())
            except (OSError, EOFError) as exc:
                raise ReproducibilityError(f"raw-evidence artifact is missing/corrupt: {relative}") from exc
            if len(replayed) != size or sha256_bytes(replayed) != digest:
                raise ReproducibilityError(f"raw-evidence replay hash mismatch: {relative}")
            try:
                decoded = json.loads(replayed)
            except json.JSONDecodeError as exc:
                raise ReproducibilityError(f"raw-evidence JSON is invalid: {relative}") from exc
            if canonical_json(decoded).encode("utf-8") != replayed:
                raise ReproducibilityError(f"raw-evidence encoding is not canonical: {relative}")
            evidence_files.append(artifact)
    provenance = episode.get("provenance_summary", {})
    context = provenance.get("model_context", {})
    derived = {
        "interventions": sum(bool(row.get("intervention_applied")) for row in timeline),
        "model_turns": sum(bool(row.get("model_provenance")) for row in timeline),
        "prompt_tokens_max": max(((row.get("model_provenance") or {}).get("prompt_tokens", 0) for row in timeline), default=0),
        "token_truncated_turns": sum(bool((row.get("model_provenance") or {}).get("token_truncated")) for row in timeline),
    }
    actual = {
        "interventions": provenance.get("interventions"),
        "model_turns": context.get("model_turns"),
        "prompt_tokens_max": context.get("prompt_tokens_max"),
        "token_truncated_turns": context.get("token_truncated_turns"),
    }
    if actual != derived:
        raise ReproducibilityError(f"episode provenance does not match its trace: {label}")
    return evidence_files


def _verify_plot_manifest(payload: dict[str, Any], root: Path, label: str) -> list[Path]:
    plots = payload.get("plots")
    if not isinstance(plots, dict) or not plots:
        raise ReproducibilityError(f"{label} evaluation plot manifest is empty")
    verified = []
    expected_names = {
        "benchmark_summary_table", "comparison_grades", "comparison_operations",
        "comparison_radar", "scenario_timeline", "reward_distributions",
        "reward_frontier", "reward_turn_dynamics",
    }
    if set(plots) != expected_names:
        raise ReproducibilityError(f"{label} evaluation plot manifest has missing/extra entries")
    for name, raw_path in plots.items():
        raw = Path(raw_path)
        if raw.is_absolute():
            raise ReproducibilityError(f"{label} plot path is not portable: {name}")
        expected_relative = f"plots_{label.replace('/', '_')}/{name}.png"
        if raw.as_posix() != expected_relative:
            raise ReproducibilityError(f"{label} plot path differs from the canonical layout: {name}")
        path = (root / raw).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReproducibilityError(f"{label} plot escapes the run directory: {name}") from exc
        if not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ReproducibilityError(f"{label} plot is missing or not PNG: {name}")
        verified.append(path)
    return verified


def validate_evaluation(
    path: Path,
    spec: dict[str, Any],
    fingerprint: str,
    split: str,
    *,
    source_commit: str | None = None,
    model_kind: str | None = None,
    require_sidecars: bool = True,
    require_plots: bool = True,
) -> dict[str, Any]:
    payload = read_json(path)
    config = payload.get("config", {})
    expected_seed = int(spec["evaluation"][f"{split}_seed"])
    expected_episodes = int(spec["evaluation"]["episodes_per_level"])
    expected_total = len(spec["evaluation"]["required_agents"]) * len(spec["trajectory"]["levels"]) * expected_episodes
    expected_config = {
        "run_fingerprint": fingerprint,
        "evaluation_split": split,
        "seed": expected_seed,
        "episodes_per_level": expected_episodes,
        "max_steps": spec["evaluation"]["max_steps"],
        "deterministic_trained_eval": spec["evaluation"]["deterministic_decoding"],
        "trained_policy": spec["evaluation"]["trained_policy"],
        "reward_schema_version": spec["evaluation"]["reward_schema_version"],
        "grader_schema_version": spec["evaluation"]["grader_schema_version"],
        "model_revision": spec["base_model"]["revision"],
        "model_precision": spec["evaluation"]["model_precision"],
        "model_prompt_max_tokens": spec["evaluation"]["model_prompt_max_tokens"],
        "model_max_new_tokens": spec["evaluation"]["model_max_new_tokens"],
        "timeline_level": spec["evaluation"]["timeline_level"],
        "episode_evidence_schema": spec["evaluation"]["episode_evidence_schema"],
        "spec_sha256": spec_sha256(spec),
    }
    if source_commit is not None:
        expected_config["source_commit"] = source_commit
    if model_kind == "base":
        expected_config["model_artifact_identity"] = f"hf:{spec['base_model']['id']}@{spec['base_model']['revision']}"
    elif model_kind == "candidate":
        frozen = read_json(path.parent / spec["outputs"]["model_manifest"])
        expected_config["model_artifact_identity"] = f"sha256:{frozen.get('aggregate_sha256')}"
    differences = {key: {"expected": value, "actual": config.get(key)} for key, value in expected_config.items() if config.get(key) != value}
    if differences:
        raise ReproducibilityError(f"{split} evaluation configuration mismatch: {canonical_json(differences)}")
    if payload.get("status") != "complete":
        raise ReproducibilityError(f"canonical evaluation is not marked complete: {path}")
    if model_kind == "base" and config.get("model") != spec["base_model"]["id"]:
        raise ReproducibilityError(f"{split} base evaluation used the wrong model")
    if model_kind == "candidate":
        expected_model = path.parent / spec["outputs"]["merged_model_dir"]
        try:
            raw_model = Path(config.get("model", ""))
            actual_model = (path.parent / raw_model).resolve() if not raw_model.is_absolute() else raw_model.resolve()
        except (OSError, TypeError) as exc:
            raise ReproducibilityError(f"{split} candidate model path is invalid") from exc
        if actual_model != expected_model.resolve():
            raise ReproducibilityError(f"{split} candidate evaluation used another model directory")
    expected_plan = evaluation_seed_plan(spec, split)
    if payload.get("seed_plan") != expected_plan:
        raise ReproducibilityError(f"{split} evaluation seed plan differs from the frozen plan")
    progress = payload.get("evaluation_progress", {})
    if progress.get("completed_episodes") != expected_total or progress.get("total_episodes") != expected_total:
        raise ReproducibilityError(f"canonical evaluation is incomplete ({progress.get('completed_episodes')}/{expected_total} episodes)")

    seen: set[tuple[str, str, int]] = set()
    final_by_key: dict[str, dict[str, Any]] = {}
    raw_evidence_files: list[Path] = []
    for agent in spec["evaluation"]["required_agents"]:
        agent_payload = payload.get("agents", {}).get(agent)
        if not isinstance(agent_payload, dict):
            raise ReproducibilityError(f"missing agent {agent} in {path}")
        all_agent_episodes = []
        for level in spec["trajectory"]["levels"]:
            planned = expected_plan[level]
            episodes = agent_payload.get("episodes", {}).get(level, [])
            if len(episodes) != expected_episodes:
                raise ReproducibilityError(f"canonical evaluation is incomplete ({agent}/{level}: {len(episodes)}/{expected_episodes})")
            for index, (seed, episode) in enumerate(zip(planned, episodes, strict=True), start=1):
                key_tuple = (agent, level, int(seed))
                if key_tuple in seen:
                    raise ReproducibilityError(f"duplicate episode identity in {path}: {key_tuple}")
                seen.add(key_tuple)
                if int(episode.get("seed", -1)) != int(seed) or episode.get("level") != level:
                    raise ReproducibilityError(f"episode identity mismatch: {agent}/{level}/{index}")
                raw_evidence_files.extend(
                    _verify_episode_trace(
                        episode, spec, f"{agent}/{level}/{index}/{seed}", path.parent
                    )
                )
                provenance = episode.get("provenance_summary", {})
                context = provenance.get("model_context", {})
                if agent == "trained":
                    if provenance.get("interventions") != 0:
                        raise ReproducibilityError(f"raw-model evaluation contains an intervention: {level}/{seed}")
                    if context.get("model_turns") != episode.get("steps"):
                        raise ReproducibilityError(f"model-context coverage mismatch: {level}/{seed}")
                    if context.get("prompt_tokens_max", 10**9) > spec["evaluation"]["model_prompt_max_tokens"]:
                        raise ReproducibilityError(f"model prompt exceeded the frozen context: {level}/{seed}")
                    if context.get("token_truncated_turns") != 0:
                        raise ReproducibilityError(f"model evaluation used token truncation: {level}/{seed}")
                final_by_key[f"{agent}|{level}|{index}|{seed}"] = episode
            summary = agent_payload.get("summary", {}).get(level, {})
            _verify_raw_summary(episodes, summary, f"{agent}/{level}", level)
            all_agent_episodes.extend(episodes)
        _verify_raw_summary(
            all_agent_episodes, agent_payload.get("overall", {}), f"{agent}/overall", "overall"
        )

    if len(seen) != expected_total:
        raise ReproducibilityError(f"canonical evaluation is incomplete ({len(seen)}/{expected_total} episodes)")
    expected_rows = [
        {**payload["agents"][agent]["summary"][level], "agent": agent}
        for agent in spec["evaluation"]["required_agents"]
        for level in spec["trajectory"]["levels"]
    ]
    if payload.get("comparison_rows") != expected_rows:
        raise ReproducibilityError(f"{split} comparison rows do not match verified summaries")
    sidecar_files: list[Path] = []
    if require_sidecars:
        checkpoint_path = path.with_suffix(path.suffix + ".episodes.jsonl")
        progress_path = path.with_suffix(path.suffix + ".progress.json")
        try:
            checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReproducibilityError(f"evaluation episode checkpoint is missing: {checkpoint_path}") from exc
        checkpoint_records = {}
        for line_number, line in enumerate(checkpoint_lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReproducibilityError(f"corrupt evaluation checkpoint at line {line_number}: {checkpoint_path}") from exc
            if record.get("checkpoint_schema_version") != 1:
                raise ReproducibilityError(f"unsupported evaluation checkpoint schema at line {line_number}")
            key = f"{record.get('agent')}|{record.get('level')}|{record.get('episode_idx')}|{record.get('seed')}"
            if key in checkpoint_records:
                raise ReproducibilityError(f"duplicate evaluation checkpoint key: {key}")
            if record.get("config") != config or record.get("episode") != final_by_key.get(key):
                raise ReproducibilityError(f"evaluation checkpoint/final JSON mismatch: {key}")
            checkpoint_records[key] = record
        if set(checkpoint_records) != set(final_by_key):
            raise ReproducibilityError(f"evaluation checkpoint matrix is incomplete ({len(checkpoint_records)}/{expected_total})")
        side_progress = read_json(progress_path)
        if side_progress.get("status") != "complete" or side_progress.get("completed_episodes") != expected_total or side_progress.get("total_episodes") != expected_total:
            raise ReproducibilityError(f"evaluation progress sidecar is incomplete: {progress_path}")
        sidecar_files.extend([checkpoint_path, progress_path])
    plot_files = _verify_plot_manifest(payload, path.parent, f"{split}/{model_kind}") if require_plots else []
    payload["_verified_files"] = [path, *sidecar_files, *plot_files, *raw_evidence_files]
    return payload


def _validate_training(root: Path, spec: dict[str, Any], lock: dict[str, Any]) -> tuple[list[Path], int]:
    outputs = spec["outputs"]
    fingerprint = lock["run_fingerprint"]
    provenance = read_json(root / outputs["provenance"])
    if (
        provenance.get("run_fingerprint") != fingerprint
        or provenance.get("git", {}).get("dirty")
        or provenance.get("git", {}).get("commit") != lock["source_commit"]
        or provenance.get("spec_sha256") != lock["spec_sha256"]
        or provenance.get("runtime", {}).get("packages") != spec["dependencies"]
        or not provenance.get("runtime", {}).get("all_packages")
        or provenance.get("deterministic_environment") != spec["runtime"].get("deterministic_environment", {})
    ):
        raise ReproducibilityError("provenance is dirty or belongs to another run")
    preflight = read_json(root / outputs["preflight_report"])
    if (
        not preflight.get("passed")
        or preflight.get("run_fingerprint") != fingerprint
        or preflight.get("source_commit") != lock["source_commit"]
        or preflight.get("spec_sha256") != lock["spec_sha256"]
        or preflight.get("dependencies") != spec["dependencies"]
        or preflight.get("cpu_smoke_only")
        or preflight.get("deterministic_environment") != spec["runtime"].get("deterministic_environment", {})
    ):
        raise ReproducibilityError("canonical GPU preflight evidence is missing or incompatible")
    runtime = preflight.get("runtime", {})
    gpu = runtime.get("gpu", {})
    if (
        not str(runtime.get("python", "")).startswith("3.11.")
        or gpu.get("cuda_available") is not True
        or not isinstance(gpu.get("vram_gb"), (int, float))
        or float(gpu["vram_gb"]) < float(spec["runtime"]["minimum_vram_gb"])
        or gpu.get("torch_build") != (
            f"{spec['runtime']['torch_distribution_version']}+{spec['runtime']['torch_local_build']}"
        )
        or str(gpu.get("cuda")) != str(spec["runtime"]["torch_cuda_runtime"])
        or not isinstance(gpu.get("compute_capability"), list)
        or not gpu.get("compute_capability")
        or int(gpu["compute_capability"][0]) < int(spec["runtime"]["minimum_compute_capability_major"])
        or gpu.get("bf16_supported") is not True
    ):
        raise ReproducibilityError("canonical Python/CUDA/BF16/VRAM preflight evidence is incompatible")
    if preflight.get("seed_separation", {}).get("status") != "verified-disjoint":
        raise ReproducibilityError("training/evaluation seed separation was not verified")
    probes = [item.get("report", {}) for item in preflight.get("validations", []) if item.get("report")]
    first_optimizer_seed = spec["training"]["optimization_seed_by_level"][spec["trajectory"]["levels"][0]]
    if not probes or not all(
        item.get("passed")
        and item.get("precision") == spec["training"]["precision"]
        and item.get("assistant_only_loss") is True
        and int(item.get("valid_label_tokens", 0)) > 0
        and item.get("optimizer_seed") == first_optimizer_seed
        and item.get("gpu") == gpu.get("name")
        and all(
            isinstance(item.get(key), (int, float))
            and math.isfinite(float(item[key]))
            and float(item[key]) > 0
            for key in ("loss", "gradient_norm", "peak_allocated_vram_gb")
        )
        for item in probes
    ):
        raise ReproducibilityError("canonical GPU memory/numerics probe evidence is missing")

    state = read_json(root / outputs["curriculum_state"])
    if state.get("run_fingerprint") != fingerprint or state.get("source_commit") != lock["source_commit"]:
        raise ReproducibilityError("curriculum state belongs to another experiment")
    if state.get("completed_levels") != spec["trajectory"]["levels"]:
        raise ReproducibilityError("curriculum state does not show every canonical level in order")
    if state.get("seed") != spec["trajectory"]["training_seed"] or state.get("episodes_per_level") != spec["trajectory"]["episodes_per_level"]:
        raise ReproducibilityError("curriculum state seed/episode count differs from the frozen spec")

    important = [
        root / outputs["run_lock"], root / outputs["provenance"], root / outputs["provenance_markdown"],
        root / outputs["preflight_report"], root / outputs["curriculum_state"], root / outputs["events"],
    ]
    expected_training_seeds = training_seed_plan(spec)
    trainer_base = {
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
    for level in spec["trajectory"]["levels"]:
        trainer = {
            **trainer_base,
            "optimizer_seed": spec["training"]["optimization_seed_by_level"][level],
        }
        stage_dir = root / f"trl_model_{level}"
        metadata = read_json(stage_dir / "run_metadata.json")
        expected_fields = {
            "run_fingerprint": fingerprint, "source_commit": lock["source_commit"],
            "spec_sha256": lock["spec_sha256"], "status": "completed", "level": level,
            "input_model": previous_model, "base_model": spec["base_model"],
            "episodes_per_level": spec["trajectory"]["episodes_per_level"],
            "seed": spec["trajectory"]["training_seed"],
            "trajectory_schema_version": spec["trajectory"]["schema_version"], "trainer": trainer,
        }
        differences = {key: {"expected": value, "actual": metadata.get(key)} for key, value in expected_fields.items() if metadata.get(key) != value}
        if differences:
            raise ReproducibilityError(f"stage metadata mismatch for {level}: {canonical_json(differences)}")
        adapter_weights = sorted(stage_dir.glob("*.safetensors"))
        if not adapter_weights or not (stage_dir / "adapter_config.json").is_file():
            raise ReproducibilityError(f"completed stage artifact is missing adapter files: {level}")
        for item in adapter_weights:
            verify_safetensors_header(item)
        for checkpoint in stage_dir.glob("checkpoint-*"):
            checkpoint_metadata = read_json(checkpoint / "run_metadata.json")
            checkpoint_expected = {key: value for key, value in expected_fields.items() if key != "status"}
            if any(checkpoint_metadata.get(key) != value for key, value in checkpoint_expected.items()):
                raise ReproducibilityError(f"checkpoint belongs to another experiment: {checkpoint}")
        data_path = root / f"training_data_{level}.jsonl"
        metrics_path = root / f"expert_metrics_{level}.json"
        meta_path = root / f"training_data_{level}.meta.json"
        meta = read_json(meta_path)
        expected_meta = {
            "task_level": level,
            "num_episodes": spec["trajectory"]["episodes_per_level"],
            "max_seq_length": spec["training"]["max_sequence_length"],
            "trajectory_schema_version": spec["trajectory"]["schema_version"],
            "model_name": previous_model,
            "seed": spec["trajectory"]["training_seed"],
            "runtime_profile": spec["runtime"]["profile"],
            "run_fingerprint": fingerprint,
            "source_commit": lock["source_commit"],
            "spec_sha256": lock["spec_sha256"],
            "episode_seeds": expected_training_seeds[level],
            "episode_seed_plan_sha256": sha256_bytes(canonical_json(expected_training_seeds[level]).encode("utf-8")),
        }
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise ReproducibilityError(f"expert-data identity/seed plan mismatch: {level}")
        if meta.get("training_data_sha256") != sha256_file(data_path) or meta.get("expert_metrics_sha256") != sha256_file(metrics_path):
            raise ReproducibilityError(f"expert-data artifact hash mismatch: {level}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, list) or [int(item.get("seed", -1)) for item in metrics] != expected_training_seeds[level]:
            raise ReproducibilityError(f"expert-metrics episode plan mismatch: {level}")
        line_count = 0
        try:
            with data_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ReproducibilityError(f"corrupt training JSONL for {level} at line {line_number}") from exc
                    if not isinstance(row.get("text"), str) or "<|im_start|>assistant" not in row["text"]:
                        raise ReproducibilityError(f"missing assistant-labelled training example for {level} at line {line_number}")
                    line_count += 1
        except OSError as exc:
            raise ReproducibilityError(f"training JSONL is missing: {level}") from exc
        if line_count != meta.get("num_examples") or line_count == 0:
            raise ReproducibilityError(f"training example count mismatch: {level}")
        important.extend(item for item in stage_dir.rglob("*") if item.is_file())
        important.extend([data_path, metrics_path, meta_path])
        previous_model = f"trl_model_{level}"

    model_dir = root / outputs["merged_model_dir"]
    model_metadata = read_json(model_dir / "run_metadata.json")
    expected_model_metadata = {
        "run_fingerprint": fingerprint,
        "source_commit": lock["source_commit"],
        "spec_sha256": lock["spec_sha256"],
        "status": "merged",
        "adapter_path": f"trl_model_{spec['trajectory']['levels'][-1]}",
        "base_model": spec["base_model"],
    }
    if any(model_metadata.get(key) != value for key, value in expected_model_metadata.items()):
        raise ReproducibilityError("merged model identity metadata is incompatible")
    for name in spec.get("required_model_files", []):
        if not (model_dir / name).is_file():
            raise ReproducibilityError(f"required merged-model file is missing: {name}")
    tokenizer_options = spec.get("required_tokenizer_any_of", [])
    if not any((model_dir / name).is_file() for name in tokenizer_options):
        raise ReproducibilityError("merged model has no recognized tokenizer vocabulary")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise ReproducibilityError("merged model has no safetensors weights")
    for weight in weights:
        verify_safetensors_header(weight)
    important.extend(item for item in model_dir.rglob("*") if item.is_file())
    model_manifest_path = root / outputs["model_manifest"]
    frozen_actual = read_json(model_manifest_path)
    frozen_expected = expected_model_manifest(root, spec)
    if canonical_json(frozen_actual) != canonical_json(frozen_expected):
        raise ReproducibilityError("merged model bytes changed after the model was frozen")
    important.append(model_manifest_path)

    events = []
    try:
        event_lines = (root / outputs["events"]).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproducibilityError("structured event log is missing") from exc
    for line_number, line in enumerate(event_lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReproducibilityError(f"corrupt event JSONL at line {line_number}") from exc
        if event.get("run_fingerprint") != fingerprint or event.get("source_commit") != lock["source_commit"]:
            raise ReproducibilityError(f"event identity mismatch at line {line_number}")
        for name in ("loss", "grad_norm", "learning_rate"):
            value = event.get(name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ReproducibilityError(f"non-finite {name} at event line {line_number}")
        events.append(event)
    run_starts = [event for event in events if event.get("event") == "run_start"]
    if not run_starts:
        raise ReproducibilityError("run_start event is missing")
    expected_start = {
        "curriculum": True, "requested_levels": spec["trajectory"]["levels"],
        "episodes": spec["trajectory"]["episodes_per_level"], "model": spec["base_model"]["id"],
        "runtime_profile": spec["runtime"]["profile"], "max_seq_length": spec["training"]["max_sequence_length"],
        "epochs": spec["training"]["epochs"], "batch_size": spec["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": spec["training"]["gradient_accumulation_steps"],
        "seed": spec["trajectory"]["training_seed"], "merge": True,
    }
    for event in run_starts:
        if any(event.get(key) != value for key, value in expected_start.items()):
            raise ReproducibilityError("run_start history contains configuration drift")
    for level in spec["trajectory"]["levels"]:
        if not any(event.get("event") == "level_complete" and event.get("level") == level for event in events):
            raise ReproducibilityError(f"level completion event is missing: {level}")
        sanity = [event for event in events if event.get("event") == "train_sanity" and event.get("level") == level]
        if not sanity or not all(int(event.get("valid_label_tokens", 0)) > 0 for event in sanity):
            raise ReproducibilityError(f"positive-label training sanity evidence is missing: {level}")
        completions = [event for event in events if event.get("event") == "train_complete" and event.get("level") == level]
        if not completions or not all(float(event.get("metrics", {}).get("train_loss", 0)) > 0 for event in completions):
            raise ReproducibilityError(f"positive finite training loss evidence is missing: {level}")
    for required in ("merge_complete", "run_complete"):
        if not any(event.get("event") == required for event in events):
            raise ReproducibilityError(f"required lifecycle event is missing: {required}")

    plot_dir = root / outputs["training_plots_dir"]
    plot = plot_dir / "optimizer_diagnostics.png"
    plot_data = read_json(plot_dir / "optimizer_diagnostics.json")
    if not plot.is_file() or plot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n" or plot_data.get("run_fingerprint") != fingerprint:
        raise ReproducibilityError("training diagnostics plot/metadata is invalid")
    if set(plot_data.get("levels", {})) != set(spec["trajectory"]["levels"]):
        raise ReproducibilityError("training diagnostics omit curriculum levels")
    important.extend([plot, plot_dir / "optimizer_diagnostics.json"])
    return important, len(events)


def _verify_evaluation_events(root: Path, spec: dict[str, Any], lock: dict[str, Any]) -> None:
    events_path = root / spec["outputs"]["events"]
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    frozen = read_json(root / spec["outputs"]["model_manifest"])
    identities = {
        "base": f"hf:{spec['base_model']['id']}@{spec['base_model']['revision']}",
        "candidate": f"sha256:{frozen['aggregate_sha256']}",
    }
    definitions = (
        ("canonical_base_evaluation", "canonical", "base"),
        ("canonical_candidate_evaluation", "canonical", "candidate"),
        ("confirmation_base_evaluation", "confirmation", "base"),
        ("confirmation_evaluation", "confirmation", "candidate"),
    )
    recognized = set()
    for output_key, split, model_kind in definitions:
        output_name = spec["outputs"][output_key]
        identity = identities[model_kind]
        expected = [
            (agent, level, index, seed)
            for agent in spec["evaluation"]["required_agents"]
            for level, seeds in evaluation_seed_plan(spec, split).items()
            for index, seed in enumerate(seeds, start=1)
        ]
        relevant = [
            event for event in events
            if event.get("event") == "evaluation_episode_completed"
            and event.get("output") == output_name
            and event.get("model_artifact_identity") == identity
        ]
        actual = [
            (event.get("agent"), event.get("level"), event.get("episode_idx"), event.get("seed"))
            for event in relevant
        ]
        if actual != expected:
            raise ReproducibilityError(
                f"evaluation lifecycle is missing, duplicated, rerun, or out of order: {output_name} ({len(actual)}/{len(expected)})"
            )
        completions = [
            event for event in events
            if event.get("event") == "evaluation_completed"
            and event.get("output") == output_name
            and event.get("model_artifact_identity") == identity
        ]
        if not completions or any(event.get("completed_episodes") != len(expected) for event in completions):
            raise ReproducibilityError(f"evaluation completion event is missing/incompatible: {output_name}")
        recognized.update(id(event) for event in relevant)
    all_episode_events = [event for event in events if event.get("event") == "evaluation_episode_completed"]
    if len(recognized) != len(all_episode_events):
        raise ReproducibilityError("event log contains an unrecognized development/foreign evaluation episode")


def verify_run(
    run_dir: str | Path,
    spec_path: str | Path,
    *,
    require_evaluations: bool = True,
    enforce_checkout: bool = True,
) -> dict[str, Any]:
    spec, _ = load_spec(spec_path)
    verify_seed_separation(spec)
    root = Path(run_dir).resolve()
    outputs = spec["outputs"]
    lock = read_json(root / outputs["run_lock"])
    if enforce_checkout:
        source = git_metadata(ROOT)
        if source["dirty"] or source["commit"] != lock.get("source_commit"):
            raise ReproducibilityError("source commit does not match run identity or checkout is dirty")
    expected_lock = build_run_lock(spec, lock.get("source_commit", ""))
    if canonical_json(lock) != canonical_json(expected_lock):
        raise ReproducibilityError("run lock does not match the supplied source/spec identity")
    important, event_count = _validate_training(root, spec, lock)
    acceptance_reports = {}
    accepted = None
    if require_evaluations:
        definitions = (
            ("canonical_base_evaluation", "canonical", "base"),
            ("canonical_candidate_evaluation", "canonical", "candidate"),
            ("confirmation_base_evaluation", "confirmation", "base"),
            ("confirmation_evaluation", "confirmation", "candidate"),
        )
        evaluated = {}
        for key, split, model_kind in definitions:
            payload = validate_evaluation(
                root / outputs[key], spec, lock["run_fingerprint"], split,
                source_commit=lock["source_commit"], model_kind=model_kind,
            )
            important.extend(payload.pop("_verified_files"))
            evaluated[(split, model_kind)] = payload
        _verify_evaluation_events(root, spec, lock)
        for split, output_key in (("canonical", "canonical_acceptance_report"), ("confirmation", "confirmation_acceptance_report")):
            try:
                report = evaluate_acceptance(evaluated[(split, "base")], evaluated[(split, "candidate")])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReproducibilityError(f"cannot recompute {split} acceptance from raw-verified summaries") from exc
            report.update({
                "schema_version": 1, "evaluation_split": split,
                "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"],
                "interpretation": "A positive aggregate metric cannot override any failed security/acceptance check.",
            })
            report_path = root / outputs[output_key]
            atomic_write_json(report_path, report)
            important.append(report_path)
            acceptance_reports[split] = report
        accepted = all(report["accepted"] for report in acceptance_reports.values())

    unique = sorted(set(path for path in important if path.is_file()))
    files = [{
        "path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path),
    } for path in unique]
    manifest = {
        "schema_version": 1,
        "integrity_verified": True,
        "scientific_status": "not-evaluated" if accepted is None else ("accepted" if accepted else "rejected"),
        "accepted": accepted,
        "run_fingerprint": lock["run_fingerprint"],
        "source_commit": lock["source_commit"],
        "spec_sha256": lock["spec_sha256"],
        "events": event_count,
        "files": files,
        "acceptance": {split: {"accepted": report["accepted"], "failed_checks": [check for check in report["checks"] if not check["passed"]]} for split, report in acceptance_reports.items()},
        "interpretation": "Integrity verifies identity/completeness. Scientific success additionally requires every canonical and confirmation acceptance check.",
    }
    manifest_path = root / outputs["artifact_manifest"]
    atomic_write_json(manifest_path, manifest)
    checksum_items = [*files, {
        "path": outputs["artifact_manifest"], "size": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path),
    }]
    (root / outputs["checksums"]).write_text(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in checksum_items) + "\n",
        encoding="utf-8",
    )
    return {
        "integrity_verified": True, "accepted": accepted,
        "scientific_status": manifest["scientific_status"],
        "run_fingerprint": lock["run_fingerprint"], "files_hashed": len(checksum_items),
        "acceptance": manifest["acceptance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    parser.add_argument("--training-only", action="store_true", help="Verify through merged model before evaluation")
    args = parser.parse_args()
    try:
        report = verify_run(args.run_dir, args.spec, require_evaluations=not args.training_only)
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2))
    if report["accepted"] is False:
        print("EVIDENCE VERIFIED, BUT SCIENTIFIC ACCEPTANCE FAILED: this run is non-canonical evidence.", file=sys.stderr)
        raise SystemExit(2)
    if report["accepted"] is True:
        print("VERIFICATION PASS: integrity and every canonical/confirmation acceptance gate passed.")
    else:
        print("TRAINING-ONLY INTEGRITY PASS: no scientific acceptance claim was evaluated.")


if __name__ == "__main__":
    main()
