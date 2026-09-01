#!/usr/bin/env python3
"""Batch evaluation for Random, Heuristic, and Trained ARGUS agents."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from generate_evaluation_plots import render_evaluation_plots, to_builtin
from environment import REWARD_SCHEMA_VERSION
from grader import GRADER_SCHEMA_VERSION
from inference_local import (
    DEFAULT_MODEL,
    LEVELS,
    HeuristicPolicy,
    LocalModelPolicy,
    RandomPolicy,
    SecurityFirstPolicy,
    run_episode,
    select_representative_episode,
    summarize_level_results,
    utc_now_iso,
)
from research_repro import (
    ReproducibilityError,
    append_event,
    canonical_json,
    compute_run_fingerprint,
    evaluation_seed_plan,
    git_metadata,
    load_spec,
    sha256_bytes,
    spec_sha256,
)
from tools.freeze_model_artifact import create_or_verify_model_manifest

AGENT_ORDER = ["random", "heuristic", "trained"]
AGENT_LABELS = {"random": "Random", "heuristic": "Heuristic", "trained": "Trained"}
CHECKPOINT_SCHEMA_VERSION = 1


def overall_summary(episodes: list[dict[str, Any]], level_label: str) -> dict[str, Any]:
    return summarize_level_results(level_label, episodes)


def build_seed_plan(episodes_per_level: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    flat = rng.sample(range(1_000_000), len(LEVELS) * episodes_per_level)
    return {
        level: flat[index * episodes_per_level:(index + 1) * episodes_per_level]
        for index, level in enumerate(LEVELS)
    }


def compact_episode_evidence(episode: dict[str, Any], schema: str) -> dict[str, Any]:
    """Remove repeated large prompts/states while retaining hashes and audit telemetry."""
    compact = dict(episode)
    compact["evidence_schema_version"] = schema
    compact_timeline = []
    large_fields = ("prompt", "messages", "observation_before", "observation_after")
    for record in episode.get("timeline", []):
        row = {key: value for key, value in record.items() if key not in large_fields}
        for key in large_fields:
            encoded = canonical_json(record.get(key)).encode("utf-8")
            row[f"{key}_sha256"] = sha256_bytes(encoded)
            row[f"{key}_bytes"] = len(encoded)
        compact_timeline.append(row)
    compact["timeline"] = compact_timeline
    return compact


def print_summary_table(agent_payloads: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print(f"{'AGENT':<12} | {'LEVEL':<8} | {'GRADE (+/- STD)':<18} | {'REWARD':>8} | {'REV':>7} | {'SEC':>7} | {'CAUGHT':>7}")
    print("-" * 92)
    for agent_key in AGENT_ORDER:
        label = AGENT_LABELS[agent_key]
        summaries = agent_payloads[agent_key]["summary"]
        for level in LEVELS:
            row = summaries[level]
            print(
                f"{label:<12} | {level:<8} | "
                f"{row['grade_mean']:.3f} +/- {row['grade_std']:.3f} | "
                f"{row['reward_mean']:>8.2f} | {row['revenue_mean']:>7.1f} | "
                f"{row['security_mean']:>7.1f} | {row['sleepers_caught_mean']:>7.2f}"
            )
        print("-" * 92)


def write_showcase_payload(agent_payloads: dict[str, dict[str, Any]], output_path: Path, model_ref: str) -> None:
    showcase = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "model": model_ref,
        "levels": {},
    }
    for level in LEVELS:
        showcase["levels"][level] = {
            "trained": select_representative_episode(agent_payloads["trained"]["episodes"][level]),
            "heuristic": select_representative_episode(agent_payloads["heuristic"]["episodes"][level]),
            "random": select_representative_episode(agent_payloads["random"]["episodes"][level]),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_builtin(showcase), indent=2), encoding="utf-8")


def build_payload_config(args: argparse.Namespace) -> dict[str, Any]:
    model_ref = args.model
    if getattr(args, "run_fingerprint", None) and Path(args.model).is_absolute():
        try:
            model_ref = Path(args.model).resolve().relative_to(Path(args.output).resolve().parent).as_posix()
        except ValueError:
            pass
    return {
        "model": model_ref,
        "episodes_per_level": args.episodes,
        "seed": args.seed,
        "timeline_level": args.timeline_level,
        "deterministic_trained_eval": not args.sampled,
        "max_steps": args.max_steps,
        "trained_policy": args.trained_policy,
        "model_revision": args.model_revision or None,
        "model_precision": args.model_precision,
        "model_prompt_max_tokens": args.model_prompt_max_tokens,
        "model_max_new_tokens": args.model_max_new_tokens,
        "episode_evidence_schema": getattr(args, "episode_evidence_schema", None),
        "model_artifact_identity": getattr(args, "model_artifact_identity", None),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "grader_schema_version": GRADER_SCHEMA_VERSION,
        "evaluation_split": getattr(args, "evaluation_split", "development"),
        "run_fingerprint": getattr(args, "run_fingerprint", None),
        "source_commit": getattr(args, "source_commit", None),
        "spec_sha256": getattr(args, "spec_sha256", None),
    }


def sidecar_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_suffix(output_path.suffix + suffix)


def episode_key(agent_key: str, level: str, episode_idx: int, seed: int) -> str:
    return f"{agent_key}|{level}|{episode_idx}|{seed}"


def iter_episode_plan(seed_plan: dict[str, list[int]]):
    for agent_key in AGENT_ORDER:
        for level in LEVELS:
            for episode_idx, episode_seed in enumerate(seed_plan[level], start=1):
                yield agent_key, level, episode_idx, episode_seed


def total_planned_episodes(episodes_per_level: int) -> int:
    return len(AGENT_ORDER) * len(LEVELS) * episodes_per_level


def append_episode_checkpoint(checkpoint_path: Path, record: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_builtin(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def checkpoint_config_matches(record_config: dict[str, Any], expected_config: dict[str, Any]) -> bool:
    return all(record_config.get(key) == value for key, value in expected_config.items())


def load_episode_checkpoints(
    checkpoint_path: Path, expected_config: dict[str, Any], *, strict: bool = False,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return records

    mismatched_lines: list[int] = []
    bad_lines: list[int] = []

    with checkpoint_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append(line_no)
                continue

            if record.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
                bad_lines.append(line_no)
                continue
            if not checkpoint_config_matches(record.get("config", {}), expected_config):
                mismatched_lines.append(line_no)
                continue

            agent_key = record.get("agent")
            level = record.get("level")
            episode_idx = record.get("episode_idx")
            episode_seed = record.get("seed")
            if agent_key not in AGENT_ORDER or level not in LEVELS or not isinstance(episode_idx, int):
                bad_lines.append(line_no)
                continue

            key = episode_key(agent_key, level, episode_idx, int(episode_seed))
            if key in records:
                bad_lines.append(line_no)
                continue
            records[key] = record

    if mismatched_lines:
        raise ReproducibilityError(
            "Existing evaluation checkpoint was created with a different configuration. "
            "Use the original source/spec/model/run directory or choose a fresh experiment. "
            f"Mismatched lines: {mismatched_lines[:8]}"
        )

    if bad_lines:
        if strict:
            raise ReproducibilityError(
                f"canonical evaluation checkpoint contains corrupt/duplicate records: {bad_lines[:8]}"
            )
        print(f"[WARN] Ignored incomplete or invalid checkpoint lines: {bad_lines[:8]}")

    return records


def load_completed_output_as_checkpoints(
    output_path: Path,
    expected_config: dict[str, Any],
    seed_plan: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if "agents" not in payload or not checkpoint_config_matches(payload.get("config", {}), expected_config):
        return {}

    records: dict[str, dict[str, Any]] = {}
    for agent_key, level, episode_idx, episode_seed in iter_episode_plan(seed_plan):
        try:
            episode = payload["agents"][agent_key]["episodes"][level][episode_idx - 1]
        except (KeyError, IndexError, TypeError):
            return {}
        record = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "completed_at": payload.get("created_at", utc_now_iso()),
            "config": expected_config,
            "agent": agent_key,
            "level": level,
            "episode_idx": episode_idx,
            "seed": episode_seed,
            "episode": episode,
        }
        records[episode_key(agent_key, level, episode_idx, episode_seed)] = record
    return records


def reconcile_canonical_evaluation_events(
    *, records: dict[str, dict[str, Any]], seed_plan: dict[str, list[int]],
    events_path: Path, output_name: str, model_artifact_identity: str,
    run_fingerprint: str, source_commit: str, split: str,
) -> None:
    """Repair only the checkpoint-written/event-not-written crash window; reject reruns."""
    try:
        existing_events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    except FileNotFoundError:
        existing_events = []
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("canonical event log is unreadable during evaluation resume") from exc
    relevant = [
        event for event in existing_events
        if event.get("event") == "evaluation_episode_completed"
        and event.get("output") == output_name
        and event.get("model_artifact_identity") == model_artifact_identity
    ]
    event_keys = [
        episode_key(event.get("agent"), event.get("level"), event.get("episode_idx"), event.get("seed"))
        for event in relevant
    ]
    if len(event_keys) != len(set(event_keys)):
        raise ReproducibilityError("canonical evaluation event history contains a rerun/duplicate episode")
    record_keys = set(records)
    if not set(event_keys) <= record_keys:
        raise ReproducibilityError("canonical evaluation history contains an episode without a durable checkpoint")
    seen = set(event_keys)
    for agent, level, episode_idx, episode_seed in iter_episode_plan(seed_plan):
        key = episode_key(agent, level, episode_idx, episode_seed)
        if key in record_keys and key not in seen:
            append_event(
                events_path,
                "evaluation_episode_completed",
                run_fingerprint=run_fingerprint,
                source_commit=source_commit,
                stage=split,
                output=output_name,
                model_artifact_identity=model_artifact_identity,
                agent=agent,
                level=level,
                episode_idx=episode_idx,
                seed=episode_seed,
                reconstructed_from_checkpoint=True,
            )
            seen.add(key)


def build_agent_payloads_from_checkpoints(
    records: dict[str, dict[str, Any]],
    seed_plan: dict[str, list[int]],
    episodes_per_level: int,
    *,
    require_complete: bool,
) -> dict[str, dict[str, Any]]:
    agent_payloads: dict[str, dict[str, Any]] = {
        agent_key: {"episodes": {level: [] for level in LEVELS}, "summary": {}, "overall": {}}
        for agent_key in AGENT_ORDER
    }

    for agent_key, level, episode_idx, episode_seed in iter_episode_plan(seed_plan):
        key = episode_key(agent_key, level, episode_idx, episode_seed)
        record = records.get(key)
        if record is None:
            if require_complete:
                raise RuntimeError(f"Missing checkpoint for {agent_key}/{level} episode {episode_idx}")
            continue
        episode = record["episode"]
        episode["agent"] = agent_key
        agent_payloads[agent_key]["episodes"][level].append(episode)

    for agent_key in AGENT_ORDER:
        agent_episodes_all: list[dict[str, Any]] = []
        for level in LEVELS:
            level_episodes = agent_payloads[agent_key]["episodes"][level]
            if require_complete and len(level_episodes) != episodes_per_level:
                raise RuntimeError(
                    f"Expected {episodes_per_level} completed episodes for {agent_key}/{level}, "
                    f"found {len(level_episodes)}"
                )
            if level_episodes:
                agent_payloads[agent_key]["summary"][level] = summarize_level_results(level, level_episodes)
                agent_episodes_all.extend(level_episodes)
        if agent_episodes_all:
            agent_payloads[agent_key]["overall"] = overall_summary(agent_episodes_all, "overall")

    return agent_payloads


def build_progress_payload(
    records: dict[str, dict[str, Any]],
    seed_plan: dict[str, list[int]],
    args: argparse.Namespace,
    output_path: Path,
    checkpoint_path: Path,
    progress_path: Path,
    plot_dir: Path,
    *,
    status: str,
    last_completed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = total_planned_episodes(args.episodes)
    completed = len(records)
    counts = {
        agent_key: {level: 0 for level in LEVELS}
        for agent_key in AGENT_ORDER
    }
    missing_next: dict[str, Any] | None = None

    for agent_key, level, episode_idx, episode_seed in iter_episode_plan(seed_plan):
        key = episode_key(agent_key, level, episode_idx, episode_seed)
        if key in records:
            counts[agent_key][level] += 1
        elif missing_next is None:
            missing_next = {
                "agent": agent_key,
                "level": level,
                "episode_idx": episode_idx,
                "seed": episode_seed,
            }

    return {
        "schema_version": "evaluation-progress-v1",
        "status": status,
        "updated_at": utc_now_iso(),
        "completed_episodes": completed,
        "total_episodes": total,
        "percent_complete": round(100.0 * completed / max(total, 1), 2),
        "episodes_per_level": args.episodes,
        "counts": counts,
        "next_episode": missing_next,
        "last_completed": last_completed,
        "output": str(output_path),
        "checkpoint_file": str(checkpoint_path),
        "progress_file": str(progress_path),
        "plot_dir": str(plot_dir),
        "resume_command_hint": (
            "Rerun the same full_evaluation.py command with the same --output path. "
            "Completed episodes will be skipped automatically."
        ),
    }


def write_progress(
    records: dict[str, dict[str, Any]],
    seed_plan: dict[str, list[int]],
    args: argparse.Namespace,
    output_path: Path,
    checkpoint_path: Path,
    progress_path: Path,
    plot_dir: Path,
    *,
    status: str,
    last_completed: dict[str, Any] | None = None,
) -> None:
    atomic_write_json(
        progress_path,
        build_progress_payload(
            records,
            seed_plan,
            args,
            output_path,
            checkpoint_path,
            progress_path,
            plot_dir,
            status=status,
            last_completed=last_completed,
        ),
    )


def write_final_payload(
    args: argparse.Namespace,
    seed_plan: dict[str, list[int]],
    records: dict[str, dict[str, Any]],
    output_path: Path,
    plot_dir: Path,
) -> dict[str, Any]:
    agent_payloads = build_agent_payloads_from_checkpoints(
        records,
        seed_plan,
        args.episodes,
        require_complete=True,
    )
    print_summary_table(agent_payloads)

    comparison_rows = []
    for agent_key in AGENT_ORDER:
        for level in LEVELS:
            row = dict(agent_payloads[agent_key]["summary"][level])
            row["agent"] = agent_key
            comparison_rows.append(row)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now_iso(),
        "config": build_payload_config(args),
        "seed_plan": seed_plan,
        "agents": agent_payloads,
        "comparison_rows": comparison_rows,
        "plots": {},
        "evaluation_progress": {
            "completed_episodes": len(records),
            "total_episodes": total_planned_episodes(args.episodes),
            "checkpointed": True,
        },
    }
    payload = to_builtin(payload)
    rendered_plots = render_evaluation_plots(payload, plot_dir, args.timeline_level)
    if getattr(args, "run_fingerprint", None):
        run_root = output_path.parent.resolve()
        payload["plots"] = {
            name: Path(value).resolve().relative_to(run_root).as_posix()
            for name, value in rendered_plots.items()
        }
    else:
        payload["plots"] = rendered_plots

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[*] Wrote evaluation results to {output_path}")
    return payload


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full Panopticon evaluation for random, heuristic, and trained agents")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo, local merged model, or adapter directory")
    parser.add_argument("--model-revision", default="", help="Immutable remote model revision")
    parser.add_argument("--model-precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--model-prompt-max-tokens", type=int, default=512)
    parser.add_argument("--model-max-new-tokens", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per level and per agent")
    parser.add_argument("--seed", type=int, default=42, help="Base seed used to create the fair comparison schedule")
    parser.add_argument("--output", default="evaluation_results.json", help="JSON output path")
    parser.add_argument("--plot-dir", default="plots", help="Directory for generated plots")
    parser.add_argument("--timeline-level", default="level_5", choices=LEVELS, help="Level used for representative timeline plot")
    parser.add_argument("--showcase-output", default="", help="Optional dashboard-ready showcase JSON path")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--verbose", action="store_true", help="Print per-turn logs during evaluation")
    parser.add_argument("--sampled", action="store_true", help="Use sampled decoding instead of deterministic decoding for the trained model")
    parser.add_argument("--trained-policy", choices=["model_raw", "model_repair", "security_first", "model"], default="model_repair", help="Policy in the trained slot; legacy model is an alias for model_repair")
    parser.add_argument("--checkpoint-file", default="", help="Optional episode JSONL checkpoint path")
    parser.add_argument("--progress-file", default="", help="Optional lightweight progress JSON path")
    parser.add_argument("--restart", action="store_true", help="Delete existing evaluation sidecars and start this output from scratch")
    parser.add_argument("--spec", help="Locked canonical JSON experiment specification")
    parser.add_argument("--run-fingerprint", help="Expected fingerprint from the run lock")
    parser.add_argument("--evaluation-split", choices=["development", "canonical", "confirmation"], default="development")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    args.source_commit = None
    args.spec_sha256 = None
    args.episode_evidence_schema = None
    args.model_artifact_identity = None
    active_spec = None
    if args.spec:
        active_spec, resolved_spec = load_spec(args.spec)
        git = git_metadata()
        if git["dirty"]:
            raise ReproducibilityError("canonical evaluation requires a clean source checkout")
        expected_fingerprint = compute_run_fingerprint(active_spec, git["commit"])
        if args.run_fingerprint != expected_fingerprint:
            raise ReproducibilityError("Evaluation fingerprint does not match the current source/spec")
        if args.restart:
            raise ReproducibilityError("canonical evaluation sidecars cannot be deleted with --restart")
        args.source_commit = git["commit"]
        args.spec_sha256 = spec_sha256(active_spec)
        args.episode_evidence_schema = active_spec["evaluation"]["episode_evidence_schema"]
        expected_seed = active_spec["evaluation"][f"{args.evaluation_split}_seed"]
        if args.seed != expected_seed:
            raise ReproducibilityError(f"{args.evaluation_split} evaluation requires seed {expected_seed}")
        for name, actual, expected in (
            ("episodes", args.episodes, active_spec["evaluation"]["episodes_per_level"]),
            ("max_steps", args.max_steps, active_spec["evaluation"]["max_steps"]),
            ("trained_policy", args.trained_policy, active_spec["evaluation"]["trained_policy"]),
            ("model_precision", args.model_precision, active_spec["evaluation"]["model_precision"]),
            ("model_prompt_max_tokens", args.model_prompt_max_tokens, active_spec["evaluation"]["model_prompt_max_tokens"]),
            ("model_max_new_tokens", args.model_max_new_tokens, active_spec["evaluation"]["model_max_new_tokens"]),
            ("timeline_level", args.timeline_level, active_spec["evaluation"]["timeline_level"]),
        ):
            if actual != expected:
                raise ReproducibilityError(f"Evaluation {name} differs from the locked specification")
        output_name = Path(args.output).name
        outputs = active_spec["outputs"]
        if args.evaluation_split == "confirmation":
            required_name = (
                outputs["confirmation_base_evaluation"]
                if args.model == active_spec["base_model"]["id"]
                else outputs["confirmation_evaluation"]
            )
        elif args.model == active_spec["base_model"]["id"]:
            required_name = outputs["canonical_base_evaluation"]
        else:
            required_name = outputs["canonical_candidate_evaluation"]
        if args.model == active_spec["base_model"]["id"]:
            args.model_artifact_identity = (
                f"hf:{active_spec['base_model']['id']}@{active_spec['base_model']['revision']}"
            )
        else:
            metadata_path = Path(args.model) / "run_metadata.json"
            try:
                model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ReproducibilityError(f"Candidate model lacks valid run metadata: {metadata_path}") from exc
            if model_metadata.get("run_fingerprint") != expected_fingerprint:
                raise ReproducibilityError("Candidate model belongs to a different run")
            frozen, _ = create_or_verify_model_manifest(Path(args.model).resolve().parent, resolved_spec, create=False)
            args.model_artifact_identity = f"sha256:{frozen['aggregate_sha256']}"
        if args.evaluation_split != "development" and output_name != required_name:
            raise ReproducibilityError(f"Locked {args.evaluation_split} output must be named {required_name}")
        if args.model_revision != active_spec["base_model"]["revision"]:
            raise ReproducibilityError("Evaluation model revision differs from the locked base-model revision")
        args.spec = str(resolved_spec)
    seed_plan = (
        evaluation_seed_plan(active_spec, args.evaluation_split)
        if active_spec else build_seed_plan(args.episodes, args.seed)
    )
    output_path = Path(args.output)
    plot_dir = Path(args.plot_dir)
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else sidecar_path(output_path, ".episodes.jsonl")
    progress_path = Path(args.progress_file) if args.progress_file else sidecar_path(output_path, ".progress.json")
    expected_config = build_payload_config(args)

    heuristic_policy = HeuristicPolicy()
    trained_policy: LocalModelPolicy | SecurityFirstPolicy | None = None

    try:
        if args.restart:
            for stale_path in (checkpoint_path, progress_path):
                if stale_path.exists():
                    stale_path.unlink()
            existing_records: dict[str, dict[str, Any]] = {}
            print("[*] Restart requested: existing evaluation sidecars were cleared.")
        else:
            existing_records = load_episode_checkpoints(
                checkpoint_path, expected_config, strict=bool(active_spec)
            )
            if active_spec and output_path.exists() and not checkpoint_path.exists():
                raise ReproducibilityError(
                    "canonical final evaluation exists without its episode checkpoint sidecar"
                )
            if not existing_records and not active_spec:
                existing_records = load_completed_output_as_checkpoints(output_path, expected_config, seed_plan)

        if active_spec:
            reconcile_canonical_evaluation_events(
                records=existing_records,
                seed_plan=seed_plan,
                events_path=output_path.parent / active_spec["outputs"]["events"],
                output_name=output_path.name,
                model_artifact_identity=args.model_artifact_identity,
                run_fingerprint=args.run_fingerprint,
                source_commit=args.source_commit,
                split=args.evaluation_split,
            )

        total = total_planned_episodes(args.episodes)
        print(f"[*] Evaluation checkpoint file: {checkpoint_path}")
        print(f"[*] Evaluation progress file: {progress_path}")
        print(f"[*] Resumable progress: {len(existing_records)}/{total} episodes already complete")

        write_progress(
            existing_records,
            seed_plan,
            args,
            output_path,
            checkpoint_path,
            progress_path,
            plot_dir,
            status="complete" if len(existing_records) == total else "running",
        )

        if len(existing_records) == total:
            print("[*] All planned episodes are already checkpointed. Regenerating final JSON/plots from saved episodes.")
            payload = write_final_payload(args, seed_plan, existing_records, output_path, plot_dir)
            write_progress(
                existing_records,
                seed_plan,
                args,
                output_path,
                checkpoint_path,
                progress_path,
                plot_dir,
                status="complete",
            )
            if args.showcase_output:
                showcase_path = Path(args.showcase_output)
                write_showcase_payload(payload["agents"], showcase_path, args.model)
                print(f"[*] Wrote showcase payload to {showcase_path}")
            if active_spec:
                append_event(
                    output_path.parent / active_spec["outputs"]["events"],
                    "evaluation_completed",
                    run_fingerprint=args.run_fingerprint,
                    source_commit=args.source_commit,
                    stage=args.evaluation_split,
                    output=output_path.name,
                    model_artifact_identity=args.model_artifact_identity,
                    completed_episodes=len(existing_records),
                    resumed_from_complete_checkpoint=True,
                )
            return

        records = dict(existing_records)

        def get_policy(agent_key: str, episode_seed: int):
            nonlocal trained_policy
            if agent_key == "random":
                return RandomPolicy(seed=episode_seed)
            if agent_key == "heuristic":
                return heuristic_policy
            if trained_policy is None:
                if args.trained_policy == "security_first":
                    print("[*] Using deterministic security-first supervisor in the trained slot", flush=True)
                    trained_policy = SecurityFirstPolicy(policy_name="trained")
                else:
                    intervention_mode = "raw" if args.trained_policy == "model_raw" else "repair"
                    if args.trained_policy == "model":
                        print("[!] --trained-policy model is legacy and means model_repair", flush=True)
                    print(
                        f"[*] Loading trained model ({intervention_mode}) only when needed: {args.model}",
                        flush=True,
                    )
                    trained_policy = LocalModelPolicy(
                        args.model,
                        deterministic=not args.sampled,
                        intervention_mode=intervention_mode,
                        max_seq_length=args.model_prompt_max_tokens,
                        max_new_tokens=args.model_max_new_tokens,
                        revision=args.model_revision or None,
                        precision=args.model_precision,
                    )
            return trained_policy

        for agent_key in AGENT_ORDER:
            print(f"\n[Agent] {AGENT_LABELS[agent_key]}")

            for level in LEVELS:
                print(f"  Level: {level}")

                for episode_idx, episode_seed in enumerate(seed_plan[level], start=1):
                    key = episode_key(agent_key, level, episode_idx, episode_seed)
                    if key in records:
                        print(
                            f"    [SKIP] Ep {episode_idx}/{args.episodes} | seed={episode_seed} | "
                            "checkpoint already exists",
                            flush=True,
                        )
                        continue

                    policy = get_policy(agent_key, episode_seed)
                    episode = run_episode(
                        policy,
                        task_level=level,
                        seed=episode_seed,
                        max_steps=args.max_steps,
                        verbose=args.verbose,
                    )
                    if active_spec:
                        episode = compact_episode_evidence(
                            episode, active_spec["evaluation"]["episode_evidence_schema"]
                        )
                    episode["agent"] = agent_key
                    record = {
                        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "completed_at": utc_now_iso(),
                        "config": expected_config,
                        "agent": agent_key,
                        "level": level,
                        "episode_idx": episode_idx,
                        "seed": episode_seed,
                        "episode": episode,
                    }
                    append_episode_checkpoint(checkpoint_path, record)
                    if active_spec:
                        append_event(
                            output_path.parent / active_spec["outputs"]["events"],
                            "evaluation_episode_completed",
                            run_fingerprint=args.run_fingerprint,
                            source_commit=args.source_commit,
                            stage=args.evaluation_split,
                            output=output_path.name,
                            model_artifact_identity=args.model_artifact_identity,
                            agent=agent_key,
                            level=level,
                            episode_idx=episode_idx,
                            seed=episode_seed,
                        )
                    records[key] = record
                    last_completed = {
                        "agent": agent_key,
                        "level": level,
                        "episode_idx": episode_idx,
                        "seed": episode_seed,
                        "grade": episode["grade"]["score"],
                        "reward": episode["total_reward"],
                        "security": episode["final_state"]["security_score"],
                    }
                    write_progress(
                        records,
                        seed_plan,
                        args,
                        output_path,
                        checkpoint_path,
                        progress_path,
                        plot_dir,
                        status="running",
                        last_completed=last_completed,
                    )
                    print(
                        f"    Ep {episode_idx}/{args.episodes} | seed={episode_seed} | "
                        f"grade={episode['grade']['score']:.3f} | reward={episode['total_reward']:.2f} | "
                        f"rev={episode['final_state']['enterprise_revenue']:.1f} | "
                        f"sec={episode['final_state']['security_score']:.1f} | "
                        f"saved={len(records)}/{total}"
                    )

        payload = write_final_payload(args, seed_plan, records, output_path, plot_dir)
        if active_spec:
            append_event(
                output_path.parent / active_spec["outputs"]["events"],
                "evaluation_completed",
                run_fingerprint=args.run_fingerprint,
                source_commit=args.source_commit,
                stage=args.evaluation_split,
                output=output_path.name,
                model_artifact_identity=args.model_artifact_identity,
                completed_episodes=len(records),
            )
        write_progress(
            records,
            seed_plan,
            args,
            output_path,
            checkpoint_path,
            progress_path,
            plot_dir,
            status="complete",
        )

        if args.showcase_output:
            showcase_path = Path(args.showcase_output)
            write_showcase_payload(payload["agents"], showcase_path, args.model)
            print(f"[*] Wrote showcase payload to {showcase_path}")
    finally:
        if trained_policy is not None:
            trained_policy.close()


if __name__ == "__main__":
    try:
        main()
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
