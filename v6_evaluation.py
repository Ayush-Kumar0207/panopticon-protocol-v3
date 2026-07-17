#!/usr/bin/env python3
"""Preregistered, provenance-aware Panopticon V6 evaluation harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from environment import REWARD_SCHEMA_VERSION
from grader import GRADER_SCHEMA_VERSION
from inference_local import (
    DEFAULT_MODEL,
    HeuristicPolicy,
    LocalModelPolicy,
    RandomPolicy,
    SecurityFirstPolicy,
    run_episode,
    summarize_level_results,
)
from panopticon_bench.seed_plan import canonical_sha256, load_seed_plan


V6_SCHEMA_VERSION = "panopticon-evaluation-v6"
POLICIES = {"random", "heuristic", "security_first", "model_raw", "model_repair"}
INTERVENTION_LEVEL = {
    "random": "raw",
    "heuristic": "raw",
    "security_first": "supervisor",
    "model_raw": "raw",
    "model_repair": "semantic_repair",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
    }


def parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(policies) - POLICIES
    if unknown or not policies or len(policies) != len(set(policies)):
        raise argparse.ArgumentTypeError(
            f"policies must be a unique comma-separated subset of {sorted(POLICIES)}"
        )
    return policies


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compact_trace(episode: dict[str, Any]) -> dict[str, Any]:
    for turn in episode["timeline"]:
        for field in ("observation_before", "observation_after", "prompt", "messages"):
            value = turn.pop(field, None)
            if value not in (None, "", []):
                turn[f"{field}_sha256"] = canonical_sha256(value)
    return episode


def load_completed(path: Path, config_sha256: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"Existing JSONL line {line_number} belongs to a different frozen configuration"
            )
        key = record["episode_key"]
        if key in completed:
            raise RuntimeError(f"Duplicate episode key in checkpoint JSONL: {key}")
        episode = record["episode"]
        expected = canonical_sha256(episode)
        if expected != record.get("episode_sha256"):
            raise RuntimeError(f"Episode content digest mismatch at JSONL line {line_number}")
        completed[key] = record
    return completed


def build_policy(policy_name: str, args: argparse.Namespace, seed: int | None = None):
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    if policy_name == "heuristic":
        return HeuristicPolicy()
    if policy_name == "security_first":
        return SecurityFirstPolicy(policy_name="security_first")
    if policy_name == "model_raw":
        return LocalModelPolicy(
            args.model, deterministic=not args.sampled, intervention_mode="raw"
        )
    if policy_name == "model_repair":
        return LocalModelPolicy(
            args.model, deterministic=not args.sampled, intervention_mode="repair"
        )
    raise ValueError(policy_name)


def summarize(records: dict[str, dict[str, Any]], policies: list[str], levels: list[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records.values():
        grouped[record["policy_stratum"]][record["level"]].append(record["episode"])

    payload: dict[str, Any] = {}
    for policy in policies:
        level_summaries: dict[str, Any] = {}
        all_episodes: list[dict[str, Any]] = []
        for level in levels:
            episodes = grouped[policy][level]
            all_episodes.extend(episodes)
            if episodes:
                level_summaries[level] = summarize_level_results(level, episodes)
        provenance = [episode["provenance_summary"] for episode in all_episodes]
        turns = sum(item["turns"] for item in provenance)
        payload[policy] = {
            "episodes": len(all_episodes),
            "levels": level_summaries,
            "provenance": {
                "turns": turns,
                "parse_failures": sum(item["parse_failures"] for item in provenance),
                "raw_semantic_invalid": sum(item["raw_semantic_invalid"] for item in provenance),
                "interventions": sum(item["interventions"] for item in provenance),
                "intervention_rate": (
                    sum(item["interventions"] for item in provenance) / turns if turns else 0.0
                ),
            },
        }
    return payload


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-plan",
        default="research_paper/data/seed_plans/v6_seed_plan.json",
    )
    parser.add_argument("--split", choices=["pilot", "development", "final"], default="pilot")
    parser.add_argument(
        "--policies",
        type=parse_policies,
        default=parse_policies("random,heuristic,security_first"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default="argus-v5")
    parser.add_argument(
        "--hydra-checkpoint",
        default="",
        help="Optional learned HYDRA checkpoint; default is the scripted-memory baseline",
    )
    parser.add_argument("--output-dir", default="research_paper/data/v6_runs/pilot")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-episodes-per-level", type=int, default=0)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--trace-level", choices=["compact", "full"], default="compact")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    if any(name.startswith("model_") for name in args.policies) and not args.model:
        raise SystemExit("--model is required for model_raw/model_repair strata")

    seed_plan = load_seed_plan(args.seed_plan)
    split_plan = seed_plan["splits"][args.split]
    levels = seed_plan["levels"]
    per_level = seed_plan["split_sizes_per_level"][args.split]
    limit = args.max_episodes_per_level or per_level
    if not 1 <= limit <= per_level:
        raise SystemExit(f"--max-episodes-per-level must be 1..{per_level} for split {args.split}")

    output_dir = Path(args.output_dir)
    episodes_path = output_dir / "episodes.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory is non-empty; choose a new path or pass --resume: {output_dir}")

    frozen_config = {
        "schema_version": V6_SCHEMA_VERSION,
        "seed_plan_sha256": seed_plan["seed_plan_sha256"],
        "seed_plan_path": str(Path(args.seed_plan)),
        "split": args.split,
        "episodes_per_level": limit,
        "partial_split": limit != per_level,
        "policies": args.policies,
        "model_ref": args.model if any(name.startswith("model_") for name in args.policies) else None,
        "model_label": args.model_label,
        "hydra_policy": "neural_hydra_v1" if args.hydra_checkpoint else "scripted_memory_v1",
        "hydra_checkpoint_path": args.hydra_checkpoint or None,
        "hydra_checkpoint_sha256": file_sha256(args.hydra_checkpoint) if args.hydra_checkpoint else None,
        "deterministic_decoding": not args.sampled,
        "max_steps": args.max_steps,
        "trace_level": args.trace_level,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "grader_schema_version": GRADER_SCHEMA_VERSION,
    }
    config_sha256 = canonical_sha256(frozen_config)
    manifest = {
        **frozen_config,
        "config_sha256": config_sha256,
        "created_at_utc": utc_now(),
        "git": git_metadata(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "status": "running",
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("config_sha256") != config_sha256:
            raise SystemExit("Existing manifest has a different frozen configuration")
        manifest["created_at_utc"] = existing_manifest["created_at_utc"]

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    records = load_completed(episodes_path, config_sha256)
    hydra_policy = None
    if args.hydra_checkpoint:
        from hydra_neural import NeuralHydraPolicy

        hydra_policy = NeuralHydraPolicy(args.hydra_checkpoint, deterministic=True)
    planned = len(args.policies) * len(levels) * limit
    print(f"V6 config: {config_sha256} | completed {len(records)}/{planned}")

    for policy_name in args.policies:
        shared_policy = None if policy_name == "random" else build_policy(policy_name, args)
        try:
            for level in levels:
                for episode_index, seed in enumerate(split_plan[level][:limit], start=1):
                    key = f"{policy_name}|{level}|{episode_index}|{seed}"
                    if key in records:
                        print(f"[skip] {key}")
                        continue
                    policy = build_policy(policy_name, args, seed=seed) if policy_name == "random" else shared_policy
                    episode = run_episode(
                        policy,
                        task_level=level,
                        seed=seed,
                        max_steps=args.max_steps,
                        verbose=args.verbose,
                        hydra_policy=hydra_policy,
                    )
                    if args.trace_level == "compact":
                        episode = compact_trace(episode)
                    record = {
                        "record_schema_version": V6_SCHEMA_VERSION,
                        "config_sha256": config_sha256,
                        "episode_key": key,
                        "policy_stratum": policy_name,
                        "intervention_level": INTERVENTION_LEVEL[policy_name],
                        "model_label": args.model_label if policy_name.startswith("model_") else None,
                        "level": level,
                        "episode_index": episode_index,
                        "seed": seed,
                        "completed_at_utc": utc_now(),
                        "episode": episode,
                    }
                    record["episode_sha256"] = canonical_sha256(episode)
                    append_jsonl(episodes_path, record)
                    records[key] = record
                    print(
                        f"[{len(records):>4}/{planned}] {key} "
                        f"grade={episode['grade']['score']:.3f} "
                        f"interventions={episode['provenance_summary']['interventions']}"
                    )
        finally:
            if shared_policy is not None and hasattr(shared_policy, "close"):
                shared_policy.close()

    summary = {
        "schema_version": V6_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "seed_plan_sha256": seed_plan["seed_plan_sha256"],
        "created_at_utc": utc_now(),
        "complete": len(records) == planned,
        "completed_episodes": len(records),
        "planned_episodes": planned,
        "policies": summarize(records, args.policies, levels),
    }
    atomic_write_json(summary_path, summary)
    manifest["status"] = "complete" if summary["complete"] else "partial"
    manifest["completed_at_utc"] = utc_now()
    manifest["completed_episodes"] = len(records)
    atomic_write_json(manifest_path, manifest)
    print(f"Wrote {episodes_path}, {summary_path}, and {manifest_path}")


if __name__ == "__main__":
    main()
