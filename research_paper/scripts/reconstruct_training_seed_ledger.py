#!/usr/bin/env python3
"""Reconstruct the V5 training seeds from versioned code and compact metadata.

This produces useful provenance evidence, but it deliberately does not claim that
the reconstructed values were read from the original Drive-side training artifact.
Use the verification status in the output when describing seed separation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
SCHEMA_VERSION = "panopticon-training-seed-ledger-v1"
GENERATOR_FUNCTION = "generate_expert_trajectories"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def function_source_from_text(text: str, function_name: str, source_label: str) -> str:
    tree = ast.parse(text, filename=source_label)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return segment
    raise ValueError(f"Could not locate {function_name!r} in {source_label}")


def git_file_at_commit(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def reconstruct(base_seed: int, episodes_per_level: int) -> dict[str, list[int]]:
    return {
        level: [
            rng.randint(0, 999_999)
            for rng in [random.Random(f"{base_seed}:{level}")]
            for _ in range(episodes_per_level)
        ]
        for level in LEVELS
    }


def reconstruct_without_reinitializing(
    base_seed: int, episodes_per_level: int
) -> dict[str, list[int]]:
    """Readable equivalent of the compact comprehension used by reconstruct()."""
    result: dict[str, list[int]] = {}
    for level in LEVELS:
        seed_rng = random.Random(f"{base_seed}:{level}")
        result[level] = [seed_rng.randint(0, 999_999) for _ in range(episodes_per_level)]
    return result


def build_ledger(metadata_path: Path, training_source: Path) -> dict[str, Any]:
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    benchmark = metadata["benchmark"]
    base_seed = int(benchmark["seed"])
    episodes_per_level = int(benchmark["training_episodes_per_level"])
    expected_total = int(benchmark["training_total_episodes"])
    if expected_total != episodes_per_level * len(LEVELS):
        raise ValueError("training total does not equal episodes-per-level multiplied by levels")
    for level in LEVELS:
        recorded = int(metadata["training_datasets"][level]["episodes"])
        if recorded != episodes_per_level:
            raise ValueError(f"metadata episode mismatch for {level}: {recorded}")

    current_source_text = training_source.read_text(encoding="utf-8")
    current_generator_text = function_source_from_text(
        current_source_text, GENERATOR_FUNCTION, str(training_source)
    )
    recorded_commit = str(benchmark["training_source_commit"])
    source_relative_path = training_source.relative_to(ROOT).as_posix()
    historical_source_text = git_file_at_commit(recorded_commit, source_relative_path)
    generator_text = function_source_from_text(
        historical_source_text,
        GENERATOR_FUNCTION,
        f"{recorded_commit}:{source_relative_path}",
    )
    required_fragments = [
        'random.Random(f"{seed}:{task_level}")',
        "seed_rng.randint(0, 999999)",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in generator_text]
    if missing:
        raise ValueError(f"training generator no longer matches the recorded algorithm: {missing}")

    seeds_by_level = reconstruct_without_reinitializing(base_seed, episodes_per_level)
    flat_seeds = [seed for level in LEVELS for seed in seeds_by_level[level]]
    if len(flat_seeds) != expected_total:
        raise AssertionError("internal seed reconstruction count mismatch")

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": "panopticon-security-v5-ep50-reconstructed-training-seeds",
        "verification_status": "reconstructed-not-directly-verified-against-original-training-artifact",
        "claim_boundary": (
            "The values are deterministic reconstructions from repository code and compact "
            "training metadata. Confirm them against the original Drive-side data_meta.json "
            "or equivalent seed ledger before calling separation directly verified."
        ),
        "base_seed": base_seed,
        "levels": LEVELS,
        "episodes_per_level": episodes_per_level,
        "seed_count": len(flat_seeds),
        "unique_seed_count": len(set(flat_seeds)),
        "seeds_by_level": seeds_by_level,
        "seeds": flat_seeds,
        "seeds_sha256": canonical_sha256(flat_seeds),
        "reconstruction_algorithm": {
            "python_expression": 'random.Random(f"{base_seed}:{level}").randint(0, 999999)',
            "generator_function": GENERATOR_FUNCTION,
            "ordering": "level order, then episode order",
            "level_order": LEVELS,
        },
        "evidence": {
            "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
            "metadata_sha256": sha256_bytes(metadata_bytes),
            "training_source_path": source_relative_path,
            "current_training_source_sha256": sha256_bytes(training_source.read_bytes()),
            "current_generator_function_sha256": sha256_bytes(
                current_generator_text.encode("utf-8")
            ),
            "historical_training_source_sha256": sha256_bytes(
                historical_source_text.encode("utf-8")
            ),
            "historical_generator_function_sha256": sha256_bytes(
                generator_text.encode("utf-8")
            ),
            "recorded_training_source_commit": recorded_commit,
            "recorded_run_name": metadata.get("drive_folder", {}).get("name"),
        },
    }
    return ledger


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="evaluation_comparison_latest.json")
    parser.add_argument("--training-source", default="train_trl_v2.py")
    parser.add_argument(
        "--output",
        default="research_paper/data/training_seed_ledger.reconstructed.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    metadata_path = (ROOT / args.metadata).resolve()
    training_source = (ROOT / args.training_source).resolve()
    destination = (ROOT / args.output).resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
    ledger = build_ledger(metadata_path, training_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"Seeds: {ledger['seed_count']} ({ledger['unique_seed_count']} unique)")
    print(f"Seed digest: {ledger['seeds_sha256']}")
    print(f"Status: {ledger['verification_status']}")


if __name__ == "__main__":
    main()
