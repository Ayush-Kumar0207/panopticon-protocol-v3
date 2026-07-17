"""Frozen, disjoint seed plans for preregistered Panopticon evaluations."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SEED_PLAN_SCHEMA_VERSION = "panopticon-seed-plan-v1"
DEFAULT_LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
MAX_SEED = 2**31 - 1


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_without_digest(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "seed_plan_sha256"}


def build_seed_plan(
    *,
    master_seed: int,
    split_sizes: dict[str, int],
    levels: list[str] | None = None,
    excluded_training_seeds: Iterable[int] = (),
) -> dict[str, Any]:
    levels = list(levels or DEFAULT_LEVELS)
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("levels must be a non-empty unique list")
    if not split_sizes or any(size < 1 for size in split_sizes.values()):
        raise ValueError("every split must request at least one seed per level")

    excluded = {int(seed) for seed in excluded_training_seeds}
    rng = random.Random(master_seed)
    used = set(excluded)
    splits: dict[str, dict[str, list[int]]] = {}
    for split_name, per_level in split_sizes.items():
        level_plan: dict[str, list[int]] = {}
        for level in levels:
            seeds: list[int] = []
            while len(seeds) < per_level:
                candidate = rng.randint(0, MAX_SEED)
                if candidate not in used:
                    used.add(candidate)
                    seeds.append(candidate)
            level_plan[level] = seeds
        splits[split_name] = level_plan

    training_list = sorted(excluded)
    plan: dict[str, Any] = {
        "schema_version": SEED_PLAN_SCHEMA_VERSION,
        "name": "panopticon-v6-preregistered-seeds",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "master_seed": master_seed,
        "levels": levels,
        "split_sizes_per_level": split_sizes,
        "splits": splits,
        "excluded_training_seed_count": len(training_list),
        "excluded_training_seeds_sha256": canonical_sha256(training_list),
        "training_separation_status": (
            "verified-against-provided-ledger"
            if training_list
            else "unverified-no-training-seed-ledger-provided"
        ),
        "protocol": {
            "pilot": "May be used only to validate infrastructure and logging.",
            "development": "May be used for model selection, debugging, and threshold tuning.",
            "final": "Must remain untouched until code, checkpoints, metrics, and hypotheses are frozen.",
            "prohibition": "Never move a seed between splits or tune on final-split outcomes.",
        },
    }
    plan["seed_plan_sha256"] = canonical_sha256(_plan_without_digest(plan))
    validate_seed_plan(plan, excluded_training_seeds=training_list)
    return plan


def validate_seed_plan(
    plan: dict[str, Any], *, excluded_training_seeds: Iterable[int] = ()
) -> dict[str, Any]:
    if plan.get("schema_version") != SEED_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported seed-plan schema")
    levels = plan.get("levels")
    splits = plan.get("splits")
    sizes = plan.get("split_sizes_per_level")
    if not isinstance(levels, list) or not isinstance(splits, dict) or not isinstance(sizes, dict):
        raise ValueError("seed plan is missing levels, splits, or split sizes")

    all_seeds: list[int] = []
    for split_name, expected_size in sizes.items():
        if split_name not in splits or set(splits[split_name]) != set(levels):
            raise ValueError(f"split {split_name!r} does not contain exactly the declared levels")
        for level in levels:
            seeds = splits[split_name][level]
            if len(seeds) != expected_size:
                raise ValueError(f"{split_name}/{level} has {len(seeds)} seeds; expected {expected_size}")
            if any(not isinstance(seed, int) or not 0 <= seed <= MAX_SEED for seed in seeds):
                raise ValueError(f"{split_name}/{level} contains an invalid seed")
            all_seeds.extend(seeds)

    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("seed reuse detected across split/level cells")
    overlap = set(all_seeds).intersection(int(seed) for seed in excluded_training_seeds)
    if overlap:
        raise ValueError(f"evaluation seeds overlap the training ledger: {sorted(overlap)[:5]}")

    expected_digest = canonical_sha256(_plan_without_digest(plan))
    if plan.get("seed_plan_sha256") != expected_digest:
        raise ValueError("seed-plan digest mismatch; the frozen plan may have been edited")
    return plan


def load_seed_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_seed_plan(plan)


def write_seed_plan(path: str | Path, plan: dict[str, Any], *, overwrite: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace frozen seed plan: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
