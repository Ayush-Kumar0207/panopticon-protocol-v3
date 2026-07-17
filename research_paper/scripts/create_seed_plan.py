#!/usr/bin/env python3
"""Create and cryptographically freeze the Panopticon V6 evaluation seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panopticon_bench.seed_plan import build_seed_plan, write_seed_plan


def read_training_seed_ledger(path: str) -> list[int]:
    if not path:
        return []
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("seeds", [])
        if not isinstance(payload, list):
            raise ValueError("training-seed JSON must be a list or an object with a 'seeds' list")
        return [int(seed) for seed in payload]
    return [int(token) for token in text.replace(",", " ").split()]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="research_paper/data/seed_plans/v6_seed_plan.json",
        help="Destination JSON file",
    )
    parser.add_argument("--master-seed", type=int, default=20260715)
    parser.add_argument("--pilot-per-level", type=int, default=5)
    parser.add_argument("--development-per-level", type=int, default=25)
    parser.add_argument("--final-per-level", type=int, default=200)
    parser.add_argument(
        "--training-seed-ledger",
        default="",
        help="Optional JSON/text seed ledger that must remain disjoint from evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    training_seeds = read_training_seed_ledger(args.training_seed_ledger)
    plan = build_seed_plan(
        master_seed=args.master_seed,
        split_sizes={
            "pilot": args.pilot_per_level,
            "development": args.development_per_level,
            "final": args.final_per_level,
        },
        excluded_training_seeds=training_seeds,
    )
    destination = write_seed_plan(args.output, plan, overwrite=args.overwrite)
    print(f"Wrote frozen seed plan: {destination}")
    print(f"SHA-256: {plan['seed_plan_sha256']}")
    print(f"Training separation: {plan['training_separation_status']}")


if __name__ == "__main__":
    main()
