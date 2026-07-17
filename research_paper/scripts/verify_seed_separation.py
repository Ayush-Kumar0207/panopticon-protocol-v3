#!/usr/bin/env python3
"""Verify that a frozen evaluation seed plan is disjoint from a training ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panopticon_bench.seed_plan import canonical_sha256, load_seed_plan, validate_seed_plan


def evaluation_seeds(plan: dict[str, Any]) -> list[int]:
    return [
        seed
        for split in plan["splits"].values()
        for level_seeds in split.values()
        for seed in level_seeds
    ]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-plan",
        default="research_paper/data/seed_plans/v6_seed_plan.json",
    )
    parser.add_argument(
        "--training-ledger",
        default="research_paper/data/training_seed_ledger.drive_verified.json",
    )
    parser.add_argument(
        "--output",
        default="research_paper/data/seed_plans/v6_training_separation_report.json",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()
    plan_path = (ROOT / args.seed_plan).resolve()
    ledger_path = (ROOT / args.training_ledger).resolve()
    output_path = (ROOT / args.output).resolve()

    plan = load_seed_plan(plan_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    training_seeds = [int(seed) for seed in ledger["seeds"]]
    validate_seed_plan(plan, excluded_training_seeds=training_seeds)
    held_out = evaluation_seeds(plan)
    overlap = sorted(set(held_out).intersection(training_seeds))
    if overlap:
        raise ValueError(f"evaluation/training overlap detected: {overlap[:10]}")

    source_status = str(ledger.get("verification_status", "unknown-ledger-provenance"))
    directly_verified = source_status.startswith("directly-verified")
    report = {
        "schema_version": "panopticon-seed-separation-report-v1",
        "seed_plan_path": plan_path.relative_to(ROOT).as_posix(),
        "seed_plan_sha256": plan["seed_plan_sha256"],
        "training_ledger_path": ledger_path.relative_to(ROOT).as_posix(),
        "training_ledger_seeds_sha256": ledger.get(
            "seeds_sha256", canonical_sha256(training_seeds)
        ),
        "training_ledger_verification_status": source_status,
        "evaluation_seed_count": len(held_out),
        "training_seed_count": len(training_seeds),
        "overlap_count": len(overlap),
        "overlap": overlap,
        "conclusion": (
            "directly-verified-disjoint"
            if directly_verified
            else "disjoint-from-reconstructed-ledger-pending-original-artifact-confirmation"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Evaluation seeds: {len(held_out)}")
    print(f"Training seeds: {len(training_seeds)}")
    print("Overlap: 0")
    print(f"Conclusion: {report['conclusion']}")


if __name__ == "__main__":
    main()
