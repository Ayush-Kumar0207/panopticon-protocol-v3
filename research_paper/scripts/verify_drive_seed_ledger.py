#!/usr/bin/env python3
"""Verify Drive-recorded V5 seeds against reconstruction and emit a direct ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panopticon_bench.seed_plan import canonical_sha256


LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-evidence",
        default="research_paper/data/raw/v5_drive_seed_evidence.json",
    )
    parser.add_argument(
        "--reconstructed-ledger",
        default="research_paper/data/training_seed_ledger.reconstructed.json",
    )
    parser.add_argument(
        "--output",
        default="research_paper/data/training_seed_ledger.drive_verified.json",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()
    evidence_path = (ROOT / args.drive_evidence).resolve()
    reconstructed_path = (ROOT / args.reconstructed_ledger).resolve()
    output_path = (ROOT / args.output).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    reconstructed = json.loads(reconstructed_path.read_text(encoding="utf-8"))

    if list(evidence["levels"]) != LEVELS:
        raise ValueError("Drive evidence levels are missing or out of canonical order")
    if evidence["recorded_base_seed"] != reconstructed["base_seed"]:
        raise ValueError("Drive base seed differs from reconstruction metadata")
    if evidence["recorded_source_commit"] != reconstructed["evidence"]["recorded_training_source_commit"]:
        raise ValueError("Drive source commit differs from reconstruction metadata")

    direct_by_level: dict[str, list[int]] = {}
    sources: list[dict[str, object]] = []
    for level in LEVELS:
        record = evidence["levels"][level]
        expected_episodes = list(range(1, evidence["recorded_episodes_per_level"] + 1))
        if record["episodes"] != expected_episodes:
            raise ValueError(f"Drive episode order is incomplete for {level}")
        seeds = [int(seed) for seed in record["seeds"]]
        if len(seeds) != record["record_count"]:
            raise ValueError(f"Drive seed count does not match record count for {level}")
        if seeds != reconstructed["seeds_by_level"][level]:
            differing = [
                index + 1
                for index, (actual, expected) in enumerate(
                    zip(seeds, reconstructed["seeds_by_level"][level])
                )
                if actual != expected
            ]
            raise ValueError(f"Drive/reconstruction mismatch for {level}: episodes {differing[:10]}")
        direct_by_level[level] = seeds
        sources.append(
            {
                "level": level,
                "drive_file_id": record["drive_file_id"],
                "title": record["title"],
                "size_bytes": record["size_bytes"],
                "modified_time": record["modified_time"],
                "seed_count": len(seeds),
                "ordered_seed_sha256": canonical_sha256(seeds),
            }
        )

    flat = [seed for level in LEVELS for seed in direct_by_level[level]]
    if flat != reconstructed["seeds"]:
        raise AssertionError("flattened Drive ledger differs from reconstructed ledger")
    ledger = {
        "schema_version": "panopticon-training-seed-ledger-v1",
        "name": "panopticon-security-v5-ep50-drive-verified-training-seeds",
        "verification_status": "directly-verified-against-drive-expert-metrics",
        "claim_boundary": (
            "All 250 ordered episode seeds were read from the five original Drive-side "
            "expert_metrics JSON artifacts and matched the exact reconstruction from the "
            "recorded source commit. This verifies seed identity, not model safety."
        ),
        "base_seed": reconstructed["base_seed"],
        "levels": LEVELS,
        "episodes_per_level": evidence["recorded_episodes_per_level"],
        "seed_count": len(flat),
        "unique_seed_count": len(set(flat)),
        "seeds_by_level": direct_by_level,
        "seeds": flat,
        "seeds_sha256": canonical_sha256(flat),
        "direct_evidence": {
            "source_folder": evidence["source_folder"],
            "retrieval_mode": evidence["retrieval_mode"],
            "evidence_snapshot_path": evidence_path.relative_to(ROOT).as_posix(),
            "evidence_snapshot_sha256": canonical_sha256(evidence),
            "source_files": sources,
        },
        "cross_check": {
            "reconstructed_ledger_path": reconstructed_path.relative_to(ROOT).as_posix(),
            "reconstructed_seed_sha256": reconstructed["seeds_sha256"],
            "exact_ordered_match": True,
            "recorded_source_commit": evidence["recorded_source_commit"],
        },
    }
    output_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Direct Drive seeds: {len(flat)} ({len(set(flat))} unique)")
    print(f"Seed digest: {ledger['seeds_sha256']}")
    print("Drive/reconstruction exact ordered match: yes")


if __name__ == "__main__":
    main()
