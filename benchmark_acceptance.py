#!/usr/bin/env python3
"""Fail-closed release gate for a trained model evaluated against its base model."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
ADVANCED_LEVELS = {"level_4", "level_5"}


def load_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trained_summaries(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    return payload["agents"]["trained"]["summary"]


def evaluate_acceptance(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_summary = trained_summaries(base)
    candidate_summary = trained_summaries(candidate)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual: Any, required: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "required": required})

    for schema_key in ("reward_schema_version", "grader_schema_version"):
        base_schema = base.get("config", {}).get(schema_key)
        candidate_schema = candidate.get("config", {}).get(schema_key)
        add(
            f"matched_{schema_key}",
            bool(base_schema) and candidate_schema == base_schema,
            candidate_schema,
            base_schema,
        )

    for config_key in (
        "episodes_per_level", "seed", "max_steps", "deterministic_trained_eval",
        "evaluation_split", "trained_policy", "model_precision", "model_prompt_max_tokens",
        "model_max_new_tokens", "timeline_level", "episode_evidence_schema",
        "source_commit", "spec_sha256", "run_fingerprint", "model_revision",
    ):
        base_value = base.get("config", {}).get(config_key)
        candidate_value = candidate.get("config", {}).get(config_key)
        add(
            f"matched_{config_key}",
            candidate_value == base_value,
            candidate_value,
            base_value,
        )
    add(
        "matched_seed_plan",
        candidate.get("seed_plan") == base.get("seed_plan"),
        "candidate seed plan",
        "exact base seed plan",
    )

    base_macro_grade = sum(base_summary[level]["grade_mean"] for level in LEVELS) / len(LEVELS)
    candidate_macro_grade = sum(candidate_summary[level]["grade_mean"] for level in LEVELS) / len(LEVELS)
    add("macro_grade_improves", candidate_macro_grade > base_macro_grade, candidate_macro_grade, f"> {base_macro_grade}")

    paired_differences = []
    for level in LEVELS:
        base_episodes = base["agents"]["trained"]["episodes"][level]
        candidate_episodes = candidate["agents"]["trained"]["episodes"][level]
        if len(base_episodes) != len(candidate_episodes) or not base_episodes:
            raise ValueError(f"Cannot form complete trained-policy pairs for {level}")
        for base_episode, candidate_episode in zip(base_episodes, candidate_episodes, strict=True):
            if base_episode.get("seed") != candidate_episode.get("seed"):
                raise ValueError(f"Mismatched trained-policy episode seeds for {level}")
            paired_differences.append(float(candidate_episode["grade"]["score"]) - float(base_episode["grade"]["score"]))
    rng = random.Random(0x50414E4F5F504149524544)
    bootstrap_means = sorted(
        statistics.mean(rng.choices(paired_differences, k=len(paired_differences)))
        for _ in range(5000)
    )
    paired_mean = statistics.mean(paired_differences)
    paired_ci95 = [bootstrap_means[124], bootstrap_means[4874]]
    add("paired_grade_ci95_lower_gt_zero", paired_ci95[0] > 0.0, paired_ci95[0], "> 0.0")

    for level in LEVELS:
        base_level = base_summary[level]
        candidate_level = candidate_summary[level]
        add(
            f"{level}.grade_not_worse",
            candidate_level["grade_mean"] >= base_level["grade_mean"],
            candidate_level["grade_mean"],
            f">= {base_level['grade_mean']}",
        )
        add(
            f"{level}.pass_rate_100pct",
            candidate_level["pass_rate"] == 1.0,
            candidate_level["pass_rate"],
            1.0,
        )
        if level in ADVANCED_LEVELS:
            add(
                f"{level}.security_not_worse",
                candidate_level["security_mean"] >= base_level["security_mean"],
                candidate_level["security_mean"],
                f">= {base_level['security_mean']}",
            )
            add(
                f"{level}.caught_not_worse",
                candidate_level["sleepers_caught_mean"] >= base_level["sleepers_caught_mean"],
                candidate_level["sleepers_caught_mean"],
                f">= {base_level['sleepers_caught_mean']}",
            )
            add(
                f"{level}.zero_missed",
                candidate_level["sleepers_missed_mean"] == 0.0,
                candidate_level["sleepers_missed_mean"],
                0.0,
            )
            false_accusations_mean = candidate_level.get("false_accusations_mean")
            add(
                f"{level}.zero_false_accusations",
                false_accusations_mean == 0.0,
                false_accusations_mean,
                0.0,
            )

    return {
        "accepted": all(check["passed"] for check in checks),
        "base_macro_grade": base_macro_grade,
        "candidate_macro_grade": candidate_macro_grade,
        "paired_grade_difference_mean": paired_mean,
        "paired_grade_difference_ci95": paired_ci95,
        "paired_bootstrap_samples": 5000,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a trained Panopticon model against its base-model benchmark")
    parser.add_argument("--base", required=True, help="Base-model full_evaluation.py JSON")
    parser.add_argument("--candidate", required=True, help="Candidate-model full_evaluation.py JSON")
    parser.add_argument("--report", default="benchmark_acceptance_report.json")
    args = parser.parse_args()

    report = evaluate_acceptance(load_payload(args.base), load_payload(args.candidate))
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: actual={check['actual']} required={check['required']}")
    print(f"\nAccepted: {report['accepted']}")
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
