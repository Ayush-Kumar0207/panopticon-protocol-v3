#!/usr/bin/env python3
"""Validate the bounded Panopticon development-only model-selection design."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "training_specs" / "model_selection_v1.json"
BASELINE = ROOT / "training_specs" / "security_first_v5.json"


def validate(selection: dict, baseline: dict) -> None:
    errors: list[str] = []
    if selection.get("status") not in {
        "preregistered-design-compute-not-authorized",
        "preregistered-development-compute-authorized",
    }:
        errors.append("selection status is not a supported version-controlled authorization state")
    if baseline.get("status") != "provisional-fixed-method-baseline":
        errors.append("V5 must remain an explicitly provisional fixed-method baseline")
    if selection.get("allowed_namespaces") != ["training", "development"]:
        errors.append("only training and development namespaces may be used")
    if selection.get("forbidden_namespaces") != ["canonical", "confirmation"]:
        errors.append("canonical and confirmation must be explicitly forbidden")

    factors = selection.get("factors", {})
    expected = {
        (lr, lora["r"], lora["alpha"], epochs, dropout)
        for lr, lora, epochs, dropout in itertools.product(
            factors.get("learning_rate", []), factors.get("lora", []),
            factors.get("epochs", []), factors.get("lora_dropout", [])
        )
    }
    candidates = selection.get("candidates", [])
    actual = {
        (item.get("learning_rate"), item.get("lora_r"), item.get("lora_alpha"),
         item.get("epochs"), item.get("lora_dropout"))
        for item in candidates
    }
    ids = [item.get("id") for item in candidates]
    if actual != expected or len(candidates) != len(expected):
        errors.append("candidate roster is not the exact registered factorial grid")
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        errors.append("candidate IDs must be unique and sorted")

    rounds = selection.get("rounds", [])
    if [(r.get("input_candidates"), r.get("advance")) for r in rounds] != [(8, 3), (3, 2), (2, 1)]:
        errors.append("successive-halving survivor counts must be 8→3→2→1")
    if any(not r.get("optimization_seeds") for r in rounds):
        errors.append("each round must freeze at least one optimization seed")
    if any(int(r.get("development_episodes_per_level", 0)) <= 0 for r in rounds):
        errors.append("each round needs a positive development episode count")
    fixed = selection.get("fixed", {})
    if (
        fixed.get("training_data_seed") != 42
        or fixed.get("development_seed") != 41
        or fixed.get("optimization_seed_by_level_rule")
        != "optimization_seed + one_based_level_index"
    ):
        errors.append("training/development/optimization seed rules are not fully frozen")

    aggregation = selection.get("multi_seed_aggregation", {})
    uncertainty = aggregation.get("uncertainty_aware_ranking", {})
    if (
        aggregation.get("seed_shopping_prohibited") is not True
        or aggregation.get("missing_seed_policy") != "ineligible-stop"
        or "every optimization-seed run" not in aggregation.get("eligibility", "")
        or "minimum" not in aggregation.get("minimum_level_mean_grade", "")
        or "arithmetic mean" not in aggregation.get("macro_mean_grade", "")
        or "identical development seed plan" not in aggregation.get("paired_development_design", "")
        or uncertainty.get("field") != "macro_mean_grade_bootstrap_ci95_low"
        or "stratified paired bootstrap" not in uncertainty.get("method", "")
        or uncertainty.get("samples") != 5000
        or uncertainty.get("seed") != 5785238022979748179
    ):
        errors.append("multi-seed aggregation and paired-development rules are incomplete")

    eligibility = selection.get("eligibility", {})
    required_gates = {
        "all_security_regression_tests_pass": True,
        "all_development_episodes_complete": True,
        "zero_token_truncated_model_turns": True,
        "zero_sleepers_missed": True,
        "zero_false_accusations": True,
        "pass_rate_each_level": 1.0,
    }
    if any(eligibility.get(key) != value for key, value in required_gates.items()):
        errors.append("security/completeness eligibility gates may not be weakened")
    ranking = selection.get("ranking", [])
    if not ranking or ranking[-1] != {"metric": "candidate_id", "direction": "ascending"}:
        errors.append("ranking must end with deterministic candidate_id tie-breaking")
    if {"metric": "registered_training_budget", "direction": "ascending"} not in ranking:
        errors.append("ranking must use the preregistered budget rather than an unrecoverable token estimate")
    if {"metric": "macro_mean_grade_bootstrap_ci95_low", "direction": "descending"} not in ranking:
        errors.append("ranking must include the preregistered uncertainty-aware grade bound")
    final = selection.get("finalization", {})
    if final.get("required_status") != "frozen-selected-canonical":
        errors.append("final selected spec must use the heldout-authorized status")
    if not str(final.get("required_new_spec", "")).endswith("security_first_v6_selected.json"):
        errors.append("selection must produce a new versioned spec, never mutate V5 in place")
    refit = final.get("final_refit", {})
    if (
        final.get("proposal_status") != "proposed-selected-review-required"
        or refit.get("optimization_seed") != 7200
        or refit.get("training_data_seed") != 42
        or refit.get("optimization_seed_by_level_rule") != "7200 + one_based_level_index"
    ):
        errors.append("the no-seed-shopping final refit rule is incomplete")
    orchestration = selection.get("orchestration", {})
    if (
        orchestration.get("entrypoint") != "tools/run_model_selection.py"
        or orchestration.get("real_execution_requires_status")
        != "preregistered-development-compute-authorized"
        or orchestration.get("selection_candidate_status") != "development-selection-candidate"
        or orchestration.get("synthetic_fixture_is_never_evidence") is not True
    ):
        errors.append("model-selection orchestration contract is incomplete")
    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    validate(selection, baseline)
    print("MODEL SELECTION DESIGN PASS: bounded development-only search; heldout splits remain sealed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"STOP: invalid model-selection protocol: {exc}", file=sys.stderr)
        raise SystemExit(1)
