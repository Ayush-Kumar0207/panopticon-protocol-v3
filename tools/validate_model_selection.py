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
    if selection.get("status") != "preregistered-design-compute-not-authorized":
        errors.append("selection status must keep compute explicitly unauthorized")
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
    final = selection.get("finalization", {})
    if final.get("required_status") != "frozen-selected-canonical":
        errors.append("final selected spec must use the heldout-authorized status")
    if not str(final.get("required_new_spec", "")).endswith("security_first_v6_selected.json"):
        errors.append("selection must produce a new versioned spec, never mutate V5 in place")
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
