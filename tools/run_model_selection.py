#!/usr/bin/env python3
"""Run or fixture-test the locked Panopticon development-only selection campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import (  # noqa: E402
    ReproducibilityError,
    append_event,
    atomic_write_json,
    canonical_json,
    ensure_run_lock,
    expected_selection_candidate_spec,
    git_metadata,
    load_spec,
    sha256_file,
    spec_sha256,
    validate_selection_candidate_spec,
)
from tools.capture_run_provenance import write_provenance  # noqa: E402
from tools.freeze_model_artifact import create_or_verify_model_manifest  # noqa: E402
from tools.training_preflight import perform_preflight  # noqa: E402
from tools.validate_model_selection import validate as validate_protocol  # noqa: E402
from tools.verify_training_artifacts import validate_evaluation  # noqa: E402

PROTOCOL_PATH = ROOT / "training_specs" / "model_selection_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"corrupt or missing campaign JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReproducibilityError(f"expected a JSON object: {path}")
    return value


def protocol_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    baseline_path = ROOT / protocol["baseline_spec"]
    baseline, _ = load_spec(baseline_path)
    validate_protocol(protocol, baseline)
    return protocol, baseline


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if canonical_json(read_json(path)) != canonical_json(payload):
            raise ReproducibilityError(f"existing campaign file changed identity: {path.name}")
        return
    atomic_write_json(path, payload)


def external_campaign_dir(path: str | Path) -> Path:
    root = Path(path).resolve()
    try:
        root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ReproducibilityError("model-selection campaign directory must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=True)
    return root


def candidate_spec(candidate_id: str, round_id: str, optimization_seed: int) -> dict[str, Any]:
    spec = expected_selection_candidate_spec(candidate_id, round_id, optimization_seed)
    validate_selection_candidate_spec(spec, require_compute_authorized=False)
    return spec


def file_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def result_metrics(payload: dict[str, Any], spec: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    trained = payload["agents"]["trained"]
    levels = list(spec["trajectory"]["levels"])
    summaries = [trained["summary"][level] for level in levels]
    episodes = [episode for level in levels for episode in trained["episodes"][level]]
    security_test_passed = any(
        "security_regression_test.py" in row.get("command", []) and row.get("returncode") == 0
        for row in preflight.get("validations", [])
    )
    missed = sum(int(row["final_state"]["sleepers_missed"]) for row in episodes)
    false_accusations = sum(int(row["final_state"]["false_accusations"]) for row in episodes)
    truncated = sum(
        int((row.get("provenance_summary", {}).get("model_context", {}) or {}).get("token_truncated_turns", 0))
        for row in episodes
    )
    pass_rates = {level: float(trained["summary"][level]["pass_rate"]) for level in levels}
    eligible = (
        security_test_passed
        and payload.get("status") == "complete"
        and truncated == 0
        and missed == 0
        and false_accusations == 0
        and all(value == 1.0 for value in pass_rates.values())
    )
    return {
        "eligible": eligible,
        "security_regression_passed": security_test_passed,
        "development_complete": payload.get("status") == "complete",
        "token_truncated_model_turns": truncated,
        "sleepers_missed": missed,
        "false_accusations": false_accusations,
        "pass_rate_by_level": pass_rates,
        "minimum_level_mean_grade": min(float(row["grade_mean"]) for row in summaries),
        "macro_mean_grade": statistics.mean(float(row["grade_mean"]) for row in summaries),
        "grade_scores_by_level": {
            level: [float(row["grade"]["score"]) for row in trained["episodes"][level]]
            for level in levels
        },
        "registered_training_budget": (
            int(spec["trajectory"]["episodes_per_level"])
            * int(spec["training"]["epochs"])
            * len(levels)
        ),
    }


def synthetic_record(
    candidate: dict[str, Any], round_: dict[str, Any], optimization_seed: int,
) -> dict[str, Any]:
    index = int(candidate["id"][1:])
    unsafe = candidate["id"] == "c01"
    seed_offset = {4200: 0.30, 5200: -0.20, 6200: 0.10}[optimization_seed] / 100
    grade_scores = {
        level: [
            0.60 + index / 100 + seed_offset + level_index / 100
            + ((episode_index % 3) - 1) / 1000
            for episode_index in range(int(round_["development_episodes_per_level"]))
        ]
        for level_index, level in enumerate(("easy", "medium", "hard", "level_4", "level_5"))
    }
    metrics = {
        "eligible": not unsafe,
        "security_regression_passed": not unsafe,
        "development_complete": True,
        "token_truncated_model_turns": 0,
        "sleepers_missed": int(unsafe),
        "false_accusations": 0,
        "pass_rate_by_level": {level: 1.0 for level in ("easy", "medium", "hard", "level_4", "level_5")},
        "minimum_level_mean_grade": min(statistics.mean(rows) for rows in grade_scores.values()),
        "macro_mean_grade": statistics.mean(statistics.mean(rows) for rows in grade_scores.values()),
        "grade_scores_by_level": grade_scores,
        "registered_training_budget": (
            int(round_["expert_episodes_per_level"])
            * int(candidate["epochs"])
            * 5
        ),
    }
    return {
        "candidate_id": candidate["id"],
        "optimization_seed": optimization_seed,
        "metrics": metrics,
        "evidence": {
            "kind": "synthetic-fixture-not-scientific-evidence",
            "identity": hashlib.sha256(canonical_json({
                "candidate": candidate["id"], "round": round_["id"], "seed": optimization_seed,
            }).encode()).hexdigest(),
        },
    }


def run_command(command: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def real_record(
    campaign_root: Path,
    candidate: dict[str, Any],
    round_: dict[str, Any],
    optimization_seed: int,
) -> dict[str, Any]:
    name = f"{round_['id']}-{candidate['id']}-seed{optimization_seed}"
    spec = candidate_spec(candidate["id"], round_["id"], optimization_seed)
    spec_path = campaign_root / "generated_specs" / f"{name}.json"
    spec_path.parent.mkdir(exist_ok=True)
    write_once(spec_path, spec)
    validate_selection_candidate_spec(spec, require_compute_authorized=True)
    run_root = campaign_root / "runs" / name
    run_root.parent.mkdir(exist_ok=True)
    preflight = perform_preflight(spec_path, run_tests=True)
    if preflight.get("passed") is not True or preflight.get("selection_candidate_authorized") is not True:
        raise ReproducibilityError("selection-candidate preflight did not authorize development training")
    lock = ensure_run_lock(run_root, spec, preflight["source_commit"])
    atomic_write_json(run_root / spec["outputs"]["preflight_report"], preflight)
    write_provenance(str(spec_path), str(run_root), [sys.executable, *sys.argv])
    append_event(
        run_root / spec["outputs"]["events"], "preflight_passed",
        run_fingerprint=lock["run_fingerprint"], source_commit=lock["source_commit"],
        stage="development-selection", candidate_id=candidate["id"],
        round_id=round_["id"], optimization_seed=optimization_seed,
    )
    env = dict(os.environ)
    env["TRAIN_ROOT"] = str(run_root)
    env.update(spec["runtime"].get("deterministic_environment", {}))
    common = ["--spec", str(spec_path), "--run-fingerprint", lock["run_fingerprint"]]
    run_command([sys.executable, "train_trl_v2.py", "--curriculum", "--merge", *common], env=env)
    run_command([sys.executable, "tools/plot_training_diagnostics.py", str(run_root), "--spec", str(spec_path)], env=env)
    run_command([sys.executable, "tools/freeze_model_artifact.py", str(run_root), "--spec", str(spec_path), "--create"], env=env)
    eval_cfg = spec["evaluation"]
    output = run_root / "evaluation_development_candidate.json"
    common_eval = [
        "--episodes", str(eval_cfg["episodes_per_level"]),
        "--max-steps", str(eval_cfg["max_steps"]),
        "--trained-policy", eval_cfg["trained_policy"],
        "--model-revision", spec["base_model"]["revision"],
        "--model-precision", eval_cfg["model_precision"],
        "--model-prompt-max-tokens", str(eval_cfg["model_prompt_max_tokens"]),
        "--model-max-new-tokens", str(eval_cfg["model_max_new_tokens"]),
        "--timeline-level", eval_cfg["timeline_level"],
        *common,
    ]
    run_command([
        sys.executable, "full_evaluation.py",
        "--model", str(run_root / spec["outputs"]["merged_model_dir"]),
        "--seed", str(eval_cfg["development_seed"]),
        "--evaluation-split", "development",
        "--output", str(output),
        "--plot-dir", str(run_root / "plots_development_candidate"),
        *common_eval,
    ], env=env)
    create_or_verify_model_manifest(run_root, spec_path, create=False)
    payload = validate_evaluation(
        output, spec, lock["run_fingerprint"], "development",
        source_commit=lock["source_commit"], model_kind="candidate",
    )
    payload.pop("_verified_files", None)
    return {
        "candidate_id": candidate["id"],
        "optimization_seed": optimization_seed,
        "metrics": result_metrics(payload, spec, preflight),
        "evidence": {
            "run_fingerprint": lock["run_fingerprint"],
            "spec_sha256": spec_sha256(spec),
            "development_evaluation": file_identity(output, relative_to=run_root),
            "model_manifest": file_identity(
                run_root / spec["outputs"]["model_manifest"], relative_to=run_root,
            ),
            "preflight": file_identity(
                run_root / spec["outputs"]["preflight_report"], relative_to=run_root,
            ),
        },
    }


def aggregate_candidate(
    candidate: dict[str, Any], records: list[dict[str, Any]], round_: dict[str, Any],
    aggregation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_seeds = list(round_["optimization_seeds"])
    actual_seeds = [row["optimization_seed"] for row in records]
    if actual_seeds != expected_seeds:
        raise ReproducibilityError(
            f"candidate {candidate['id']} has incomplete/out-of-order seed evidence "
            f"({actual_seeds!r} != {expected_seeds!r})"
        )
    metrics = [row["metrics"] for row in records]
    uncertainty = (aggregation or protocol_payload()[0]["multi_seed_aggregation"])["uncertainty_aware_ranking"]
    cells = [
        list(metric["grade_scores_by_level"][level])
        for metric in metrics
        for level in ("easy", "medium", "hard", "level_4", "level_5")
    ]
    if any(len(cell) != int(round_["development_episodes_per_level"]) for cell in cells):
        raise ReproducibilityError(f"candidate {candidate['id']} has incomplete bootstrap cells")
    rng = random.Random(int(uncertainty["seed"]))
    bootstrap = sorted(
        statistics.mean(statistics.mean(rng.choices(cell, k=len(cell))) for cell in cells)
        for _ in range(int(uncertainty["samples"]))
    )
    bootstrap_low = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    return {
        "candidate_id": candidate["id"],
        "eligible": all(row["eligible"] for row in metrics),
        "minimum_level_mean_grade": min(row["minimum_level_mean_grade"] for row in metrics),
        "macro_mean_grade": statistics.mean(row["macro_mean_grade"] for row in metrics),
        "macro_mean_grade_bootstrap_ci95_low": bootstrap_low,
        "registered_training_budget": sum(row["registered_training_budget"] for row in metrics),
        "seed_eligibility": [
            {"optimization_seed": record["optimization_seed"], "eligible": record["metrics"]["eligible"]}
            for record in records
        ],
        "evidence": [row["evidence"] for row in records],
    }


def rank_candidates(rows: list[dict[str, Any]], advance: int) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = sorted(rows, key=lambda row: (
        not row["eligible"],
        -row["minimum_level_mean_grade"],
        -row["macro_mean_grade_bootstrap_ci95_low"],
        -row["macro_mean_grade"],
        row["registered_training_budget"],
        row["candidate_id"],
    ))
    eligible = [row for row in ordered if row["eligible"]]
    if len(eligible) < advance:
        raise ReproducibilityError(
            f"only {len(eligible)} fully eligible candidates remain; {advance} are required. "
            "STOP rather than advancing a failed seed or security gate"
        )
    return ordered, [row["candidate_id"] for row in eligible[:advance]]


def proposed_final_spec(
    baseline: dict[str, Any], protocol: dict[str, Any], winner: dict[str, Any], campaign_fingerprint: str,
) -> dict[str, Any]:
    proposal = copy.deepcopy(baseline)
    refit = protocol["finalization"]["final_refit"]
    proposal["experiment_id"] = f"security-first-v6-{winner['id']}-proposed"
    proposal["status"] = protocol["finalization"]["proposal_status"]
    proposal["trajectory"]["episodes_per_level"] = refit["expert_episodes_per_level"]
    proposal["trajectory"]["training_seed"] = refit["training_data_seed"]
    proposal["training"].update({
        "learning_rate": winner["learning_rate"],
        "lora_r": winner["lora_r"],
        "lora_alpha": winner["lora_alpha"],
        "lora_dropout": winner["lora_dropout"],
        "epochs": winner["epochs"],
        "optimization_seed_by_level": {
            level: int(refit["optimization_seed"]) + index
            for index, level in enumerate(proposal["trajectory"]["levels"], start=1)
        },
    })
    proposal["evaluation"]["development_seed"] = protocol["fixed"]["development_seed"]
    proposal["evaluation"]["episodes_per_level"] = refit["development_episodes_per_level"]
    for key in ("required_source_files", "training_critical_files"):
        proposal[key] = [
            protocol["finalization"]["required_new_spec"]
            if item == protocol["baseline_spec"] else item
            for item in proposal.get(key, [])
        ]
    proposal["selection_evidence"] = {
        "campaign_fingerprint": campaign_fingerprint,
        "decision_file": protocol["orchestration"]["decision"],
        "winner": winner["id"],
        "final_refit_optimization_seed": refit["optimization_seed"],
        "heldout_namespaces_used": False,
        "review_and_commit_required_before_frozen_status": True,
    }
    return proposal


def orchestrate(campaign_dir: str | Path, *, synthetic_fixture: bool = False) -> dict[str, Any]:
    protocol, baseline = protocol_payload()
    required_status = protocol["orchestration"]["real_execution_requires_status"]
    if not synthetic_fixture and protocol.get("status") != required_status:
        raise ReproducibilityError(
            "model-selection compute is not authorized by the committed protocol; no training started"
        )
    git = {"commit": "synthetic-fixture", "dirty": False} if synthetic_fixture else git_metadata()
    if git["dirty"]:
        raise ReproducibilityError("source commit does not match clean campaign identity; no training started")
    root = external_campaign_dir(campaign_dir)
    lock = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "mode": "synthetic-fixture" if synthetic_fixture else "real-development-only",
        "source_commit": git["commit"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "baseline_spec_sha256": spec_sha256(baseline),
    }
    lock["campaign_fingerprint"] = hashlib.sha256(canonical_json(lock).encode()).hexdigest()
    lock_path = root / protocol["orchestration"]["campaign_lock"]
    if not lock_path.exists() and any(root.iterdir()):
        raise ReproducibilityError("new campaign requires an empty directory or its exact campaign lock")
    write_once(lock_path, lock)
    state_path = root / protocol["orchestration"]["resume_state"]
    prior = read_json(state_path) if state_path.exists() else {}
    if prior and prior.get("campaign_fingerprint") != lock["campaign_fingerprint"]:
        raise ReproducibilityError("campaign state belongs to another source/protocol identity")
    prior_records = prior.get("records", [])
    prior_decisions = prior.get("rounds", prior.get("decisions", []))
    if not isinstance(prior_records, list) or not isinstance(prior_decisions, list):
        raise ReproducibilityError("campaign resume state has an invalid structure")
    candidates = {row["id"]: row for row in protocol["candidates"]}
    survivors = list(candidates)
    all_records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for round_ in protocol["rounds"]:
        if len(survivors) != int(round_["input_candidates"]):
            raise ReproducibilityError("campaign survivor count drifted from the registered 8→3→2→1 topology")
        aggregates = []
        for candidate_id in survivors:
            records = []
            for optimization_seed in round_["optimization_seeds"]:
                record = (
                    synthetic_record(candidates[candidate_id], round_, optimization_seed)
                    if synthetic_fixture else
                    real_record(root, candidates[candidate_id], round_, optimization_seed)
                )
                records.append(record)
                durable = {"round_id": round_["id"], **record}
                if len(all_records) < len(prior_records) and canonical_json(prior_records[len(all_records)]) != canonical_json(durable):
                    raise ReproducibilityError("existing campaign result does not match revalidated evidence")
                all_records.append(durable)
                atomic_write_json(state_path, {
                    **lock, "status": "running",
                    "records": [*all_records, *prior_records[len(all_records):]],
                    "decisions": [*decisions, *prior_decisions[len(decisions):]],
                })
            aggregates.append(aggregate_candidate(
                candidates[candidate_id], records, round_, protocol["multi_seed_aggregation"],
            ))
        ranking, survivors = rank_candidates(aggregates, int(round_["advance"]))
        decision = {
            "round_id": round_["id"], "ranking": ranking,
            "advanced": survivors, "manual_override": False,
        }
        if len(decisions) < len(prior_decisions) and canonical_json(prior_decisions[len(decisions)]) != canonical_json(decision):
            raise ReproducibilityError("existing survivor decision does not match deterministic reranking")
        decisions.append(decision)
        write_once(root / f"{round_['id']}_decision.json", {**lock, **decision})
        atomic_write_json(state_path, {
            **lock, "status": "running", "records": all_records,
            "decisions": [*decisions, *prior_decisions[len(decisions):]],
        })
    winner_id = survivors[0]
    decision = {
        **lock,
        "status": "synthetic-fixture-complete" if synthetic_fixture else "development-selection-complete",
        "winner": winner_id,
        "heldout_namespaces_used": False,
        "manual_override": False,
        "records": all_records,
        "rounds": decisions,
        "final_refit_optimization_seed": protocol["finalization"]["final_refit"]["optimization_seed"],
    }
    write_once(root / protocol["orchestration"]["decision"], decision)
    proposal = proposed_final_spec(baseline, protocol, candidates[winner_id], lock["campaign_fingerprint"])
    if synthetic_fixture:
        proposal["status"] = "synthetic-fixture-not-evidence"
    write_once(root / protocol["orchestration"]["proposed_spec"], proposal)
    atomic_write_json(state_path, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument(
        "--synthetic-fixture", action="store_true",
        help="Exercise identities, aggregation, ranking, resume, and proposal only; never scientific evidence",
    )
    args = parser.parse_args()
    try:
        report = orchestrate(args.campaign_dir, synthetic_fixture=args.synthetic_fixture)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ReproducibilityError, subprocess.CalledProcessError) as exc:
        print(f"STOP: model-selection campaign failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({
        "status": report["status"], "winner": report["winner"],
        "campaign_fingerprint": report["campaign_fingerprint"],
        "heldout_namespaces_used": report["heldout_namespaces_used"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
