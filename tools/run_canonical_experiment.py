#!/usr/bin/env python3
"""Single safe entry point for preflight, locked training, evaluation, and verification."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture_run_provenance import write_provenance  # noqa: E402
from training_preflight import perform_preflight  # noqa: E402
from research_repro import (  # noqa: E402
    ReproducibilityError,
    append_event,
    assert_research_stage_authorized,
    atomic_write_json,
    ensure_run_lock,
    git_metadata,
    load_spec,
)


def validate_run_storage(root: Path, spec: dict, *, require_training_capacity: bool) -> None:
    try:
        root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ReproducibilityError("canonical run directory must be outside the source checkout")
    ancestor = root
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.exists() or not ancestor.is_dir():
        raise ReproducibilityError(f"run-directory parent does not exist: {ancestor}")
    if require_training_capacity:
        free_gb = shutil.disk_usage(ancestor).free / 1024**3
        required = float(spec["runtime"]["minimum_free_disk_gb_before_training"])
        if free_gb < required:
            raise ReproducibilityError(
                f"persistent storage has {free_gb:.1f} GiB free; at least {required:.1f} GiB is required before training"
            )


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=["preflight", "train", "evaluate", "verify", "all"], default="all")
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="Preflight only; canonical training remains GPU-only")
    args = parser.parse_args()
    spec, spec_path = load_spec(args.spec)
    if args.stage in {"train", "all"}:
        assert_research_stage_authorized(spec, operation="training")
    if args.stage in {"evaluate", "all"}:
        assert_research_stage_authorized(spec, operation="evaluation", evaluation_split="canonical")
        assert_research_stage_authorized(spec, operation="evaluation", evaluation_split="confirmation")
    root = Path(args.run_dir).resolve()
    if args.allow_cpu_smoke and args.stage != "preflight":
        raise ReproducibilityError("--allow-cpu-smoke is permitted only with --stage preflight")
    validate_run_storage(root, spec, require_training_capacity=args.stage in {"preflight", "train", "all"})

    if args.stage in {"preflight", "train", "all"}:
        report = perform_preflight(spec_path, allow_cpu_smoke=args.allow_cpu_smoke, run_tests=True)
        if args.stage == "preflight":
            if report["passed"]:
                print(f"Canonical preflight passed for {report['run_fingerprint']}")
            else:
                print(f"Diagnostic-only preflight completed for {report['run_fingerprint']}; training remains prohibited")
            return
        lock = ensure_run_lock(root, spec, report["source_commit"])
        atomic_write_json(root / spec["outputs"]["preflight_report"], report)
        write_provenance(str(spec_path), str(root), [sys.executable, *sys.argv])
        append_event(root / spec["outputs"]["events"], "preflight_passed", run_fingerprint=lock["run_fingerprint"], source_commit=lock["source_commit"], stage="preflight")

    source = git_metadata()
    if source["dirty"]:
        raise ReproducibilityError("canonical stage requires a clean source checkout")
    lock = ensure_run_lock(root, spec, source["commit"])
    env = dict(os.environ)
    env["TRAIN_ROOT"] = str(root)
    env.update(spec["runtime"].get("deterministic_environment", {}))
    common = ["--spec", str(spec_path), "--run-fingerprint", lock["run_fingerprint"]]

    if args.stage in {"train", "all"}:
        run([sys.executable, "train_trl_v2.py", "--curriculum", "--merge", *common], env=env)
        run([sys.executable, "tools/plot_training_diagnostics.py", str(root), "--spec", str(spec_path)], env=env)
        run([sys.executable, "tools/freeze_model_artifact.py", str(root), "--spec", str(spec_path), "--create"], env=env)

    if args.stage in {"evaluate", "all"}:
        eval_cfg = spec["evaluation"]
        outputs = spec["outputs"]
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
        base = spec["base_model"]["id"]
        candidate = str(root / outputs["merged_model_dir"])
        freeze_check = [sys.executable, "tools/freeze_model_artifact.py", str(root), "--spec", str(spec_path)]
        run(freeze_check, env=env)
        run([sys.executable, "full_evaluation.py", "--model", base, "--seed", str(eval_cfg["canonical_seed"]), "--evaluation-split", "canonical", "--output", str(root / outputs["canonical_base_evaluation"]), "--plot-dir", str(root / "plots_canonical_base"), *common_eval], env=env)
        run(freeze_check, env=env)
        run([sys.executable, "full_evaluation.py", "--model", candidate, "--seed", str(eval_cfg["canonical_seed"]), "--evaluation-split", "canonical", "--output", str(root / outputs["canonical_candidate_evaluation"]), "--plot-dir", str(root / "plots_canonical_candidate"), *common_eval], env=env)
        run(freeze_check, env=env)
        run([sys.executable, "full_evaluation.py", "--model", base, "--seed", str(eval_cfg["confirmation_seed"]), "--evaluation-split", "confirmation", "--output", str(root / outputs["confirmation_base_evaluation"]), "--plot-dir", str(root / "plots_confirmation_base"), *common_eval], env=env)
        run(freeze_check, env=env)
        run([sys.executable, "full_evaluation.py", "--model", candidate, "--seed", str(eval_cfg["confirmation_seed"]), "--evaluation-split", "confirmation", "--output", str(root / outputs["confirmation_evaluation"]), "--plot-dir", str(root / "plots_confirmation_candidate"), *common_eval], env=env)

    if args.stage in {"verify", "all"}:
        run([sys.executable, "tools/verify_training_artifacts.py", str(root), "--spec", str(spec_path)])


if __name__ == "__main__":
    try:
        main()
    except (ReproducibilityError, subprocess.CalledProcessError, OSError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
