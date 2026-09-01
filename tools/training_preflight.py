#!/usr/bin/env python3
"""Fail-closed, CPU-safe validation before a canonical GPU run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import (  # noqa: E402
    ReproducibilityError,
    compute_run_fingerprint,
    git_metadata,
    package_versions,
    spec_sha256,
    training_critical_dirty_paths,
    load_spec,
    runtime_metadata,
    verify_seed_separation,
)


SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
SCAN_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".toml", ".yaml", ".yml", ".txt", ".log", ".env"}


def scan_secrets(root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    excluded_parts = {".git", ".venv", "venv", "__pycache__", "trained_model", "merged_model"}
    for path in root.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES and not path.name.startswith(".env"):
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": str(path.relative_to(root)), "line": line_number, "kind": kind})
    return findings


def check_required_files(spec: dict) -> list[str]:
    return [item for item in spec.get("required_source_files", []) if not (ROOT / item).is_file()]


def check_dependencies(spec: dict) -> tuple[dict[str, str | None], list[str]]:
    expected = spec.get("dependencies", {})
    actual = package_versions(expected)
    mismatches = [f"{name}: expected {wanted}, found {actual[name] or 'not installed'}" for name, wanted in expected.items() if actual[name] != wanted]
    return actual, mismatches


def run_validation_commands(spec: dict) -> list[dict[str, object]]:
    results = []
    environment = dict(os.environ)
    environment.update(spec["runtime"].get("deterministic_environment", {}))
    for raw_command in spec.get("preflight_commands", []):
        command = [sys.executable if token == "python" else token for token in raw_command]
        process = subprocess.run(command, cwd=ROOT, env=environment, text=True, encoding="utf-8", errors="replace")
        results.append({"command": raw_command, "returncode": process.returncode})
        if process.returncode:
            if "security_regression_test.py" in raw_command:
                raise ReproducibilityError("security regression test failed — training has not started")
            raise ReproducibilityError(f"Required validation failed ({process.returncode}): {' '.join(raw_command)}")
    return results


def perform_preflight(spec_path: str | Path, *, allow_cpu_smoke: bool = False, run_tests: bool = True) -> dict:
    spec, resolved_spec = load_spec(spec_path)
    if sys.version_info[:2] != (3, 11):
        raise ReproducibilityError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported; canonical runs require exactly 3.11"
        )
    git = git_metadata(ROOT)
    critical_dirty = training_critical_dirty_paths(spec, git)
    if git["dirty"]:
        raise ReproducibilityError(
            "Canonical training requires a completely clean working tree. Changed paths: "
            + ", ".join(git["status_paths"][:20])
        )
    if critical_dirty:  # explicit diagnostic retained for direct unit testing
        raise ReproducibilityError(f"Training-critical files are dirty: {critical_dirty}")
    missing = check_required_files(spec)
    if missing:
        raise ReproducibilityError(f"Required source files are missing: {missing}")
    secrets = scan_secrets(ROOT)
    if secrets:
        raise ReproducibilityError(f"Potential secrets detected (values suppressed): {json.dumps(secrets[:20])}")
    versions, dependency_mismatches = check_dependencies(spec)
    if dependency_mismatches:
        raise ReproducibilityError("Dependency lock mismatch: " + "; ".join(dependency_mismatches))
    runtime = runtime_metadata(spec.get("dependencies", {}).keys())
    gpu = runtime["gpu"]
    if spec["runtime"].get("requires_cuda") and not gpu["cuda_available"] and not allow_cpu_smoke:
        raise ReproducibilityError("CUDA is required for the canonical run. Use --allow-cpu-smoke only for infrastructure tests.")
    minimum = float(spec["runtime"].get("minimum_vram_gb", 0))
    if gpu["cuda_available"] and float(gpu["vram_gb"] or 0) < minimum:
        raise ReproducibilityError(f"GPU has {gpu['vram_gb']} GiB VRAM; the canonical profile requires at least {minimum} GiB.")
    if gpu["cuda_available"]:
        expected_torch = (
            f"{spec['runtime']['torch_distribution_version']}+"
            f"{spec['runtime']['torch_local_build']}"
        )
        if gpu.get("torch_build") != expected_torch:
            raise ReproducibilityError(
                f"CUDA PyTorch build mismatch: expected {expected_torch}, found {gpu.get('torch_build')}"
            )
        if str(gpu.get("cuda")) != str(spec["runtime"]["torch_cuda_runtime"]):
            raise ReproducibilityError(
                f"PyTorch CUDA runtime mismatch: expected {spec['runtime']['torch_cuda_runtime']}, "
                f"found {gpu.get('cuda')}"
            )
        capability = gpu.get("compute_capability")
        required_major = int(spec["runtime"]["minimum_compute_capability_major"])
        if not isinstance(capability, list) or not capability or int(capability[0]) < required_major:
            raise ReproducibilityError(
                "canonical BF16 requires NVIDIA Ampere-or-newer compute capability; T4/Turing is prohibited"
            )
    seed_separation = verify_seed_separation(spec)
    if gpu["cuda_available"] and spec["training"]["precision"] == "bf16":
        import torch
        if not torch.cuda.is_bf16_supported() or gpu.get("bf16_supported") is not True:
            raise ReproducibilityError("canonical BF16 precision is not supported by this GPU")
    tests = run_validation_commands(spec) if run_tests else []
    if gpu["cuda_available"] and run_tests:
        probe_environment = dict(os.environ)
        probe_environment.update(spec["runtime"].get("deterministic_environment", {}))
        process = subprocess.run(
            [sys.executable, "tools/gpu_training_probe.py", "--spec", str(resolved_spec)],
            cwd=ROOT,
            env=probe_environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.returncode:
            raise ReproducibilityError(
                "GPU memory/numerics probe failed — training has not started: "
                + (process.stderr.strip() or process.stdout.strip())[-2000:]
            )
        try:
            probe_report = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise ReproducibilityError("GPU probe did not return a valid report") from exc
        tests.append({
            "command": ["python", "tools/gpu_training_probe.py", "--spec", str(resolved_spec)],
            "returncode": 0,
            "report": probe_report,
        })
    cpu_smoke_only = bool(allow_cpu_smoke and not gpu["cuda_available"])
    canonical_pass = bool(run_tests and not cpu_smoke_only)
    return {
        "passed": canonical_pass,
        "diagnostic_only": not canonical_pass,
        "spec_path": str(resolved_spec),
        "spec_sha256": spec_sha256(spec),
        "source_commit": git["commit"],
        "branch": git["branch"],
        "run_fingerprint": compute_run_fingerprint(spec, git["commit"]),
        "runtime": runtime,
        "dependencies": versions,
        "seed_separation": seed_separation,
        "validations": tests,
        "cpu_smoke_only": cpu_smoke_only,
        "deterministic_environment": spec["runtime"].get("deterministic_environment", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="Development diagnostics only; never used by canonical training")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    try:
        report = perform_preflight(args.spec, allow_cpu_smoke=args.allow_cpu_smoke, run_tests=not args.skip_tests)
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["passed"]:
        print("PREFLIGHT PASS: it is safe to create the locked run directory.")
    else:
        print("DIAGNOSTIC COMPLETE — NOT A CANONICAL PREFLIGHT PASS; training remains prohibited.")


if __name__ == "__main__":
    main()
