"""Research-run identity, provenance, event, and validation primitives.

This module intentionally depends only on the Python standard library so that
preflight can explain a broken ML environment instead of failing to import it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
SECRET_KEY_RE = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential|private[_-]?key)")
SAFE_ENV_KEYS = {
    "CI", "COLAB_RELEASE_TAG", "COLAB_GPU", "CUDA_VISIBLE_DEVICES",
    "LANG", "PYTHONHASHSEED", "TRAIN_ROOT",
}


class ReproducibilityError(RuntimeError):
    """A fail-closed reproducibility or integrity error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)


def load_spec(path: str | Path) -> tuple[dict[str, Any], Path]:
    spec_path = Path(path).resolve()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"Cannot read training specification {spec_path}: {exc}") from exc
    required = {"schema_version", "experiment_id", "base_model", "trajectory", "training", "evaluation"}
    missing = sorted(required - spec.keys())
    if missing:
        raise ReproducibilityError(f"Training specification is missing keys: {missing}")
    if spec["schema_version"] != 1:
        raise ReproducibilityError(f"Unsupported training specification schema: {spec['schema_version']}")
    return spec, spec_path


def spec_sha256(spec: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(spec).encode("utf-8"))


def _git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and process.returncode:
        raise ReproducibilityError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def git_metadata(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=repo_root)
    return {
        "commit": _git("rev-parse", "HEAD", cwd=repo_root),
        "branch": _git("branch", "--show-current", cwd=repo_root, check=False) or "DETACHED",
        "dirty": bool(status),
        "status_paths": [line[3:] for line in status.splitlines() if len(line) >= 4],
    }


def training_critical_dirty_paths(spec: dict[str, Any], git: dict[str, Any]) -> list[str]:
    critical = {Path(item).as_posix() for item in spec.get("training_critical_files", [])}
    return sorted(
        path for path in git.get("status_paths", [])
        if Path(path).as_posix() in critical or Path(path).as_posix().startswith("training_specs/")
    )


def compute_run_fingerprint(spec: dict[str, Any], source_commit: str) -> str:
    critical = {
        "fingerprint_schema": 1,
        "source_commit": source_commit,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec_sha256(spec),
        "base_model": spec["base_model"],
        "trajectory": spec["trajectory"],
        "training": spec["training"],
    }
    return sha256_bytes(canonical_json(critical).encode("utf-8"))


SEED_NAMESPACES = {
    "training": (0, 499_999_999),
    "development": (500_000_000, 999_999_999),
    "canonical": (1_000_000_000, 1_499_999_999),
    "confirmation": (1_500_000_000, 1_999_999_999),
}


def _partitioned_seed_plan(
    *, namespace: str, root_seed: int, levels: list[str], episodes_per_level: int,
) -> dict[str, list[int]]:
    """Return unique deterministic seeds from a split-specific numeric namespace."""
    if namespace not in SEED_NAMESPACES:
        raise ReproducibilityError(f"Unknown seed namespace: {namespace}")
    if episodes_per_level <= 0:
        raise ReproducibilityError("Episode count must be positive")
    start, end = SEED_NAMESPACES[namespace]
    total = len(levels) * episodes_per_level
    population_size = end - start + 1
    if total > population_size:
        raise ReproducibilityError(f"Seed namespace {namespace} cannot hold {total} unique episodes")
    rng = random.Random(f"panopticon:{namespace}:v1:{int(root_seed)}")
    flat = rng.sample(range(start, end + 1), total)
    return {
        level: flat[index * episodes_per_level:(index + 1) * episodes_per_level]
        for index, level in enumerate(levels)
    }


def training_seed_plan(spec: dict[str, Any]) -> dict[str, list[int]]:
    """Derive the exact expert-data seeds used by the frozen generator."""
    return _partitioned_seed_plan(
        namespace="training",
        root_seed=int(spec["trajectory"]["training_seed"]),
        levels=list(spec["trajectory"]["levels"]),
        episodes_per_level=int(spec["trajectory"]["episodes_per_level"]),
    )


def evaluation_seed_plan(spec: dict[str, Any], split: str) -> dict[str, list[int]]:
    """Mirror full_evaluation.build_seed_plan for a named frozen split."""
    if split not in {"development", "canonical", "confirmation"}:
        raise ReproducibilityError(f"Unknown evaluation split: {split}")
    return _partitioned_seed_plan(
        namespace=split,
        root_seed=int(spec["evaluation"][f"{split}_seed"]),
        levels=list(spec["trajectory"]["levels"]),
        episodes_per_level=int(spec["evaluation"]["episodes_per_level"]),
    )


def verify_seed_separation(spec: dict[str, Any]) -> dict[str, Any]:
    plans = {"training": training_seed_plan(spec)}
    plans.update({split: evaluation_seed_plan(spec, split) for split in ("development", "canonical", "confirmation")})
    flattened = {name: {seed for values in plan.values() for seed in values} for name, plan in plans.items()}
    overlaps = {}
    names = list(flattened)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(flattened[left] & flattened[right])
            if overlap:
                overlaps[f"{left}:{right}"] = overlap
    if overlaps:
        raise ReproducibilityError(f"Training/evaluation seed plans overlap: {canonical_json(overlaps)}")
    return {
        "status": "verified-disjoint",
        "counts": {name: len(values) for name, values in flattened.items()},
        "digests": {name: sha256_bytes(canonical_json(plan).encode("utf-8")) for name, plan in plans.items()},
    }


def build_run_lock(spec: dict[str, Any], source_commit: str) -> dict[str, Any]:
    fingerprint = compute_run_fingerprint(spec, source_commit)
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "source_commit": source_commit,
        "spec_sha256": spec_sha256(spec),
        "run_fingerprint": fingerprint,
        "base_model": spec["base_model"],
        "trajectory": spec["trajectory"],
        "training": spec["training"],
    }


def ensure_run_lock(run_dir: str | Path, spec: dict[str, Any], source_commit: str) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = build_run_lock(spec, source_commit)
    lock_path = root / spec.get("outputs", {}).get("run_lock", "run_config.json")
    if lock_path.exists():
        try:
            actual = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError(f"Existing run lock is unreadable: {lock_path}: {exc}") from exc
        if actual.get("source_commit") != source_commit:
            raise ReproducibilityError("source commit does not match run identity")
        if canonical_json(actual) != canonical_json(expected):
            raise ReproducibilityError(
                "Run directory belongs to a different experiment. Use the original source/spec/config "
                "or choose a fresh run directory; existing data was not modified."
            )
    else:
        if any(root.iterdir()):
            raise ReproducibilityError(
                f"Refusing to place a new run identity in non-empty directory without a lock: {root}"
            )
        atomic_write_json(lock_path, expected)
    return expected


def assert_metadata_compatible(actual: dict[str, Any], expected: dict[str, Any], fields: Iterable[str]) -> None:
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in fields if actual.get(field) != expected.get(field)
    }
    if mismatches:
        raise ReproducibilityError(f"Incompatible resume metadata: {canonical_json(mismatches)}")


def scrub_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SECRET_KEY_RE.search(str(key)) else scrub_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_secrets(item) for item in value]
    return value


def safe_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS and not SECRET_KEY_RE.search(key)}


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            version = metadata.version(name)
            # CUDA wheels carry a local tag (for example 2.2.1+cu121).  The
            # canonical CUDA build is checked separately; dependency identity
            # uses the upstream distribution version shared with CPU CI.
            versions[name] = version.split("+", 1)[0] if name == "torch" else version
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def all_package_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(installed.items()))


def runtime_metadata(dependencies: Iterable[str]) -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_available": False,
        "name": None,
        "vram_gb": None,
        "cuda": None,
        "compute_capability": None,
        "bf16_supported": False,
        "torch_build": None,
    }
    try:
        import torch
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda"] = torch.version.cuda
        gpu["torch_build"] = str(torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                name=props.name,
                vram_gb=round(props.total_memory / 1024**3, 3),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                bf16_supported=bool(torch.cuda.is_bf16_supported()),
            )
    except Exception as exc:  # preflight reports this without leaking environment state
        gpu["torch_probe_error"] = type(exc).__name__
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "hostname": socket.gethostname(),
        "packages": package_versions(dependencies),
        "all_packages": all_package_versions(),
        "gpu": gpu,
    }


def capture_provenance(
    spec: dict[str, Any], spec_path: Path, run_dir: Path, command: list[str] | None = None,
) -> dict[str, Any]:
    git = git_metadata()
    lock = build_run_lock(spec, git["commit"])
    return scrub_secrets({
        "schema_version": 1,
        "captured_at": utc_now(),
        "git": git,
        "spec_path": str(spec_path),
        "spec_sha256": lock["spec_sha256"],
        "run_fingerprint": lock["run_fingerprint"],
        "base_model": spec["base_model"],
        "seed": spec["trajectory"]["training_seed"],
        "episodes_per_level": spec["trajectory"]["episodes_per_level"],
        "curriculum": spec["trajectory"]["levels"],
        "training": spec["training"],
        "deterministic_environment": spec["runtime"].get("deterministic_environment", {}),
        "runtime": runtime_metadata(spec.get("dependencies", {}).keys()),
        "command": command or sys.argv,
        "safe_environment": safe_environment(),
        "run_directory": str(run_dir.resolve()),
    })


def append_event(
    path: str | Path, event: str, *, run_fingerprint: str, source_commit: str,
    stage: str, **metadata_values: Any,
) -> None:
    payload = scrub_secrets({
        "schema_version": 1,
        "timestamp": utc_now(),
        "event": event,
        "run_fingerprint": run_fingerprint,
        "source_commit": source_commit,
        "stage": stage,
        **metadata_values,
    })
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
