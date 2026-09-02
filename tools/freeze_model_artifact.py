#!/usr/bin/env python3
"""Create once or verify the immutable merged-model byte manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import (  # noqa: E402
    ReproducibilityError,
    atomic_write_json,
    canonical_json,
    load_spec,
    sha256_bytes,
    sha256_file,
)


def _model_files(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    model_dir = root / spec["outputs"]["merged_model_dir"]
    if not model_dir.is_dir():
        raise ReproducibilityError("merged model directory is missing")
    files = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not files:
        raise ReproducibilityError("merged model directory is empty")
    return files


def expected_model_manifest(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    lock_path = root / spec["outputs"]["run_lock"]
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("run lock is missing or corrupt") from exc
    files = _model_files(root, spec)
    return {
        "schema_version": 1,
        "run_fingerprint": lock.get("run_fingerprint"),
        "source_commit": lock.get("source_commit"),
        "spec_sha256": lock.get("spec_sha256"),
        "files": files,
        "aggregate_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "interpretation": "These exact merged-model bytes are frozen before any canonical or confirmation evaluation.",
    }


def create_or_verify_model_manifest(
    run_dir: str | Path, spec_path: str | Path, *, create: bool,
) -> tuple[dict[str, Any], list[Path]]:
    root = Path(run_dir).resolve()
    spec, _ = load_spec(spec_path)
    expected = expected_model_manifest(root, spec)
    path = root / spec["outputs"]["model_manifest"]
    if create and not path.exists():
        atomic_write_json(path, expected)
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("frozen merged-model manifest is missing or corrupt") from exc
    if canonical_json(actual) != canonical_json(expected):
        raise ReproducibilityError("merged model bytes changed after the model was frozen")
    files = [root / item["path"] for item in actual["files"]]
    return actual, [path, *files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    parser.add_argument("--create", action="store_true", help="Create only when absent; an existing manifest is never replaced")
    args = parser.parse_args()
    try:
        manifest, _ = create_or_verify_model_manifest(args.run_dir, args.spec, create=args.create)
    except Exception as exc:
        print(f"STOP: merged-model freeze check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({
        "status": "frozen-and-verified",
        "run_fingerprint": manifest["run_fingerprint"],
        "aggregate_sha256": manifest["aggregate_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
