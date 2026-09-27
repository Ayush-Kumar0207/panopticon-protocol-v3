#!/usr/bin/env python3
"""Create a hash-indexed, no-training Track A GPU qualification bundle."""

from __future__ import annotations

import argparse
import json
import shutil
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
    sha256_file,
    spec_sha256,
    utc_now,
    validate_selection_candidate_spec,
)
from tools.run_model_selection import candidate_spec  # noqa: E402
from tools.training_preflight import perform_preflight  # noqa: E402


QUALIFICATION_IDENTITY = {"candidate_id": "c01", "round_id": "r1", "optimization_seed": 4200}
SPEC_NAME = "generated_track_a_spec.json"
PREFLIGHT_NAME = "preflight_report.json"
MANIFEST_NAME = "qualification_manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"


def qualification_directory(path: str | Path) -> Path:
    """Return a persistent output directory outside the clean source checkout."""
    output = Path(path).resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ReproducibilityError("Track A evidence directory must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(item.name for item in output.iterdir() if item.name != SPEC_NAME)
    if unexpected:
        raise ReproducibilityError(
            "Track A evidence directory must be new, empty, or contain only its generated spec: "
            + ", ".join(unexpected)
        )
    return output


def qualification_spec() -> dict[str, Any]:
    spec = candidate_spec(**QUALIFICATION_IDENTITY)
    validate_selection_candidate_spec(spec, require_compute_authorized=True)
    return spec


def write_spec_once(path: Path, spec: dict[str, Any]) -> None:
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError("existing Track A generated spec is unreadable") from exc
        if canonical_json(current) != canonical_json(spec):
            raise ReproducibilityError("existing Track A generated spec has a different identity")
        return
    atomic_write_json(path, spec)


def gpu_probe_report(preflight: dict[str, Any]) -> dict[str, Any]:
    reports = [
        row.get("report")
        for row in preflight.get("validations", [])
        if "tools/gpu_training_probe.py" in row.get("command", [])
    ]
    if (
        len(reports) != 1
        or not isinstance(reports[0], dict)
        or reports[0].get("passed") is not True
    ):
        raise ReproducibilityError(
            "Track A preflight did not retain one passing GPU optimizer probe"
        )
    return reports[0]


def file_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_bundle(
    output: Path,
    spec: dict[str, Any],
    preflight: dict[str, Any],
    free_disk_gib: float,
) -> dict[str, Any]:
    spec_path = output / SPEC_NAME
    preflight_path = output / PREFLIGHT_NAME
    manifest_path = output / MANIFEST_NAME
    write_spec_once(spec_path, spec)
    atomic_write_json(preflight_path, preflight)
    probe = gpu_probe_report(preflight)
    manifest = {
        "schema_version": 1,
        "status": "qualified-no-training-started",
        "scope": "track-a-environment-and-real-bf16-optimizer-probe-only",
        "created_at": utc_now(),
        "source_commit": preflight["source_commit"],
        "source_clean": True,
        "selection_identity": dict(QUALIFICATION_IDENTITY),
        "spec_sha256": spec_sha256(spec),
        "minimum_free_disk_gib": float(spec["runtime"]["minimum_free_disk_gb_before_training"]),
        "observed_free_disk_gib": free_disk_gib,
        "runtime": preflight["runtime"],
        "dependencies": preflight["dependencies"],
        "gpu_optimizer_probe": probe,
        "heldout_namespaces_used": False,
        "campaign_training_started": False,
        "files": [
            file_identity(spec_path, relative_to=output),
            file_identity(preflight_path, relative_to=output),
        ],
    }
    atomic_write_json(manifest_path, manifest)
    checksum_paths = [spec_path, preflight_path, manifest_path]
    (output / CHECKSUMS_NAME).write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return manifest


def qualify(output_dir: str | Path) -> dict[str, Any]:
    output = qualification_directory(output_dir)
    spec = qualification_spec()
    spec_path = output / SPEC_NAME
    write_spec_once(spec_path, spec)
    free_disk_gib = shutil.disk_usage(output).free / 1024**3
    required_disk_gib = float(spec["runtime"]["minimum_free_disk_gb_before_training"])
    if free_disk_gib < required_disk_gib:
        raise ReproducibilityError(
            f"evidence storage has {free_disk_gib:.3f} GiB free; Track A requires at least "
            f"{required_disk_gib:.3f} GiB before training"
        )
    preflight = perform_preflight(spec_path, run_tests=True)
    if (
        preflight.get("passed") is not True
        or preflight.get("selection_candidate_authorized") is not True
    ):
        raise ReproducibilityError(
            "Track A preflight did not qualify development-selection training"
        )
    return write_bundle(output, spec, preflight, free_disk_gib)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New persistent directory outside the checkout",
    )
    args = parser.parse_args()
    try:
        manifest = qualify(args.output_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ReproducibilityError) as exc:
        print(f"STOP: Track A qualification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(canonical_json({
        "status": manifest["status"],
        "source_commit": manifest["source_commit"],
        "gpu": manifest["runtime"]["gpu"],
        "evidence_directory": str(Path(args.output_dir).resolve()),
        "campaign_training_started": False,
    }))
    print(
        "TRACK A PASS: stop here and submit the evidence bundle for review before campaign compute."
    )


if __name__ == "__main__":
    main()
