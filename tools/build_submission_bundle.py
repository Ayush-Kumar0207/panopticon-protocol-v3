#!/usr/bin/env python3
"""Verify a Panopticon run and create a bounded, status-labelled evidence ZIP."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import ReproducibilityError, canonical_json, load_spec, sha256_file  # noqa: E402
from tools.verify_training_artifacts import verify_run  # noqa: E402


MAX_EMBEDDED_FILE_BYTES = 10 * 1024 * 1024
MAX_EMBEDDED_TOTAL_BYTES = 50 * 1024 * 1024
ALWAYS_EXTERNAL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def _zip_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def build_bundle(
    run_dir: str | Path, spec_path: str | Path, *, enforce_checkout: bool = True,
) -> tuple[Path, dict]:
    root = Path(run_dir).resolve()
    spec, _ = load_spec(spec_path)
    verification = verify_run(root, spec_path, enforce_checkout=enforce_checkout)
    outputs = spec["outputs"]
    manifest_path = root / outputs["artifact_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = root / outputs["submission_bundle"]
    if output.exists():
        raise ReproducibilityError(
            f"submission bundle already exists: {output}; move it aside explicitly before rebuilding"
        )

    included: list[dict] = []
    external_required: list[dict] = []
    embedded_total = 0
    for item in sorted(manifest["files"], key=lambda value: value["path"]):
        path = root / item["path"]
        is_checkpoint = any(part.startswith("checkpoint-") for part in Path(item["path"]).parts)
        too_large = item["size"] > MAX_EMBEDDED_FILE_BYTES or embedded_total + item["size"] > MAX_EMBEDDED_TOTAL_BYTES
        if path.suffix.lower() in ALWAYS_EXTERNAL_SUFFIXES or is_checkpoint or too_large:
            external_required.append({**item, "reason": "large/model/checkpoint artifact; upload separately without changing bytes"})
        else:
            included.append(item)
            embedded_total += item["size"]

    for required_name in (outputs["artifact_manifest"], outputs["checksums"]):
        path = root / required_name
        item = {"path": required_name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        if not any(value["path"] == required_name for value in included):
            included.append(item)

    index = {
        "schema_version": 1,
        "scientific_status": verification["scientific_status"],
        "accepted": verification["accepted"],
        "run_fingerprint": verification["run_fingerprint"],
        "bundle_policy": {
            "maximum_embedded_file_bytes": MAX_EMBEDDED_FILE_BYTES,
            "maximum_embedded_total_bytes": MAX_EMBEDDED_TOTAL_BYTES,
            "meaning": "Excluded files remain mandatory external artifacts and are bound by artifact_manifest.json hashes.",
        },
        "included": sorted(included, key=lambda value: value["path"]),
        "external_required": external_required,
        "interpretation": (
            "ACCEPTED: every canonical and confirmation gate passed."
            if verification["accepted"] else
            "REJECTED/NONCANONICAL: integrity evidence is preserved, but one or more scientific gates failed."
        ),
    }
    with zipfile.ZipFile(output, "x") as archive:
        for item in index["included"]:
            _zip_bytes(archive, item["path"], (root / item["path"]).read_bytes())
        _zip_bytes(archive, "SUBMISSION_INDEX.json", (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return output, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    args = parser.parse_args()
    try:
        output, index = build_bundle(args.run_dir, args.spec)
    except Exception as exc:
        print(f"STOP: submission bundle was not created: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(canonical_json({"bundle": str(output), "sha256": sha256_file(output), "status": index["scientific_status"]}))
    if not index["accepted"]:
        print("BUNDLE CREATED AS REJECTED/NONCANONICAL EVIDENCE; do not present it as accepted.", file=sys.stderr)
        raise SystemExit(2)
    print("SUBMISSION BUNDLE READY: integrity and all scientific gates passed.")


if __name__ == "__main__":
    main()
