#!/usr/bin/env python3
"""Capture sanitized machine- and human-readable run provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import atomic_write_json, capture_provenance, load_spec  # noqa: E402


def write_provenance(spec_path: str, run_dir: str, command: list[str] | None = None) -> dict:
    spec, resolved_spec = load_spec(spec_path)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = capture_provenance(spec, resolved_spec, root, command)
    outputs = spec.get("outputs", {})
    json_path = root / outputs.get("provenance", "run_provenance.json")
    md_path = root / outputs.get("provenance_markdown", "RUN_PROVENANCE.md")
    atomic_write_json(json_path, payload)
    runtime = payload["runtime"]
    gpu = runtime["gpu"]
    markdown = f"""# Run provenance

- Captured: `{payload['captured_at']}`
- Source commit: `{payload['git']['commit']}`
- Working tree clean: `{not payload['git']['dirty']}`
- Training spec SHA-256: `{payload['spec_sha256']}`
- Run fingerprint: `{payload['run_fingerprint']}`
- Base model: `{payload['base_model']['id']}@{payload['base_model']['revision']}`
- Training seed: `{payload['seed']}`
- Episodes per level: `{payload['episodes_per_level']}`
- Python: `{runtime['python'].splitlines()[0]}`
- Platform: `{runtime['platform']}`
- GPU: `{gpu.get('name') or 'none'}`
- GPU VRAM GiB: `{gpu.get('vram_gb')}`
- CUDA: `{gpu.get('cuda')}`

The JSON file beside this document is authoritative. Environment variables are
allowlisted and secret-like keys are excluded/redacted.
"""
    md_path.write_text(markdown, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    payload = write_provenance(args.spec, args.run_dir)
    print(json.dumps({"run_fingerprint": payload["run_fingerprint"], "run_dir": args.run_dir}, indent=2))


if __name__ == "__main__":
    main()
