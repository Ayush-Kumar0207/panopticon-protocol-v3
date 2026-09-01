#!/usr/bin/env python3
"""Install and validate the exact CUDA-enabled Panopticon training stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = "2.2.1"
TORCH_BUILD = "cu121"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def install_commands() -> list[list[str]]:
    """Return the auditable commands used by the canonical installer."""
    return [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            TORCH_INDEX,
            f"torch=={TORCH_VERSION}",
        ],
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-training.txt")],
        [sys.executable, "-m", "pip", "check"],
    ]


def validate_runtime() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on contributor runtimes
        raise RuntimeError(f"PyTorch cannot be imported: {type(exc).__name__}") from exc

    distribution_version = str(torch.__version__)
    base_version, _, local_build = distribution_version.partition("+")
    if base_version != TORCH_VERSION or local_build != TORCH_BUILD:
        raise RuntimeError(
            f"expected torch {TORCH_VERSION}+{TORCH_BUILD}, found {distribution_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("installed PyTorch does not expose CUDA")
    if str(torch.version.cuda) != "12.1":
        raise RuntimeError(f"expected PyTorch CUDA runtime 12.1, found {torch.version.cuda}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "canonical BF16 requires an NVIDIA Ampere-or-newer GPU; T4/Turing is not supported"
        )
    return {
        "torch": distribution_version,
        "torch_cuda_runtime": str(torch.version.cuda),
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "bf16_supported": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the exact commands without changing the environment"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        print(
            f"STOP: canonical installation requires Python 3.11; found "
            f"{sys.version_info.major}.{sys.version_info.minor}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    commands = install_commands()
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return
    try:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        report = validate_runtime()
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"STOP: canonical CUDA environment is not ready: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("INSTALL PASS: exact CUDA/BF16 runtime is ready for canonical preflight.")


if __name__ == "__main__":
    main()
