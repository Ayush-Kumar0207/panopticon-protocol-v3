import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from research_repro import ReproducibilityError
from tools.run_track_a_qualification import (
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    PREFLIGHT_NAME,
    QUALIFICATION_IDENTITY,
    SPEC_NAME,
    gpu_probe_report,
    qualification_directory,
    write_bundle,
)


def _spec():
    return {
        "runtime": {"minimum_free_disk_gb_before_training": 25.0},
        "experiment_id": "qualification-fixture",
        "base_model": {"id": "model", "revision": "revision"},
        "trajectory": {},
        "training": {},
    }


def _preflight():
    return {
        "source_commit": "a" * 40,
        "runtime": {"gpu": {"name": "fixture", "bf16_supported": True}},
        "dependencies": {"torch": "2.2.1"},
        "validations": [{
            "command": ["python", "tools/gpu_training_probe.py", "--spec", "fixture.json"],
            "returncode": 0,
            "report": {"passed": True, "precision": "bf16", "loss": 1.0, "gradient_norm": 2.0},
        }],
    }


@contextmanager
def _external_test_directory():
    root = Path(__file__).resolve().parents[2] / f"track-a-test-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_gpu_probe_report_requires_exactly_one_passing_report():
    assert gpu_probe_report(_preflight())["precision"] == "bf16"
    with pytest.raises(ReproducibilityError, match="one passing"):
        gpu_probe_report({"validations": []})


def test_write_bundle_is_hash_indexed():
    with _external_test_directory() as root:
        output = root / "track-a"
        output.mkdir()
        spec = _spec()
        (output / SPEC_NAME).write_text(json.dumps(spec), encoding="utf-8")
        manifest = write_bundle(output, spec, _preflight(), 99.0)

        assert manifest["status"] == "qualified-no-training-started"
        assert manifest["selection_identity"] == QUALIFICATION_IDENTITY
        assert manifest["campaign_training_started"] is False
        assert manifest["heldout_namespaces_used"] is False

        lines = (output / CHECKSUMS_NAME).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        for name in (SPEC_NAME, PREFLIGHT_NAME, MANIFEST_NAME):
            raw = (output / name).read_bytes()
            assert f"{hashlib.sha256(raw).hexdigest()}  {name}" in lines


def test_qualification_directory_rejects_source_checkout():
    with pytest.raises(ReproducibilityError, match="outside"):
        qualification_directory(Path(__file__).resolve().parents[1] / "track-a-output")


def test_qualification_directory_rejects_unrelated_existing_files():
    with _external_test_directory() as root:
        output = root / "track-a"
        output.mkdir()
        (output / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
        with pytest.raises(ReproducibilityError, match="new, empty"):
            qualification_directory(output)
