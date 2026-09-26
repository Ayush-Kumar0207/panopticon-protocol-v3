"""Synthetic artifact-integrity regression tests."""

import gzip
import hashlib
import json

from research_paper.stage1_state_coverage.validation import (
    validate_episode_header,
    validate_episode_turns,
    validate_evidence_blob,
    validate_evidence_reference,
)


def evidence(value):
    plain = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    digest = hashlib.sha256(plain).hexdigest()
    blob = gzip.compress(plain, compresslevel=9, mtime=0)

    return blob, digest, len(plain)


def test_gap_and_wrong_start_rejected():
    assert validate_episode_turns(
        [{"turn": 0}, {"turn": 2}]
    )["status"] == "rejected"

    assert validate_episode_turns(
        [{"turn": 3}]
    )["status"] == "rejected"


def test_empty_and_noninteger_turns_rejected():
    assert validate_episode_turns([])["status"] == "rejected"

    for turn in (True, 0.0, "0"):
        assert validate_episode_turns(
            [{"turn": turn}]
        )["status"] == "rejected"


def test_expected_count_detects_contiguous_truncation():
    result = validate_episode_turns(
        [{"turn": 0}, {"turn": 1}],
        expected_turn_count=3,
    )

    assert result["reason"] == "turn_count_mismatch"


def test_terminal_evidence_and_premature_terminal():
    assert validate_episode_turns(
        [{"turn": 0}],
        require_terminal=True,
    )["reason"] == "missing_terminal_evidence"

    assert validate_episode_turns(
        [
            {"turn": 0, "done": True},
            {"turn": 1, "done": True},
        ]
    )["reason"] == "premature_terminal_at_index_0"

    assert validate_episode_turns(
        [{"turn": 0, "done": True}],
        expected_turn_count=1,
        require_terminal=True,
    )["status"] == "accepted"


def test_header_missing_and_type_coercion_fail_closed():
    assert validate_episode_header(
        {"seed": False},
        {"seed": 0},
    )["status"] == "rejected"

    assert validate_episode_header(
        {},
        {"seed": None},
    )["status"] == "rejected"


def test_valid_synthetic_content_addressed_blob(tmp_path):
    blob, digest, size = evidence(
        {"turn": 0, "observation": {"workers": []}}
    )

    path = (
        tmp_path
        / "synthetic"
        / "sha256"
        / digest[:2]
        / f"{digest}.json.gz"
    )

    path.parent.mkdir(parents=True)
    path.write_bytes(blob)

    assert validate_evidence_blob(
        blob,
        digest,
        size,
    )["status"] == "accepted"

    reference = {
        "sha256": digest,
        "bytes": size,
        "artifact": path.relative_to(tmp_path).as_posix(),
    }

    assert validate_evidence_reference(
        reference,
        tmp_path,
    )["status"] == "accepted"


def test_wrong_hash_and_size_detected():
    blob, digest, size = evidence({"x": 1})

    assert validate_evidence_blob(
        blob,
        "0" * 64,
        size,
    )["reason"] == "evidence_sha256_mismatch"

    assert validate_evidence_blob(
        blob,
        digest,
        size - 1,
    )["reason"] == "evidence_size_mismatch"


def test_corrupt_gzip_rejected():
    blob, digest, size = evidence({"x": 1})

    assert validate_evidence_blob(
        b"not gzip",
        digest,
        size,
    )["reason"] == "corrupt_evidence_gzip"

    assert validate_evidence_blob(
        blob[:-5],
        digest,
        size,
    )["reason"] == "corrupt_evidence_gzip"


def test_noncanonical_json_rejected():
    noncanonical = b'{"z": 2, "a": 1}'
    blob = gzip.compress(noncanonical, mtime=0)
    digest = hashlib.sha256(noncanonical).hexdigest()

    assert validate_evidence_blob(
        blob,
        digest,
        len(noncanonical),
    )["reason"] == "noncanonical_evidence_json"


def test_unsafe_reference_path_rejected(tmp_path):
    blob, digest, size = evidence({"x": 1})

    reference = {
        "sha256": digest,
        "bytes": size,
        "artifact": (
            f"../sha256/{digest[:2]}/{digest}.json.gz"
        ),
    }

    assert validate_evidence_reference(
        reference,
        tmp_path,
    )["reason"] == "unsafe_evidence_path"

    reference["artifact"] = (
        f"C:\\other\\sha256\\{digest[:2]}\\{digest}.json.gz"
    )

    assert validate_evidence_reference(
        reference,
        tmp_path,
    )["reason"] == "unsafe_evidence_path"
