"""Fail-closed Stage 1 artifact-integrity validation.

A matching declared SHA-256 in an episode header does not verify
actual evidence bytes. Use validate_evidence_blob or
validate_evidence_reference to perform that verification.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


def validate_episode_header(
    header: dict,
    expected_identity: dict,
) -> dict:
    """Compare declared identity fields without type coercion."""

    if (
        not isinstance(header, dict)
        or not isinstance(expected_identity, dict)
        or not expected_identity
    ):
        return {
            "status": "rejected",
            "reasons": ["invalid_header_or_expectations"],
        }

    mismatches = [
        f"{key}_mismatch"
        for key, expected in expected_identity.items()
        if (
            key not in header
            or type(header[key]) is not type(expected)
            or header[key] != expected
        )
    ]

    if mismatches:
        return {"status": "rejected", "reasons": mismatches}

    return {"status": "accepted"}


def validate_episode_turns(
    turns: list[dict],
    *,
    expected_turn_count: int | None = None,
    require_terminal: bool = False,
    start_turn: int = 0,
) -> dict:
    """Validate chronological, contiguous episode turns.

    The expected count must come from independent episode metadata.
    Otherwise, a contiguous but truncated prefix may go undetected.
    """

    if not isinstance(turns, list) or not turns:
        return {
            "status": "rejected",
            "reason": "missing_or_empty_turns",
        }

    if type(start_turn) is not int or start_turn < 0:
        return {
            "status": "rejected",
            "reason": "invalid_start_turn",
        }

    if expected_turn_count is not None:
        if (
            type(expected_turn_count) is not int
            or expected_turn_count < 1
        ):
            return {
                "status": "rejected",
                "reason": "invalid_expected_turn_count",
            }

    for index, row in enumerate(turns):
        if not isinstance(row, dict) or "turn" not in row:
            return {
                "status": "rejected",
                "reason": f"missing_turn_at_index_{index}",
            }

        turn = row["turn"]

        if type(turn) is not int:
            return {
                "status": "rejected",
                "reason": f"invalid_turn_at_index_{index}",
            }

        if turn != start_turn + index:
            kind = (
                "duplicate"
                if index and turn == turns[index - 1]["turn"]
                else "noncontiguous"
            )
            return {
                "status": "rejected",
                "reason": f"{kind}_turn_at_index_{index}",
            }

        for flag in ("done", "truncated"):
            if flag in row and type(row[flag]) is not bool:
                return {
                    "status": "rejected",
                    "reason": f"invalid_{flag}_at_index_{index}",
                }

        if index < len(turns) - 1:
            if row.get("done") is True or row.get("truncated") is True:
                return {
                    "status": "rejected",
                    "reason": f"premature_terminal_at_index_{index}",
                }

    if (
        expected_turn_count is not None
        and len(turns) != expected_turn_count
    ):
        return {
            "status": "rejected",
            "reason": "turn_count_mismatch",
        }

    if require_terminal:
        if not (
            turns[-1].get("done") is True
            or turns[-1].get("truncated") is True
        ):
            return {
                "status": "rejected",
                "reason": "missing_terminal_evidence",
            }

    return {
        "status": "accepted",
        "valid_count": len(turns),
    }


def validate_evidence_blob(
    compressed_blob: bytes,
    expected_sha256: str,
    expected_bytes: int,
) -> dict:
    """Verify gzip evidence against uncompressed canonical JSON bytes."""

    if not isinstance(compressed_blob, bytes):
        return {
            "status": "rejected",
            "reason": "missing_evidence_bytes",
        }

    if (
        not isinstance(expected_sha256, str)
        or not _SHA256.fullmatch(expected_sha256)
    ):
        return {
            "status": "rejected",
            "reason": "invalid_evidence_sha256",
        }

    if (
        type(expected_bytes) is not int
        or not (1 <= expected_bytes <= _MAX_EVIDENCE_BYTES)
    ):
        return {
            "status": "rejected",
            "reason": "invalid_evidence_size",
        }

    try:
        with gzip.GzipFile(
            fileobj=io.BytesIO(compressed_blob),
            mode="rb",
        ) as gz:
            raw = gz.read(expected_bytes + 1)

            if len(raw) == expected_bytes:
                trailing = gz.read(1)

                if trailing:
                    return {
                        "status": "rejected",
                        "reason": "evidence_size_mismatch",
                    }

    except (OSError, EOFError, ValueError):
        return {
            "status": "rejected",
            "reason": "corrupt_evidence_gzip",
        }

    if len(raw) != expected_bytes:
        return {
            "status": "rejected",
            "reason": "evidence_size_mismatch",
        }

    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return {
            "status": "rejected",
            "reason": "evidence_sha256_mismatch",
        }

    try:
        parsed = json.loads(raw.decode("utf-8"))

        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    except (UnicodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "rejected",
            "reason": "invalid_evidence_json",
        }

    if raw != canonical:
        return {
            "status": "rejected",
            "reason": "noncanonical_evidence_json",
        }

    return {
        "status": "accepted",
        "sha256": expected_sha256,
        "bytes": expected_bytes,
    }


def validate_evidence_reference(
    reference: dict,
    artifact_root: Path,
) -> dict:
    """Safely resolve a content-addressed path and verify its bytes."""

    if (
        not isinstance(reference, dict)
        or not isinstance(reference.get("artifact"), str)
    ):
        return {
            "status": "rejected",
            "reason": "invalid_evidence_reference",
        }

    digest = reference.get("sha256")
    relative = reference["artifact"]

    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
    ):
        return {
            "status": "rejected",
            "reason": "invalid_evidence_sha256",
        }

    if (
        "\\" in relative
        or ":" in relative
        or "\x00" in relative
    ):
        return {
            "status": "rejected",
            "reason": "unsafe_evidence_path",
        }

    path = PurePosixPath(relative)

    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 4
        or path.parts[-3:] != (
            "sha256",
            digest[:2],
            f"{digest}.json.gz",
        )
    ):
        return {
            "status": "rejected",
            "reason": "unsafe_evidence_path",
        }

    try:
        root = Path(artifact_root).resolve(strict=True)
        target = root.joinpath(*path.parts).resolve(strict=True)

        if not target.is_relative_to(root) or not target.is_file():
            return {
                "status": "rejected",
                "reason": "unsafe_evidence_path",
            }

        if target.stat().st_size > _MAX_EVIDENCE_BYTES:
            return {
                "status": "rejected",
                "reason": "oversize_evidence_file",
            }

        blob = target.read_bytes()

    except (OSError, ValueError, TypeError):
        return {
            "status": "rejected",
            "reason": "missing_evidence_file",
        }

    return validate_evidence_blob(
        blob,
        digest,
        reference.get("bytes"),
    )
