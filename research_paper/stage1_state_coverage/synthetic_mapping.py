
"""Synthetic-only row correspondence; never authenticates real provenance."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .provenance import (
    TrainingManifest,
    _read_synthetic_file,
    _strict_equal,
    verify_synthetic_artifact_bytes,
)


ROW_MAP_VERSION = "panopticon-stage1-synthetic-row-map-v1"
LEDGER_VERSION = "panopticon-stage1-synthetic-row-ledger-v1"


class RowWitness(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    dataset_path: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    level: Literal["easy", "medium", "hard", "level_4", "level_5"]
    turn: int = Field(ge=0)


class RowDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str
    synthetic: Literal[True]
    rows: list[RowWitness] = Field(min_length=1)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_document(content, version):
    try:
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=lambda value: (
                _ for _ in ()
            ).throw(ValueError(value)),
        )
        document = RowDocument.model_validate(raw)
        if document.schema_version != version:
            return None
        return document.rows
    except (UnicodeError, ValueError, ValidationError):
        return None


def _index_rows(rows, dataset_hashes, label):
    indexed = {}
    episodes = {}
    seed_groups = {}
    turns = {}
    repetitions = {}

    for row in rows:
        key = (row.dataset_path, row.row_index)

        if key not in dataset_hashes:
            return None, f"{label}_unknown_dataset_row"

        if key in indexed:
            return None, f"{label}_duplicate_row_index"

        if row.row_sha256 != dataset_hashes[key]:
            return None, f"{label}_row_hash_mismatch"

        identity = (row.seed, row.level)
        previous = episodes.setdefault(row.episode_id, identity)

        if previous != identity:
            return None, f"{label}_contradictory_episode_identity"

        previous_episode = seed_groups.setdefault(
            identity, row.episode_id
        )

        if previous_episode != row.episode_id:
            return None, f"{label}_contradictory_seed_episode"

        turn_key = (row.episode_id, row.turn)
        previous_hash = turns.setdefault(
            turn_key, row.row_sha256
        )

        if previous_hash != row.row_sha256:
            return None, f"{label}_contradictory_duplicate_turn"

        repetitions[turn_key] = (
            repetitions.get(turn_key, 0) + 1
        )

        indexed[key] = row

    if indexed.keys() != dataset_hashes.keys():
        return None, f"{label}_incomplete_row_coverage"

    return (indexed, seed_groups, repetitions), None


def verify_synthetic_row_mapping(
    manifest, expected_identity, artifact_root
):
    """Compare exact synthetic row occurrences with a supplied ledger.

    Matching documents do not authenticate their origins or independence.
    Real-data analysis remains unavailable.
    """
    preliminary = verify_synthetic_artifact_bytes(
        manifest, expected_identity, artifact_root
    )

    if preliminary["status"] != "unverified":
        return preliminary

    record = TrainingManifest.model_validate(manifest)
    root = Path(artifact_root).resolve(strict=True)

    files = [
        *record.dataset_files,
        record.mapping.mapping_file,
        record.mapping.independent_evidence,
    ]

    blobs = {}

    for item in files:
        content, reason = _read_synthetic_file(root, item)

        if reason is not None:
            return {
                "status": "rejected",
                "reason": reason,
                "artifact": item.path,
            }

        blobs[item.path] = content

    mapping = _parse_document(
        blobs[record.mapping.mapping_file.path],
        ROW_MAP_VERSION,
    )
    ledger = _parse_document(
        blobs[record.mapping.independent_evidence.path],
        LEDGER_VERSION,
    )

    if mapping is None:
        return {
            "status": "rejected",
            "reason": "invalid_synthetic_row_map",
        }

    if ledger is None:
        return {
            "status": "rejected",
            "reason": "invalid_synthetic_row_ledger",
        }

    dataset_hashes = {}

    for item in record.dataset_files:
        lines = blobs[item.path].split(b"\n")

        if lines[-1] == b"":
            lines.pop()

        for index, line in enumerate(lines):
            dataset_hashes[(item.path, index)] = (
                hashlib.sha256(line).hexdigest()
            )

    mapped, reason = _index_rows(
        mapping, dataset_hashes, "map"
    )

    if reason is not None:
        return {"status": "rejected", "reason": reason}

    witnessed, reason = _index_rows(
        ledger, dataset_hashes, "ledger"
    )

    if reason is not None:
        return {"status": "rejected", "reason": reason}

    mapped_rows, groups, repetitions = mapped
    ledger_rows, _, _ = witnessed

    for key, row in mapped_rows.items():
        if not _strict_equal(
            row.model_dump(),
            ledger_rows[key].model_dump(),
        ):
            return {
                "status": "rejected",
                "reason": "synthetic_ledger_disagreement",
            }

    if len(groups) != record.mapping.episode_groups:
        return {
            "status": "rejected",
            "reason": "synthetic_episode_group_mismatch",
        }

    return {
        "status": "unverified",
        "reason": (
            "synthetic_rows_correspond_but_"
            "independence_not_authenticated"
        ),
        "checked_row_occurrences": len(mapped_rows),
        "declared_episode_seed_groups": len(groups),
        "weighted_duplicate_occurrences": sum(
            count - 1 for count in repetitions.values()
        ),
    }
