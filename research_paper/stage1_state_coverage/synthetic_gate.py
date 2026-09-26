"""First-turn synthetic integration; never authorizes real analysis.

This provisional entry point checks existing components together.
It does not authenticate independent provenance or support later turns.
"""

import hashlib
import json
from pathlib import Path

from ._synthetic_features import CANDIDATE_FEATURE_VERSION
from .feature_eligibility import screen_synthetic_feature_parity
from .provenance import TrainingManifest, _read_synthetic_file
from .replay import (
    OracleUnavailable,
    OracleValidationFailure,
    label_learner_turn,
)
from .synthetic_mapping import (
    ROW_MAP_VERSION,
    _parse_document,
    _unique_keys,
    verify_synthetic_row_mapping,
)
from .validation import validate_episode_header


def _unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def screen_synthetic_first_turn(
    manifest,
    expected_manifest_identity,
    artifact_root,
    *,
    dataset_path,
    row_index,
    episode_header,
    current_row,
):
    """Check one actual persisted synthetic row and its initial oracle."""

    # All filesystem access follows the existing synthetic-only checks.
    mapping_result = verify_synthetic_row_mapping(
        manifest, expected_manifest_identity, artifact_root
    )

    if mapping_result["status"] != "unverified":
        return mapping_result

    if mapping_result.get("reason") != (
        "synthetic_rows_correspond_but_"
        "independence_not_authenticated"
    ):
        return _unavailable("unexpected_mapping_status")

    record = TrainingManifest.model_validate(manifest)

    if (
        record.feature_extractor_version
        != CANDIDATE_FEATURE_VERSION
    ):
        return _unavailable("feature_version_mismatch")

    if (
        type(dataset_path) is not str
        or type(row_index) is not int
        or row_index < 0
    ):
        return _unavailable("invalid_selected_row")

    datasets = {
        item.path: item for item in record.dataset_files
    }

    if dataset_path not in datasets:
        return _unavailable("unknown_dataset_path")

    # Re-read both files against their declared byte hashes.
    # This is a synthetic fixture check, not a race-proof security gate.
    root = Path(artifact_root).resolve(strict=True)

    dataset, error = _read_synthetic_file(
        root, datasets[dataset_path]
    )

    if error is not None:
        return _unavailable(error)

    mapping_bytes, error = _read_synthetic_file(
        root, record.mapping.mapping_file
    )

    if error is not None:
        return _unavailable(error)

    mapped_rows = _parse_document(
        mapping_bytes, ROW_MAP_VERSION
    )

    if mapped_rows is None:
        return _unavailable("invalid_synthetic_row_map")

    selected = [
        row for row in mapped_rows
        if row.dataset_path == dataset_path
        and row.row_index == row_index
    ]

    if len(selected) != 1:
        return _unavailable("selected_row_not_uniquely_mapped")

    witness = selected[0]

    lines = dataset.split(b"\n")

    if lines[-1] == b"":
        lines.pop()

    if row_index >= len(lines):
        return _unavailable("selected_row_out_of_range")

    raw_row = lines[row_index]

    if (
        hashlib.sha256(raw_row).hexdigest()
        != witness.row_sha256
    ):
        return _unavailable("selected_row_hash_mismatch")

    try:
        persisted = json.loads(
            raw_row.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=lambda value: (
                _ for _ in ()
            ).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError):
        return _unavailable("invalid_selected_jsonl_row")

    if (
        type(persisted) is not dict
        or type(persisted.get("text")) is not str
    ):
        return _unavailable("missing_persisted_training_text")

    # This initial integration deliberately handles turn zero only.
    if witness.turn != 0:
        return _unavailable("noninitial_turn_not_yet_supported")

    identity = {
        "synthetic": True,
        "experiment_id": record.experiment_id,
        "run_fingerprint": record.run_fingerprint,
        "checkpoint_sha256": record.checkpoint_sha256,
        "source_commit": record.source_commit,
        "feature_extractor_version": (
            record.feature_extractor_version
        ),
        "episode_id": witness.episode_id,
        "seed": witness.seed,
        "level": witness.level,
    }

    bound_header = validate_episode_header(
        episode_header, identity
    )

    if bound_header["status"] != "accepted":
        return _unavailable("episode_identity_mismatch")

    if type(current_row) is not dict:
        return _unavailable("invalid_current_row")

    observation = current_row.get("observation_before")

    if (
        type(current_row.get("turn")) is not int
        or current_row["turn"] != 0
        or type(observation) is not dict
        or type(observation.get("turn")) is not int
        or observation["turn"] != 0
    ):
        return _unavailable("selected_turn_mismatch")

    parity = screen_synthetic_feature_parity(
        persisted["text"],
        observation,
        synthetic=True,
        expected_feature_version=(
            record.feature_extractor_version
        ),
    )

    if parity["status"] != "unverified":
        return {
            "status": "unavailable",
            "reason": "feature_screen_unavailable",
            "detail": parity,
        }

    try:
        oracle_label = label_learner_turn(
            current_row,
            witness.level,
            [],
            episode_header=episode_header,
            expected_identity=identity,
        )
    except (OracleUnavailable, OracleValidationFailure) as exc:
        return {
            "status": "unavailable",
            "reason": "replay_unavailable",
            "detail": str(exc),
        }

    return {
        "status": "unverified",
        "reason": (
            "synthetic_first_turn_checks_passed_"
            "independence_unverified"
        ),
        "dataset_path": dataset_path,
        "row_index": row_index,
        "feature_version": record.feature_extractor_version,
        "feature_count": len(parity["features"]),
        "oracle_label": oracle_label,
    }


def screen_synthetic_two_turn(
    manifest,
    expected_manifest_identity,
    artifact_root,
    *,
    dataset_path,
    prior_row_index,
    row_index,
    episode_header,
    prior_row,
    current_row,
):
    """Check a two-turn synthetic record; independence remains unverified."""

    if (
        type(prior_row_index) is not int
        or type(row_index) is not int
        or prior_row_index < 0
        or row_index < 0
        or prior_row_index == row_index
    ):
        return _unavailable("invalid_two_turn_selection")

    # Check the actual persisted prior row through the existing gate.
    prior_result = screen_synthetic_first_turn(
        manifest,
        expected_manifest_identity,
        artifact_root,
        dataset_path=dataset_path,
        row_index=prior_row_index,
        episode_header=episode_header,
        current_row=prior_row,
    )

    if prior_result["status"] != "unverified":
        return {**prior_result, "failed_turn": 0}

    if prior_result.get("reason") != (
        "synthetic_first_turn_checks_passed_"
        "independence_unverified"
    ):
        return _unavailable("unexpected_prior_gate_status")

    # Recheck artifact bytes and the complete row-to-ledger mapping.
    mapping_result = verify_synthetic_row_mapping(
        manifest, expected_manifest_identity, artifact_root
    )

    if mapping_result["status"] != "unverified":
        return mapping_result

    record = TrainingManifest.model_validate(manifest)
    datasets = {
        item.path: item for item in record.dataset_files
    }

    if dataset_path not in datasets:
        return _unavailable("unknown_dataset_path")

    root = Path(artifact_root).resolve(strict=True)

    dataset, error = _read_synthetic_file(
        root, datasets[dataset_path]
    )
    if error is not None:
        return _unavailable(error)

    mapping_bytes, error = _read_synthetic_file(
        root, record.mapping.mapping_file
    )
    if error is not None:
        return _unavailable(error)

    mapped_rows = _parse_document(
        mapping_bytes, ROW_MAP_VERSION
    )
    if mapped_rows is None:
        return _unavailable("invalid_synthetic_row_map")

    prior_matches = [
        row for row in mapped_rows
        if row.dataset_path == dataset_path
        and row.row_index == prior_row_index
    ]
    current_matches = [
        row for row in mapped_rows
        if row.dataset_path == dataset_path
        and row.row_index == row_index
    ]

    if len(prior_matches) != 1 or len(current_matches) != 1:
        return _unavailable("two_turn_rows_not_uniquely_mapped")

    previous = prior_matches[0]
    selected = current_matches[0]

    if previous.turn != 0 or selected.turn != 1:
        return _unavailable("two_turn_witness_mismatch")

    if (
        previous.episode_id,
        previous.seed,
        previous.level,
    ) != (
        selected.episode_id,
        selected.seed,
        selected.level,
    ):
        return _unavailable("prior_mapping_identity_mismatch")

    lines = dataset.split(b"\n")
    if lines[-1] == b"":
        lines.pop()

    if row_index >= len(lines):
        return _unavailable("selected_row_out_of_range")

    raw = lines[row_index]

    if hashlib.sha256(raw).hexdigest() != selected.row_sha256:
        return _unavailable("selected_row_hash_mismatch")

    try:
        persisted = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=lambda value: (
                _ for _ in ()
            ).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError):
        return _unavailable("invalid_selected_jsonl_row")

    if (
        type(persisted) is not dict
        or type(persisted.get("text")) is not str
    ):
        return _unavailable("missing_persisted_training_text")

    identity = {
        "synthetic": True,
        "experiment_id": record.experiment_id,
        "run_fingerprint": record.run_fingerprint,
        "checkpoint_sha256": record.checkpoint_sha256,
        "source_commit": record.source_commit,
        "feature_extractor_version": (
            record.feature_extractor_version
        ),
        "episode_id": selected.episode_id,
        "seed": selected.seed,
        "level": selected.level,
    }

    if validate_episode_header(
        episode_header, identity
    )["status"] != "accepted":
        return _unavailable("episode_identity_mismatch")

    if (
        type(current_row) is not dict
        or type(current_row.get("turn")) is not int
        or current_row["turn"] != 1
    ):
        return _unavailable("selected_turn_mismatch")

    observation = current_row.get("observation_before")

    if (
        type(observation) is not dict
        or type(observation.get("turn")) is not int
        or observation["turn"] != 1
    ):
        return _unavailable("selected_turn_mismatch")

    if validate_episode_header(
        current_row, identity
    )["status"] != "accepted":
        return _unavailable("current_identity_mismatch")

    parity = screen_synthetic_feature_parity(
        persisted["text"],
        observation,
        synthetic=True,
        expected_feature_version=(
            record.feature_extractor_version
        ),
    )

    if parity["status"] != "unverified":
        return {
            "status": "unavailable",
            "reason": "feature_screen_unavailable",
            "failed_turn": 1,
            "detail": parity,
        }

    try:
        oracle_label = label_learner_turn(
            current_row,
            selected.level,
            [prior_row],
            episode_header=episode_header,
            expected_identity=identity,
        )
    except (OracleUnavailable, OracleValidationFailure) as exc:
        return {
            "status": "unavailable",
            "reason": "replay_unavailable",
            "detail": str(exc),
        }

    return {
        "status": "unverified",
        "reason": (
            "synthetic_two_turn_checks_passed_"
            "independence_unverified"
        ),
        "dataset_path": dataset_path,
        "row_index": row_index,
        "checked_turns": 2,
        "feature_version": record.feature_extractor_version,
        "feature_count": len(parity["features"]),
        "oracle_label": oracle_label,
    }
