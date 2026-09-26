
"""Synthetic-only tests; never verifies actual historical artifacts."""

import pytest

from research_paper.stage1_state_coverage.provenance import (
    TRAINING_MANIFEST_VERSION,
    validate_training_manifest,
)


def manifest():
    return {
        "schema_version": TRAINING_MANIFEST_VERSION,
        "synthetic": True,
        "experiment_id": "synthetic-stage1-test",
        "run_fingerprint": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "source_commit": "3" * 40,
        "training_stage": "synthetic-v5",
        "feature_extractor_version": "synthetic-extractor-v1",
        "dataset_files": [{
            "path": "training/synthetic.jsonl", "sha256": "a" * 64,
            "bytes": 123, "rows": 6,
        }],
        "metadata_files": [{
            "path": "training/synthetic-metadata.json",
            "sha256": "b" * 64, "bytes": 456,
        }],
        "tokenizer": {
            "identifier": "synthetic-tokenizer",
            "revision": "synthetic-tokenizer-revision",
            "chat_template_sha256": "d" * 64,
        },
        "formatter": {
            "source_commit": "3" * 40,
            "function": "train_trl_v2.format_observation",
        },
        "transforms": {"compaction": "unknown", "token_truncation": "unknown"},
        "mapping": {
            "mapping_file": {
                "path": "training/synthetic-row-map.json",
                "sha256": "c" * 64, "bytes": 345,
            },
            "independent_evidence": {
                "path": "evidence/synthetic-independent-ledger.json",
                "sha256": "e" * 64, "bytes": 567,
            },
            "mapped_rows": 6,
            "episode_groups": 6,
            "claimed_method": "independent_ledger",
        },
    }


def separate_expectations():
    # Deliberately defined independently of manifest()'s return value.
    return {
        "experiment_id": "synthetic-stage1-test",
        "run_fingerprint": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "source_commit": "3" * 40,
        "training_stage": "synthetic-v5",
        "feature_extractor_version": "synthetic-extractor-v1",
        "tokenizer_identifier": "synthetic-tokenizer",
        "tokenizer_revision": "synthetic-tokenizer-revision",
        "chat_template_sha256": "d" * 64,
        "formatter_source_commit": "3" * 40,
        "formatter_function": "train_trl_v2.format_observation",
        "compaction": "unknown",
        "token_truncation": "unknown",
        "dataset_sha256_by_path": {"training/synthetic.jsonl": "a" * 64},
        "dataset_bytes_by_path": {"training/synthetic.jsonl": 123},
        "dataset_rows_by_path": {"training/synthetic.jsonl": 6},
        "metadata_sha256_by_path": {"training/synthetic-metadata.json": "b" * 64},
        "metadata_bytes_by_path": {"training/synthetic-metadata.json": 456},
        "mapping_file_sha256": "c" * 64,
        "mapping_file_path": "training/synthetic-row-map.json",
        "mapping_file_bytes": 345,
        "mapping_evidence_sha256": "e" * 64,
        "mapping_evidence_path": "evidence/synthetic-independent-ledger.json",
        "mapping_evidence_bytes": 567,
        "mapped_rows": 6,
        "episode_groups": 6,
        "claimed_mapping_method": "independent_ledger",
    }


def test_complete_synthetic_manifest_stays_unverified():
    result = validate_training_manifest(manifest(), separate_expectations())
    assert result == {
        "status": "unverified",
        "reason": "structure_only_bytes_and_mapping_not_verified",
        "declared_dataset_rows": 6,
        "declared_episode_groups": 6,
    }


@pytest.mark.parametrize("missing", [
    "schema_version", "synthetic", "checkpoint_sha256", "source_commit",
    "dataset_files", "metadata_files", "tokenizer", "formatter",
    "transforms", "mapping",
])
def test_missing_required_contract_field(missing):
    value = manifest()
    del value[missing]
    assert validate_training_manifest(value, separate_expectations())["status"] == "rejected"


@pytest.mark.parametrize("path", [
    "../training/file.jsonl", "/absolute/file.jsonl", "C:/training/file.jsonl",
    r"C:\training\file.jsonl", "training//file.jsonl",
    "training/./file.jsonl", "training/file.jsonl\n",
])
def test_unsafe_dataset_paths_rejected(path):
    value = manifest()
    value["dataset_files"][0]["path"] = path
    result = validate_training_manifest(value, separate_expectations())
    assert result["status"] == "rejected"
    assert any("dataset_files.0.path" in reason for reason in result["reasons"])


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unexpected-v2"),
    ("checkpoint_sha256", "a" * 63),
    ("checkpoint_sha256", "A" * 64),
    ("synthetic", "true"),
    ("source_commit", "short"),
])
def test_malformed_top_level_values_rejected(field, value):
    record = manifest()
    record[field] = value
    assert validate_training_manifest(record, separate_expectations())["status"] == "rejected"


def test_mapping_does_not_silently_omit_weighted_rows():
    value = manifest()
    value["mapping"]["mapped_rows"] = 5
    result = validate_training_manifest(value, separate_expectations())
    assert result["status"] == "rejected"
    assert any("mapping_must_cover_every_persisted_row" in reason for reason in result["reasons"])


def test_duplicate_artifact_paths_rejected():
    value = manifest()
    value["mapping"]["mapping_file"]["path"] = "training/synthetic.jsonl"
    assert validate_training_manifest(value, separate_expectations())["status"] == "rejected"


def test_unexpected_fields_in_nested_record_rejected():
    value = manifest()
    value["mapping"]["independently_verified"] = True
    assert validate_training_manifest(value, separate_expectations())["status"] == "rejected"


def test_independent_expectations_are_mandatory():
    assert validate_training_manifest(manifest(), None) == {
        "status": "unavailable", "reason": "missing_independent_expectations",
    }


def test_expectation_hash_mismatch_rejected():
    expected = separate_expectations()
    expected["dataset_sha256_by_path"] = {
        "training/synthetic.jsonl": "f" * 64
    }
    assert validate_training_manifest(manifest(), expected) == {
        "status": "rejected",
        "reasons": ["dataset_sha256_by_path_mismatch"],
    }


def test_real_manifest_cannot_authorize_analysis():
    value = manifest()
    value["synthetic"] = False
    assert validate_training_manifest(value, separate_expectations()) == {
        "status": "unavailable", "reason": "real_requires_integrated_gate",
    }


def test_fake_self_consistent_hashes_do_not_establish_verification():
    value = manifest()
    expected = separate_expectations()
    value["dataset_files"][0]["sha256"] = "f" * 64
    expected["dataset_sha256_by_path"] = {
        "training/synthetic.jsonl": "f" * 64
    }
    assert validate_training_manifest(value, expected)["status"] == "unverified"


def test_boolean_row_count_and_missing_evidence_rejected():
    value = manifest()
    value["dataset_files"][0]["rows"] = True
    assert validate_training_manifest(value, separate_expectations())["status"] == "rejected"
    value = manifest()
    del value["mapping"]["independent_evidence"]
    assert validate_training_manifest(value, separate_expectations())["status"] == "rejected"



@pytest.mark.parametrize("field,wrong", [
    ("feature_extractor_version", "unexpected-extractor"),
    ("tokenizer_identifier", "different-tokenizer"),
    ("formatter_function", "other.formatter"),
    ("compaction", "present"),
    ("token_truncation", "present"),
    ("dataset_bytes_by_path", {"training/synthetic.jsonl": 124}),
    ("dataset_rows_by_path", {"training/synthetic.jsonl": 7}),
    ("metadata_bytes_by_path", {"training/synthetic-metadata.json": 457}),
    ("mapping_file_path", "training/other-map.json"),
    ("mapping_file_bytes", 346),
    ("mapping_evidence_path", "evidence/other-ledger.json"),
    ("mapping_evidence_bytes", 568),
    ("mapped_rows", 7),
    ("episode_groups", 5),
    ("claimed_mapping_method", "original_row_metadata"),
])
def test_new_expectation_fields_are_enforced(field, wrong):
    independent = separate_expectations()
    independent[field] = wrong
    assert validate_training_manifest(manifest(), independent) == {
        "status": "rejected",
        "reasons": [field + "_mismatch"],
    }


def test_nested_numeric_type_coercion_is_not_allowed():
    independent = separate_expectations()
    independent["dataset_rows_by_path"] = {
        "training/synthetic.jsonl": 6.0
    }
    assert validate_training_manifest(manifest(), independent) == {
        "status": "rejected",
        "reasons": ["dataset_rows_by_path_mismatch"],
    }


def test_matching_updated_row_totals_are_still_independently_checked():
    value = manifest()
    value["dataset_files"][0]["rows"] = 7
    value["mapping"]["mapped_rows"] = 7

    result = validate_training_manifest(
        value, separate_expectations()
    )
    assert result == {
        "status": "rejected",
        "reasons": [
            "dataset_rows_by_path_mismatch",
            "mapped_rows_mismatch",
        ],
    }



# Actual-file tests use temporary synthetic fixtures only.
import hashlib

from research_paper.stage1_state_coverage.provenance import (
    verify_synthetic_artifact_bytes,
)


def synthetic_tree(tmp_path):
    root = tmp_path / "synthetic-root"
    root.mkdir()

    value = manifest()
    expected = separate_expectations()

    payloads = {
        "training/synthetic.jsonl":
            b'{"text":"synthetic"}\n' * 6,
        "training/synthetic-metadata.json":
            b'{"synthetic":true}\n',
        "training/synthetic-row-map.json":
            b'{"mapping":"fixture-only"}\n',
        "evidence/synthetic-independent-ledger.json":
            b'{"independent":"claimed"}\n',
    }

    for relative, content in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    for entry in value["dataset_files"]:
        raw = payloads[entry["path"]]
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        entry["bytes"] = len(raw)

        expected["dataset_sha256_by_path"][entry["path"]] = (
            entry["sha256"]
        )
        expected["dataset_bytes_by_path"][entry["path"]] = len(raw)

    for entry in value["metadata_files"]:
        raw = payloads[entry["path"]]
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        entry["bytes"] = len(raw)

        expected["metadata_sha256_by_path"][entry["path"]] = (
            entry["sha256"]
        )
        expected["metadata_bytes_by_path"][entry["path"]] = len(raw)

    for kind, sha_field, size_field in (
        ("mapping_file", "mapping_file_sha256", "mapping_file_bytes"),
        (
            "independent_evidence",
            "mapping_evidence_sha256",
            "mapping_evidence_bytes",
        ),
    ):
        entry = value["mapping"][kind]
        raw = payloads[entry["path"]]

        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        entry["bytes"] = len(raw)

        expected[sha_field] = entry["sha256"]
        expected[size_field] = len(raw)

    return root, value, expected


def test_synthetic_byte_checks_remain_unverified(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    assert verify_synthetic_artifact_bytes(value, expected, root) == {
        "status": "unverified",
        "reason": "synthetic_bytes_verified_mapping_not_authenticated",
        "byte_verified_files": 4,
        "byte_verified_dataset_rows": 6,
    }


def test_same_size_mutation_rejected(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)
    path = root / "training/synthetic.jsonl"

    path.write_bytes(
        path.read_bytes().replace(b"synthetic", b"altered__")
    )

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "synthetic_file_sha256_mismatch"


def test_matching_forged_declarations_fail_against_bytes(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    value["dataset_files"][0]["sha256"] = "f" * 64
    expected["dataset_sha256_by_path"][
        "training/synthetic.jsonl"
    ] = "f" * 64

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "synthetic_file_sha256_mismatch"


def test_row_count_even_if_declarations_agree(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    value["dataset_files"][0]["rows"] = 5
    value["mapping"]["mapped_rows"] = 5
    value["mapping"]["episode_groups"] = 5

    expected["dataset_rows_by_path"][
        "training/synthetic.jsonl"
    ] = 5
    expected["mapped_rows"] = 5
    expected["episode_groups"] = 5

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "synthetic_row_count_mismatch"


@pytest.mark.parametrize("bad", [b"not-json", b"[]", b""])
def test_invalid_jsonl_rejected_even_with_correct_hash(tmp_path, bad):
    root, value, expected = synthetic_tree(tmp_path)

    content = b'{"text":"synthetic"}\n' * 5 + bad + b"\n"
    path = root / "training/synthetic.jsonl"
    path.write_bytes(content)

    value["dataset_files"][0]["sha256"] = (
        hashlib.sha256(content).hexdigest()
    )
    value["dataset_files"][0]["bytes"] = len(content)

    expected["dataset_sha256_by_path"][
        "training/synthetic.jsonl"
    ] = value["dataset_files"][0]["sha256"]

    expected["dataset_bytes_by_path"][
        "training/synthetic.jsonl"
    ] = len(content)

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "invalid_synthetic_jsonl"


def test_missing_synthetic_file_rejected(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)
    (root / "training/synthetic-metadata.json").unlink()

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "missing_synthetic_file"


def test_synthetic_file_symlink_rejected(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    target = root / "training/synthetic-metadata.json"
    target.unlink()

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")

    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is restricted on this machine")

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "unsafe_synthetic_path"


def test_synthetic_root_symlink_rejected(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)
    link = tmp_path / "root-link"

    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is restricted on this machine")

    result = verify_synthetic_artifact_bytes(
        value, expected, link
    )
    assert result["reason"] == "invalid_artifact_root"


def test_invalid_synthetic_root_rejected(tmp_path):
    _, value, expected = synthetic_tree(tmp_path)

    result = verify_synthetic_artifact_bytes(
        value, expected, tmp_path / "missing"
    )
    assert result["reason"] == "invalid_artifact_root"


def test_real_manifest_does_not_open_fixture_path(tmp_path):
    _, value, expected = synthetic_tree(tmp_path)
    value["synthetic"] = False

    assert verify_synthetic_artifact_bytes(
        value, expected, tmp_path / "missing"
    ) == {
        "status": "unavailable",
        "reason": "real_requires_integrated_gate",
    }


def test_expectation_mismatch_precedes_io(tmp_path):
    _, value, expected = synthetic_tree(tmp_path)
    expected["checkpoint_sha256"] = "9" * 64

    assert verify_synthetic_artifact_bytes(
        value, expected, tmp_path / "missing"
    ) == {
        "status": "rejected",
        "reasons": ["checkpoint_sha256_mismatch"],
    }


def test_self_consistent_wrong_size_rejected(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    value["metadata_files"][0]["bytes"] += 1
    expected["metadata_bytes_by_path"][
        "training/synthetic-metadata.json"
    ] += 1

    result = verify_synthetic_artifact_bytes(
        value, expected, root
    )
    assert result["reason"] == "synthetic_file_size_mismatch"



# Synthetic row-mapping correspondence tests.
import json

from research_paper.stage1_state_coverage.synthetic_mapping import (
    LEDGER_VERSION,
    ROW_MAP_VERSION,
    verify_synthetic_row_mapping,
)


def _write_row_document(root, value, expected, kind, rows=None, raw=None):
    fields = {
        "mapping_file": (
            ROW_MAP_VERSION,
            "mapping_file_sha256",
            "mapping_file_bytes",
        ),
        "independent_evidence": (
            LEDGER_VERSION,
            "mapping_evidence_sha256",
            "mapping_evidence_bytes",
        ),
    }
    version, hash_field, size_field = fields[kind]

    if raw is None:
        raw = json.dumps(
            {
                "schema_version": version,
                "synthetic": True,
                "rows": rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"

    entry = value["mapping"][kind]
    (root / entry["path"]).write_bytes(raw)

    digest = hashlib.sha256(raw).hexdigest()
    entry["sha256"] = digest
    entry["bytes"] = len(raw)

    expected[hash_field] = digest
    expected[size_field] = len(raw)


def _mapped_tree(tmp_path):
    root, value, expected = synthetic_tree(tmp_path)

    # Six persisted row occurrences, including weighted duplicates.
    identities = [
        ("episode-a", 11, "easy"),
        ("episode-a", 11, "easy"),
        ("episode-a", 11, "easy"),
        ("episode-b", 12, "easy"),
        ("episode-b", 12, "easy"),
        ("episode-c", 13, "level_4"),
    ]

    row_digest = hashlib.sha256(
        b'{"text":"synthetic"}'
    ).hexdigest()

    rows = [
        {
            "dataset_path": "training/synthetic.jsonl",
            "row_index": index,
            "row_sha256": row_digest,
            "episode_id": episode,
            "seed": seed,
            "level": level,
            "turn": 0,
        }
        for index, (episode, seed, level)
        in enumerate(identities)
    ]

    mapping_rows = [dict(row) for row in rows]
    ledger_rows = [dict(row) for row in rows]

    value["mapping"]["episode_groups"] = 3
    expected["episode_groups"] = 3

    _write_row_document(
        root, value, expected, "mapping_file", mapping_rows
    )
    _write_row_document(
        root, value, expected, "independent_evidence", ledger_rows
    )

    return root, value, expected, mapping_rows, ledger_rows


def test_mapping_occurrences_remain_unverified(tmp_path):
    root, value, expected, _, _ = _mapped_tree(tmp_path)

    assert verify_synthetic_row_mapping(
        value, expected, root
    ) == {
        "status": "unverified",
        "reason": (
            "synthetic_rows_correspond_but_"
            "independence_not_authenticated"
        ),
        "checked_row_occurrences": 6,
        "declared_episode_seed_groups": 3,
        "weighted_duplicate_occurrences": 3,
    }


def test_mapping_missing_persisted_row_rejected(tmp_path):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)
    rows.pop()

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "map_incomplete_row_coverage"


def test_mapping_duplicate_row_index_rejected(tmp_path):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)
    rows.append(dict(rows[0]))

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "map_duplicate_row_index"


@pytest.mark.parametrize("field,wrong,reason", [
    ("row_index", 99, "map_unknown_dataset_row"),
    ("row_sha256", "f" * 64, "map_row_hash_mismatch"),
    ("level", "unsupported", "invalid_synthetic_row_map"),
    ("episode_id", "", "invalid_synthetic_row_map"),
])
def test_mapping_invalid_row_rejected(
    tmp_path, field, wrong, reason
):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)
    rows[0][field] = wrong

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == reason


def test_mapping_contradictory_episode_seed_rejected(tmp_path):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)
    rows[1]["seed"] = 999

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "map_contradictory_episode_identity"


def test_mapping_same_seed_assigned_two_episodes_rejected(tmp_path):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)

    rows[5]["seed"] = 11
    rows[5]["level"] = "easy"

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "map_contradictory_seed_episode"


def test_mapping_ledger_disagreement_rejected(tmp_path):
    root, value, expected, _, ledger = _mapped_tree(tmp_path)

    for row in ledger[:3]:
        row["episode_id"] = "episode-alternative"

    _write_row_document(
        root, value, expected, "independent_evidence", ledger
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "synthetic_ledger_disagreement"


def test_mapping_incomplete_ledger_rejected(tmp_path):
    root, value, expected, _, ledger = _mapped_tree(tmp_path)
    ledger.pop()

    _write_row_document(
        root, value, expected, "independent_evidence", ledger
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "ledger_incomplete_row_coverage"


def test_mapping_episode_group_count_checked(tmp_path):
    root, value, expected, _, _ = _mapped_tree(tmp_path)

    value["mapping"]["episode_groups"] = 4
    expected["episode_groups"] = 4

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "synthetic_episode_group_mismatch"


def test_mapping_conflicting_weighted_duplicate_rejected(tmp_path):
    root, value, expected, rows, _ = _mapped_tree(tmp_path)

    dataset = root / "training/synthetic.jsonl"
    lines = dataset.read_bytes().splitlines()
    lines[1] = b'{"text":"different"}'

    content = b"\n".join(lines) + b"\n"
    dataset.write_bytes(content)

    entry = value["dataset_files"][0]
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    entry["bytes"] = len(content)

    expected["dataset_sha256_by_path"][
        entry["path"]
    ] = entry["sha256"]
    expected["dataset_bytes_by_path"][
        entry["path"]
    ] = len(content)

    rows[1]["row_sha256"] = hashlib.sha256(
        lines[1]
    ).hexdigest()

    _write_row_document(
        root, value, expected, "mapping_file", rows
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "map_contradictory_duplicate_turn"


def test_mapping_duplicate_json_keys_rejected(tmp_path):
    root, value, expected, _, _ = _mapped_tree(tmp_path)

    raw = (
        b'{"schema_version":'
        b'"panopticon-stage1-synthetic-row-map-v1",'
        b'"synthetic":true,"synthetic":true,"rows":[]}'
    )

    _write_row_document(
        root, value, expected, "mapping_file", raw=raw
    )

    result = verify_synthetic_row_mapping(
        value, expected, root
    )
    assert result["reason"] == "invalid_synthetic_row_map"


def test_mapping_real_data_remains_unavailable(tmp_path):
    root, value, expected, _, _ = _mapped_tree(tmp_path)
    value["synthetic"] = False

    result = verify_synthetic_row_mapping(
        value, expected, root / "nonexistent"
    )
    assert result == {
        "status": "unavailable",
        "reason": "real_requires_integrated_gate",
    }


def test_mapping_identity_mismatch_precedes_io(tmp_path):
    root, value, expected, _, _ = _mapped_tree(tmp_path)
    expected["checkpoint_sha256"] = "9" * 64

    result = verify_synthetic_row_mapping(
        value, expected, root / "nonexistent"
    )
    assert result == {
        "status": "rejected",
        "reasons": ["checkpoint_sha256_mismatch"],
    }
