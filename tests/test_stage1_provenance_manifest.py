
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
        "tokenizer_revision": "synthetic-tokenizer-revision",
        "chat_template_sha256": "d" * 64,
        "formatter_source_commit": "3" * 40,
        "dataset_sha256_by_path": {"training/synthetic.jsonl": "a" * 64},
        "metadata_sha256_by_path": {"training/synthetic-metadata.json": "b" * 64},
        "mapping_file_sha256": "c" * 64,
        "mapping_evidence_sha256": "e" * 64,
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
