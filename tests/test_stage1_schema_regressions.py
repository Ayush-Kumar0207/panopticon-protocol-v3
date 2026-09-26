"""Strict Stage 1 metric schema regression tests."""

import pytest
from pydantic import ValidationError

from research_paper.stage1_state_coverage.schemas import (
    TURN_METRICS_VERSION,
    TurnMetric,
)


def valid_record():
    return {
        "schema_version": TURN_METRICS_VERSION,
        "synthetic": True,
        "experiment_id": "synthetic-stage1-test",
        "run_fingerprint": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "source_commit": "3" * 40,
        "feature_extractor_version": "synthetic-extractor-v1",
        "seed": 42,
        "evidence_sha256": "4" * 64,
        "evidence_bytes": 123,
        "episode_id": "synthetic-episode-1",
        "level": "level_4",
        "turn": 0,
        "parse_success": True,
        "raw_semantic_valid": True,
        "executed_semantic_valid": True,
        "oracle_legal": None,
        "intervention_applied": False,
    }


@pytest.mark.parametrize("field", ["schema_version", "synthetic"])
def test_version_and_classification_are_mandatory(field):
    record = valid_record()
    del record[field]
    with pytest.raises(ValidationError):
        TurnMetric(**record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unexpected-v2"),
        ("synthetic", "true"),
        ("level", "level_6"),
        ("turn", -1),
        ("turn", True),
        ("parse_success", 1),
        ("episode_id", "  "),
        ("oracle_legal", "false"),
        ("executed_semantic_valid", None),
    ],
)
def test_invalid_fields_are_rejected(field, value):
    record = valid_record()
    record[field] = value
    with pytest.raises(ValidationError):
        TurnMetric(**record)


def test_unexpected_fields_are_rejected():
    record = valid_record()
    record["parse_sucess"] = True
    with pytest.raises(ValidationError):
        TurnMetric(**record)


@pytest.mark.parametrize(
    "raw_valid,executed_valid",
    [(True, False), (False, True)],
)
def test_validity_mismatch_requires_intervention(
    raw_valid, executed_valid
):
    record = valid_record()
    record["raw_semantic_valid"] = raw_valid
    record["executed_semantic_valid"] = executed_valid

    with pytest.raises(ValidationError):
        TurnMetric(**record)


def test_validity_repair_with_intervention_is_allowed():
    record = valid_record()
    record["raw_semantic_valid"] = False
    record["executed_semantic_valid"] = True
    record["intervention_applied"] = True
    assert TurnMetric(**record).intervention_applied is True


@pytest.mark.parametrize("intervened", [False, True])
def test_parse_failure_rejects_raw_validity(intervened):
    record = valid_record()
    record["parse_success"] = False
    record["intervention_applied"] = intervened

    with pytest.raises(
        ValidationError,
        match="parse_failed_raw_cannot_be_semantically_valid",
    ):
        TurnMetric(**record)


def test_parse_failure_without_valid_fallback():
    record = valid_record()
    record["parse_success"] = False
    record["raw_semantic_valid"] = False
    record["executed_semantic_valid"] = False

    metric = TurnMetric(**record)
    assert metric.raw_semantic_valid is False
    assert metric.executed_semantic_valid is False
    assert metric.intervention_applied is False


def test_parse_failure_with_valid_executed_fallback():
    record = valid_record()
    record["parse_success"] = False
    record["raw_semantic_valid"] = False
    record["executed_semantic_valid"] = True
    record["intervention_applied"] = True

    metric = TurnMetric(**record)
    assert metric.raw_semantic_valid is False
    assert metric.executed_semantic_valid is True
    assert metric.intervention_applied is True



@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "run_fingerprint",
        "checkpoint_sha256",
        "source_commit",
        "feature_extractor_version",
        "seed",
        "evidence_sha256",
        "evidence_bytes",
    ],
)
def test_missing_provenance_is_rejected(field):
    record = valid_record()
    del record[field]

    with pytest.raises(ValidationError):
        TurnMetric(**record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "  "),
        ("run_fingerprint", "not-a-hash"),
        ("run_fingerprint", "A" * 64),
        ("checkpoint_sha256", "bad"),
        ("source_commit", "short"),
        ("source_commit", "Z" * 40),
        ("feature_extractor_version", ""),
        ("seed", -1),
        ("seed", True),
        ("evidence_sha256", "0" * 63),
        ("evidence_bytes", 0),
        ("evidence_bytes", True),
    ],
)
def test_invalid_provenance_is_rejected(field, value):
    record = valid_record()
    record[field] = value

    with pytest.raises(ValidationError):
        TurnMetric(**record)


def test_provenance_survives_json_round_trip():
    record = valid_record()
    serialized = TurnMetric(**record).model_dump_json()
    restored = TurnMetric.model_validate_json(serialized)

    assert restored.model_dump() == record


def test_provenance_assignment_is_validated():
    metric = TurnMetric(**valid_record())

    with pytest.raises(ValidationError):
        metric.run_fingerprint = "invalid"


def test_real_classification_does_not_require_synthetic_flag():
    # Classification alone does not verify or authorize real evidence.
    record = valid_record()
    record["synthetic"] = False

    assert TurnMetric(**record).synthetic is False
