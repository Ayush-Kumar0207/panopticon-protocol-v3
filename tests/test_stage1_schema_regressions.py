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


def test_parse_failure_does_not_imply_semantic_invalidity():
    # A fallback NOOP can be semantically legal even after parse failure.
    record = valid_record()
    record["parse_success"] = False
    assert TurnMetric(**record).raw_semantic_valid is True
