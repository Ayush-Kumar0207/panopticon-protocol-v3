import pytest
from research_paper.stage1_state_coverage.validation import validate_episode_header, validate_episode_turns

def test_seed_pairing_identity():
    header = {"level": "level_4", "seed": 42, "hydra_policy": "v2"}
    expected = {"level": "level_4", "seed": 42, "hydra_policy": "v2"}
    assert validate_episode_header(header, expected)["status"] == "accepted"

    bad_header = {"level": "level_4", "seed": 99, "hydra_policy": "v2"}
    res = validate_episode_header(bad_header, expected)
    assert res["status"] == "rejected"
    assert "seed_mismatch" in res["reasons"]

def test_hash_mismatch_rejected():
    res = validate_episode_header({"sha256": "bad"}, {"sha256": "good"})
    assert res["status"] == "rejected"
    assert "sha256_mismatch" in res["reasons"]

def test_duplicate_out_of_order_truncated():
    assert validate_episode_turns([{"turn": 1}, {"turn": 1}])["status"] == "rejected"
    assert validate_episode_turns([{"turn": 2}, {"turn": 1}])["status"] == "rejected"
    assert validate_episode_turns([{"step": 1}])["status"] == "rejected"

def test_synthetic_label_mandatory():
    # Verifies the required flag exists in the schema module
    from research_paper.stage1_state_coverage.schemas import TurnMetric
    metric = TurnMetric(
        schema_version="panopticon-stage1-turn-metrics-v2",
        synthetic=True,
        experiment_id="synthetic-stage1-test",
        run_fingerprint="1" * 64,
        checkpoint_sha256="2" * 64,
        source_commit="3" * 40,
        feature_extractor_version="synthetic-extractor-v1",
        seed=42,
        evidence_sha256="4" * 64,
        evidence_bytes=123,
        episode_id="ep1", level="level_4", turn=1, parse_success=True,
        raw_semantic_valid=True, executed_semantic_valid=True, intervention_applied=False
    )
    assert metric.synthetic is True
    assert metric.schema_version == "panopticon-stage1-turn-metrics-v2"
