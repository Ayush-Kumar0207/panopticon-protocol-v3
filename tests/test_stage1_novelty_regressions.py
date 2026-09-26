"""Synthetic fail-closed and held-out calibration regression tests."""

from research_paper.stage1_state_coverage.novelty import (
    score_learner_novelty,
    calculate_distance,
    calibrate_expert_threshold,
)

def test_unshared_features_fail_closed():
    refs = [{"episode_id": f"ep{i}", "features": {"security": float(i)}} for i in range(5)]
    result = score_learner_novelty({"turn": 3}, refs, "query")
    assert result == {"status":"unavailable", "reason":"incomparable_features"}
    assert calculate_distance({"turn":2}, {"security":3}, {}) is None

def test_repeated_rows_do_not_count_as_independent_episodes():
    refs = [{"episode_id": "ep1", "features": {"val": float(i)}} for i in range(10)]
    result = score_learner_novelty({"val": 1.0}, refs, "query")
    assert result["reason"] == "insufficient_reference_episodes"

def test_seed_grouping_prevents_duplicate_episode_names():
    refs = [{"episode_id": f"ep{i}","seed":10,"level":"level_4","features":{"val":float(i)}} for i in range(5)]
    result = score_learner_novelty({"val": 1.0}, refs, "query", expected_level="level_4")
    assert result["reason"] == "insufficient_reference_episodes"

def test_held_out_calibration_groups_by_episode():
    refs = [{"episode_id":f"ep{i}","seed":i,"level":"level_4","features":{"val":float(i)}} for i in range(6)]
    result = calibrate_expert_threshold(refs, expected_level="level_4")
    assert result["status"] == "available"
    assert result["calibration_episodes"] == 6
    assert result["threshold"] >= 0

def test_calibration_requires_one_extra_episode():
    refs = [{"episode_id":f"ep{i}","features":{"val":float(i)}} for i in range(5)]
    result = calibrate_expert_threshold(refs)
    assert result["reason"] == "insufficient_calibration_episodes"

def test_repeated_turns_aggregate_per_episode():
    refs = [{"episode_id":f"ep{i}","seed":i,"level":"level_4","features":{"val":float(i+j/100)}} for i in range(6) for j in range(3)]
    result = calibrate_expert_threshold(refs, expected_level="level_4")
    assert result["status"] == "available"
    assert result["calibration_episodes"] == 6
