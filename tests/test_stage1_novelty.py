import pytest
from research_paper.stage1_state_coverage.novelty import score_learner_novelty, fit_reference_scaler

def test_missing_reference_or_few_neighbors():
    # Only 4 neighbors available (guide requires 5)
    refs = [{"episode_id": f"ep{i}", "features": {"turn": i}} for i in range(4)]
    res = score_learner_novelty({"turn": 1}, refs, "query_ep")
    assert res["status"] == "unavailable"
    assert res["reason"] == "insufficient_reference_episodes"

def test_no_self_neighbor():
    # 5 distinct out-of-group episodes, plus 1 in-group episode
    refs = [{"episode_id": f"ep{i}", "features": {"val": 10.0}} for i in range(5)]
    refs.append({"episode_id": "query_ep", "features": {"val": 99.0}})
    
    res = score_learner_novelty({"val": 99.0}, refs, "query_ep")
    assert res["status"] == "available"
    # The distance should be calculated against the 10.0 values, ignoring the 99.0 self-neighbor
    assert res["novelty_score"] > 0.0

def test_categorical_mismatch():
    refs = [{"episode_id": f"ep{i}", "features": {"phase": "planning"}} for i in range(5)]
    res = score_learner_novelty({"phase": "execution"}, refs, "query_ep")
    assert res["status"] == "available"
    # Categorical mismatch is defined as exactly 1.0 distance
    assert res["novelty_score"] == 1.0

def test_scaling_fit_on_reference_only():
    refs = [{"episode_id": f"ep{i}", "features": {"security": float(i)}} for i in range(5)]
    scaler = fit_reference_scaler([r["features"] for r in refs])
    assert "security" in scaler
    assert scaler["security"]["iqr"] > 0
