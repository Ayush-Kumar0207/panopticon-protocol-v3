import pytest
from research_paper.stage1_state_coverage.replay import reconstruct_expert_state, OracleUnavailable

def test_new_memory_zeroed():
    state = reconstruct_expert_state([])
    assert state["audit_idx"] == 0
    assert state["monitor_idx"] == 0
    assert len(state["canaried_departments"]) == 0
    assert state["turned_sleeper"] is False

def test_prior_valid_canary_updates_memory():
    prior_turns = [{
        "executed_action": {"action_type": "canary", "target": "HR", "sub_action": "none"},
        "executed_semantic_valid": True,
        "info": {"valid": True}
    }]
    state = reconstruct_expert_state(prior_turns)
    assert "HR" in state["canaried_departments"]

def test_missing_validity_unavailable():
    prior_turns = [{
        "executed_action": {"action_type": "canary", "target": "HR", "sub_action": "none"},
        "executed_semantic_valid": True
        # Missing info['valid'] intentionally
    }]
    with pytest.raises(OracleUnavailable):
        reconstruct_expert_state(prior_turns)

def test_prior_invalid_action_does_not_advance():
    prior_turns = [{
        "executed_action": {"action_type": "monitor", "target": "IT", "sub_action": "none"},
        "executed_semantic_valid": False,
        "info": {"valid": True}
    }]
    state = reconstruct_expert_state(prior_turns)
    assert state["monitor_idx"] == 0
