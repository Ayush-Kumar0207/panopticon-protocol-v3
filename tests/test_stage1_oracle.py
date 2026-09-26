"""Synthetic tests for observable-history expert labeling."""

import pytest

from models import EnvironmentObservation
from security_policy import (
    choose_security_first_action,
    new_security_expert_state,
)
from research_paper.stage1_state_coverage.canonical import canonicalize_action
from research_paper.stage1_state_coverage.replay import (
    OracleUnavailable,
    _label_observation_from_history as label_learner_turn,
)


def make_observation(turn=0):
    return {
        "workers": [
            {
                "id": "w-001",
                "name": "Synthetic Worker",
                "department": "engineering",
                "state": "loyal",
                "hire_turn": 0,
                "suspicion_level": 0.0,
                "turning_in_progress": False,
            }
        ],
        "active_leaks": [],
        "canary_traps": [],
        "double_agents": [],
        "turn": turn,
        "max_turns": 40,
        "phase_number": 1,
        "security_score": 100.0,
    }


def valid_canary_turn():
    return {
        "turn": 0,
        "executed_action": {
            "action_type": "canary",
            "target": "engineering",
            "sub_action": "none",
        },
        "executed_semantic_valid": True,
        "info": {"valid": True},
    }


def test_initial_oracle_matches_original_expert():
    obs = make_observation(0)

    expected = choose_security_first_action(
        EnvironmentObservation.model_validate(obs),
        "level_4",
        new_security_expert_state(),
    )

    actual = label_learner_turn(obs, "level_4", [])

    assert actual == canonicalize_action(expected.model_dump())
    assert actual == ("canary", "engineering", "none")


def test_replay_reproduces_next_expert_action():
    state = new_security_expert_state()

    # The original expert acts at turn zero and updates its memory.
    choose_security_first_action(
        EnvironmentObservation.model_validate(make_observation(0)),
        "level_4",
        state,
    )

    # Its next action provides the controlled parity reference.
    expected = choose_security_first_action(
        EnvironmentObservation.model_validate(make_observation(1)),
        "level_4",
        state,
    )

    actual = label_learner_turn(
        make_observation(1),
        "level_4",
        [valid_canary_turn()],
    )

    assert actual == canonicalize_action(expected.model_dump())
    assert actual[0] == "monitor"


def test_missing_history_is_unavailable():
    with pytest.raises(OracleUnavailable):
        label_learner_turn(make_observation(1), "level_4", [])


def test_missing_validity_is_unavailable():
    turn = valid_canary_turn()
    del turn["info"]

    with pytest.raises(OracleUnavailable):
        label_learner_turn(make_observation(1), "level_4", [turn])


def test_invalid_action_does_not_advance_memory():
    turn = valid_canary_turn()
    turn["executed_semantic_valid"] = False

    result = label_learner_turn(
        make_observation(1),
        "level_4",
        [turn],
    )

    assert result == ("canary", "engineering", "none")


def test_missing_visible_worker_field_is_unavailable():
    obs = make_observation(0)
    del obs["workers"][0]["suspicion_level"]

    with pytest.raises(OracleUnavailable):
        label_learner_turn(obs, "level_4", [])


def test_multiple_consecutive_expert_actions_reproduce():
    """Replay must reproduce the original expert over several turns."""
    state = new_security_expert_state()
    history = []

    for turn in range(6):
        obs_dict = make_observation(turn)
        observation = EnvironmentObservation.model_validate(obs_dict)

        expected = choose_security_first_action(
            observation,
            "level_4",
            state,
        )

        actual = label_learner_turn(
            obs_dict,
            "level_4",
            history,
        )

        assert actual == canonicalize_action(expected.model_dump())

        history.append(
            {
                "turn": turn,
                "executed_action": expected.model_dump(),
                "executed_semantic_valid": True,
                "info": {"valid": True},
            }
        )


def test_nonconsecutive_history_is_unavailable():
    history = [valid_canary_turn()]
    history[0]["turn"] = 5

    with pytest.raises(OracleUnavailable):
        label_learner_turn(
            make_observation(1),
            "level_4",
            history,
        )


def test_replay_uses_executed_action_not_raw_action():
    history = [valid_canary_turn()]
    history[0]["raw_action"] = {
        "action_type": "monitor",
        "target": "dark_web",
        "sub_action": "none",
    }

    actual = label_learner_turn(
        make_observation(1),
        "level_4",
        history,
    )

    assert actual[0] == "monitor"


def test_illegal_expert_recommendation_is_rejected(monkeypatch):
    from models import AgentAction
    from research_paper.stage1_state_coverage import replay

    def illegal_expert(*args):
        return AgentAction(
            action_type="work",
            target="nonexistent_department",
        )

    monkeypatch.setattr(
        replay,
        "choose_security_first_action",
        illegal_expert,
    )

    with pytest.raises(replay.OracleValidationFailure):
        label_learner_turn(
            make_observation(0),
            "level_4",
            [],
        )



@pytest.mark.parametrize(
    "level",
    ["", "level_6", "amateur", None, 42],
)
def test_unsupported_task_level_is_unavailable(level):
    with pytest.raises(OracleUnavailable, match="unsupported task level"):
        label_learner_turn(make_observation(0), level, [])


@pytest.mark.parametrize(
    "level",
    ["easy", "medium", "hard", "level_4", "level_5"],
)
def test_supported_task_level_can_be_labeled(level):
    result = label_learner_turn(make_observation(0), level, [])
    assert result == ("canary", "engineering", "none")
