"""CPU-only synthetic expert-memory parity tests."""

import pytest

from environment import Environment
from models import validate_action
from security_policy import (
    choose_security_first_action,
    new_security_expert_state,
)

from research_paper.stage1_state_coverage.canonical import (
    canonicalize_action,
)
from research_paper.stage1_state_coverage.replay import (
    OracleUnavailable,
    label_learner_turn,
    reconstruct_expert_state,
)


def recorded(turn, kind, target="", sub="none",
             semantic=True, environment=True):
    return {
        "turn": turn,
        "executed_action": {
            "action_type": kind,
            "target": target,
            "sub_action": sub,
        },
        "executed_semantic_valid": semantic,
        "info": {"valid": environment},
    }


def test_every_expert_memory_transition():
    history = [
        recorded(0, "canary", "engineering"),
        recorded(1, "canary", "finance"),
        recorded(2, "canary", "operations"),
        recorded(3, "canary", "finance"),
        recorded(4, "investigate", "w-001", "audit"),
        recorded(5, "investigate", "w-002", "audit"),
        recorded(6, "monitor", "dark_web"),
        recorded(7, "monitor", "social_media"),
        recorded(8, "neutralize", "w-001", "turn"),
    ]

    state = reconstruct_expert_state(history)

    assert state["canaried_departments"] == {
        "engineering", "finance", "operations"
    }
    assert state["audit_idx"] == 2
    assert state["monitor_idx"] == 2
    assert state["turned_sleeper"] is True


@pytest.mark.parametrize(
    "semantic,environment",
    [(False, True), (True, False), (False, False)],
)
def test_invalid_actions_never_advance_memory(
    semantic, environment
):
    history = [
        recorded(
            0, "canary", "engineering",
            semantic=semantic, environment=environment,
        ),
        recorded(
            1, "investigate", "w-001", "audit",
            semantic=semantic, environment=environment,
        ),
        recorded(
            2, "monitor", "dark_web",
            semantic=semantic, environment=environment,
        ),
        recorded(
            3, "neutralize", "w-001", "turn",
            semantic=semantic, environment=environment,
        ),
    ]

    assert reconstruct_expert_state(
        history
    ) == new_security_expert_state()


@pytest.mark.parametrize(
    "problem",
    ["missing_info", "string_validity", "missing_action"],
)
def test_unverifiable_history_is_unavailable(problem):
    row = recorded(0, "canary", "engineering")

    if problem == "missing_info":
        del row["info"]
    elif problem == "string_validity":
        row["executed_semantic_valid"] = "true"
    else:
        row["executed_action"] = None

    with pytest.raises(OracleUnavailable):
        reconstruct_expert_state([row])


@pytest.mark.parametrize(
    "level",
    ["easy", "medium", "hard", "level_4", "level_5"],
)
@pytest.mark.parametrize("seed", [3, 17, 101])
def test_deterministic_environment_replay_parity(level, seed):
    # These are synthetic CPU test episodes, not research rollouts.
    env = Environment(seed=seed)
    observation = env.reset(task_level=level, seed=seed)

    original_expert_state = new_security_expert_state()
    recorded_history = []

    for turn in range(8):
        assert observation.turn == turn

        expected = choose_security_first_action(
            observation,
            level,
            original_expert_state,
        )

        actual = label_learner_turn(
            observation.model_dump(),
            level,
            recorded_history,
        )

        assert actual == canonicalize_action(
            expected.model_dump()
        )

        semantic_valid, reason = validate_action(
            expected, observation
        )
        assert semantic_valid, reason

        result = env.step(expected)
        assert result.info["valid"] is True

        recorded_history.append({
            "turn": turn,
            "executed_action": expected.model_dump(),
            "executed_semantic_valid": semantic_valid,
            "info": {"valid": result.info["valid"]},
        })

        observation = result.observation


def test_hypothetical_label_does_not_change_replay_memory():
    env = Environment(seed=17)
    obs = env.reset(
        task_level="level_4", seed=17
    ).model_dump()

    first = label_learner_turn(obs, "level_4", [])
    second = label_learner_turn(obs, "level_4", [])

    assert first == second
