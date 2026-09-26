"""Synthetic identity-consistency tests for bound replay."""

import copy
import pytest

from research_paper.stage1_state_coverage.replay import (
    OracleUnavailable,
    label_learner_turn,
)


IDENTITY = {
    "synthetic": True,
    "experiment_id": "synthetic-stage1-test",
    "run_fingerprint": "1" * 64,
    "checkpoint_sha256": "2" * 64,
    "source_commit": "3" * 40,
    "feature_extractor_version": "synthetic-extractor-v1",
    "episode_id": "synthetic-episode-1",
    "seed": 42,
    "level": "level_4",
}


def observation(turn):
    return {
        "workers": [{
            "id": "w-001",
            "name": "Synthetic Worker",
            "department": "engineering",
            "state": "loyal",
            "hire_turn": 0,
            "suspicion_level": 0.0,
            "turning_in_progress": False,
        }],
        "active_leaks": [],
        "canary_traps": [],
        "double_agents": [],
        "turn": turn,
        "max_turns": 40,
        "phase_number": 1,
        "security_score": 100.0,
    }


def row(turn):
    return {
        **IDENTITY,
        "turn": turn,
        "observation_before": observation(turn),
        "executed_action": {
            "action_type": "canary",
            "target": "engineering",
            "sub_action": "none",
        },
        "executed_semantic_valid": True,
        "info": {"valid": True},
    }


def bound(current, history=None, header=None, identity=None):
    return label_learner_turn(
        current,
        "level_4",
        [] if history is None else history,
        episode_header=IDENTITY.copy() if header is None else header,
        expected_identity=IDENTITY.copy() if identity is None else identity,
    )


def test_bound_initial_turn():
    assert bound(row(0)) == ("canary", "engineering", "none")


def test_bound_replay_after_executed_canary():
    assert bound(row(1), [row(0)])[0] == "monitor"


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "run_fingerprint",
        "checkpoint_sha256",
        "source_commit",
        "feature_extractor_version",
        "episode_id",
        "seed",
        "level",
        "synthetic",
    ],
)
def test_cross_identity_history_rejected(field):
    prior = row(0)
    prior[field] = (
        False if field == "synthetic"
        else 99 if field == "seed"
        else "other"
    )
    with pytest.raises(OracleUnavailable, match="row identity mismatch"):
        bound(row(1), [prior])


@pytest.mark.parametrize(
    "field",
    ["run_fingerprint", "checkpoint_sha256",
     "source_commit", "episode_id", "seed", "level"],
)
def test_wrong_episode_header_rejected(field):
    header = IDENTITY.copy()
    header[field] = 99 if field == "seed" else "other"
    with pytest.raises(OracleUnavailable, match="episode-header"):
        bound(row(0), header=header)


def test_missing_independent_identity_rejected():
    with pytest.raises(OracleUnavailable, match="independent"):
        label_learner_turn(row(0), "level_4", [])


def test_real_classification_cannot_bypass_gate():
    identity = IDENTITY.copy()
    identity["synthetic"] = False
    with pytest.raises(OracleUnavailable, match="integrated provenance"):
        bound(row(0), identity=identity)


def test_cross_episode_current_row_rejected():
    current = row(0)
    current["episode_id"] = "different-episode"
    with pytest.raises(OracleUnavailable, match="row identity mismatch"):
        bound(current)


def test_missing_middle_turn_rejected():
    with pytest.raises(OracleUnavailable, match="incomplete"):
        bound(row(2), [row(0)])


def test_wrong_observation_turn_rejected():
    prior = row(0)
    prior["observation_before"]["turn"] = 3
    with pytest.raises(OracleUnavailable, match="observation"):
        bound(row(1), [prior])


def test_terminal_history_rejected():
    prior = row(0)
    prior["done"] = True
    with pytest.raises(OracleUnavailable, match="sequence"):
        bound(row(1), [prior])


def test_invalid_recorded_action_does_not_advance_memory():
    prior = row(0)
    prior["executed_semantic_valid"] = False
    assert bound(row(1), [prior]) == (
        "canary", "engineering", "none"
    )



@pytest.mark.parametrize("level", [None, 42, True, [], {}])
def test_malformed_replay_level_is_unavailable(level):
    identity = IDENTITY.copy()
    identity["level"] = level

    with pytest.raises(
        OracleUnavailable, match="unsupported replay level"
    ):
        bound(row(0), identity=identity)
