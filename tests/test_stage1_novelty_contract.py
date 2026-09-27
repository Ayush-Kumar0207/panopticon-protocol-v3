"""Synthetic tests for the proposed Stage 1 novelty contract."""

import math
import pytest

from research_paper.stage1_state_coverage.novelty import (
    calculate_distance,
    calibrate_expert_threshold,
    score_learner_novelty,
    score_learner_episode_novelty,
)


def reference(seed, value, level="level_4"):
    return {
        "episode_id": f"ep{seed}",
        "seed": seed,
        "level": level,
        "features": {"val": float(value)},
    }


def test_pairwise_clipping_preserves_local_difference():
    scaler = {"val": {"median": 0.0, "iqr": 1.0}}

    assert calculate_distance(
        {"val": 100.0}, {"val": 98.0}, scaler
    ) == 2.0

    assert calculate_distance(
        {"val": 100.0}, {"val": 90.0}, scaler
    ) == 5.0


def test_numeric_categorical_average():
    scaler = {"val": {"median": 0.0, "iqr": 1.0}}

    assert calculate_distance(
        {"val": 100.0, "phase": "b"},
        {"val": 98.0, "phase": "a"},
        scaler,
    ) == 1.5


def test_query_seed_excludes_alias_episode():
    refs = [reference(i, 0.0) for i in range(6)]
    alias = reference(9, 10.0)
    alias["episode_id"] = "another-name"
    refs.append(alias)

    result = score_learner_novelty(
        {"val": 10.0},
        refs,
        "query-episode",
        expected_level="level_4",
        query_seed=9,
    )

    assert result["status"] == "available"
    assert result["independent_reference_episodes"] == 6
    assert result["novelty_score"] == 5.0


def test_invalid_query_seed_fails_closed():
    result = score_learner_novelty(
        {"val": 1.0},
        [reference(i, 0.0) for i in range(6)],
        "query",
        expected_level="level_4",
        query_seed=True,
    )
    assert result["reason"] == "invalid_query_seed"


def test_episode_median_and_strict_threshold():
    refs = [reference(i, 0.0) for i in range(5)]
    turns = [{"val": 0.0}, {"val": 9.0}, {"val": 12.0}]

    result = score_learner_episode_novelty(
        turns, refs, "query",
        query_seed=42,
        expected_level="level_4",
        threshold=4.0,
    )

    assert result["status"] == "available"
    assert result["turn_scores"] == [0.0, 5.0, 5.0]
    assert result["episode_novelty_score"] == 5.0
    assert result["aggregation"] == "median"
    assert result["low_coverage"] is True

    at_threshold = score_learner_episode_novelty(
        turns, refs, "query",
        query_seed=42,
        expected_level="level_4",
        threshold=5.0,
    )
    assert at_threshold["low_coverage"] is False


def test_invalid_threshold_fails_closed():
    result = score_learner_episode_novelty(
        [{"val": 1.0}],
        [reference(i, 0.0) for i in range(5)],
        "query",
        query_seed=42,
        expected_level="level_4",
        threshold=math.nan,
    )
    assert result["reason"] == "invalid_threshold"


def test_other_levels_cannot_supply_missing_neighbors():
    refs = [
        reference(i, 0.0) for i in range(4)
    ] + [
        reference(i + 10, 0.0, "level_3")
        for i in range(10)
    ]

    result = score_learner_episode_novelty(
        [{"val": 1.0}], refs, "query",
        query_seed=42,
        expected_level="level_4",
    )
    assert result["reason"] == "insufficient_reference_episodes"


def test_calibration_uses_same_episode_median():
    refs = [
        reference(i, i + j / 10)
        for i in range(6)
        for j in range(3)
    ]

    calibration = calibrate_expert_threshold(
        refs, expected_level="level_4"
    )

    assert calibration["status"] == "available"
    assert calibration["threshold_unit"] == "episode_median"
    assert calibration["threshold_comparator"] == ">"

    manual = []
    for seed in range(6):
        held_out = [
            row["features"]
            for row in refs if row["seed"] == seed
        ]
        training = [
            row for row in refs if row["seed"] != seed
        ]

        result = score_learner_episode_novelty(
            held_out, training, f"ep{seed}",
            query_seed=seed,
            expected_level="level_4",
        )

        assert result["status"] == "available"
        manual.append(result["episode_novelty_score"])

    assert calibration["held_out_episode_scores"] == pytest.approx(manual)



@pytest.mark.parametrize("operation", ["turn", "calibration"])
@pytest.mark.parametrize("mixed", [False, True])
def test_levelled_references_require_explicit_level(
    operation, mixed
):
    refs = [
        reference(
            i, i,
            "level_5" if mixed and i % 2 else "level_4"
        )
        for i in range(6)
    ]

    if operation == "turn":
        result = score_learner_novelty(
            {"val": 3.0}, refs, "query"
        )
    else:
        result = calibrate_expert_threshold(refs)

    assert result == {
        "status": "unavailable",
        "reason": "missing_expected_level",
    }


@pytest.mark.parametrize("operation", ["turn", "calibration"])
@pytest.mark.parametrize(
    "bad_level",
    ["", "unknown", "level_3", True, " level_4 "],
)
def test_invalid_expected_level_rejected(operation, bad_level):
    refs = [reference(i, i) for i in range(6)]

    if operation == "turn":
        result = score_learner_novelty(
            {"val": 3.0}, refs, "query",
            expected_level=bad_level,
        )
    else:
        result = calibrate_expert_threshold(
            refs, expected_level=bad_level
        )

    assert result == {
        "status": "unavailable",
        "reason": "invalid_expected_level",
    }
