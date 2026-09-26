"""Synthetic formatter and feature-recoverability tests.

The character tokenizer below tests truncation behavior only.
It does not establish parity with the historical model tokenizer.
"""

import ast
import pytest
import hashlib
import json
from pathlib import Path

from research_paper.stage1_state_coverage._synthetic_features import (
    TRUNCATION_MARKER,
    candidate_header,
    candidate_triggered_canaries,
    candidate_workforce_summary,
    candidate_learner_features,
    candidate_text_leak_asset_features,
    candidate_learner_leak_asset_features,
)

from models import (
    CanaryTrap,
    Department,
    DoubleAgentAsset,
    EnvironmentObservation,
    LeakEvent,
    Worker,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    Path(__file__).parent
    / "fixtures/stage1_state_coverage/feature_recoverability_v1.json"
)

FUNCTIONS = (
    "active_departments_from_observation",
    "format_observation",
    "compact_observation_lines",
    "truncate_observation_tokens",
)


def load_formatter():
    # Importing train_trl_v2 normally requires GPU/training libraries.
    # Compile only its existing formatting functions.
    source = (ROOT / "train_trl_v2.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in FUNCTIONS
    ]

    if {node.name for node in selected} != set(FUNCTIONS):
        raise AssertionError("Upstream formatter functions changed")

    namespace = {
        "Department": Department,
        "EnvironmentObservation": EnvironmentObservation,
    }
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    exec(compile(module, "train_trl_v2.py", "exec"), namespace)

    segments = [
        ast.get_source_segment(source, node)
        for node in selected
    ]
    digest = hashlib.sha256(
        "\n".join(segments).encode("utf-8")
    ).hexdigest()

    return namespace, digest


def synthetic_observation():
    return EnvironmentObservation(
        workers=[
            Worker(
                id="w-001",
                name="Synthetic A",
                department="engineering",
                state="suspected",
                suspicion_level=0.87,
            ),
            Worker(
                id="w-002",
                name="Synthetic B",
                department="finance",
                state="loyal",
                suspicion_level=0.0,
            ),
            Worker(
                id="w-003",
                name="Synthetic C",
                department="operations",
                state="loyal",
                suspicion_level=0.0,
            ),
        ],
        active_leaks=[
            LeakEvent(
                id="leak-001",
                channel="dark_web",
                department="engineering",
                is_canary=True,
            )
        ],
        canary_traps=[
            CanaryTrap(
                id=f"canary-{index:03d}",
                department=(
                    "engineering" if index % 2 else "operations"
                ),
                planted_turn=0,
                unique_hash=f"synthetic-{index}",
                triggered=index in (1, 2),
            )
            for index in range(1, 9)
        ],
        double_agents=[
            DoubleAgentAsset(
                worker_id="w-004",
                turned_turn=1,
            )
        ],
        turn=7,
        max_turns=150,
        phase="orientation",
        phase_number=1,
        enterprise_revenue=101.6,
        security_score=97.6,
    )


class CharacterTokenizer:
    """Deterministic synthetic tokenizer, not a model tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


def build_fixture():
    formatter, source_hash = load_formatter()
    observation = synthetic_observation()

    full = formatter["format_observation"](observation)
    compacted = formatter["compact_observation_lines"](full)
    truncated = formatter["truncate_observation_tokens"](
        CharacterTokenizer(), compacted, 96
    )

    assert TRUNCATION_MARKER in truncated

    return {
        "schema_version": "stage1-feature-parity-synthetic-v1",
        "synthetic": True,
        "formatter_functions_sha256": source_hash,
        "expected_header": {
            "turn": 7,
            "max_turns": 150,
            "phase": "orientation",
            "phase_number": 1,
            "displayed_revenue": 102,
            "displayed_security": 98,
        },
        "expected_triggered_canaries": 2,
        "text": {
            "original": full,
            "compacted": compacted,
            "character_token_truncated": truncated,
        },
    }






def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_versioned_fixture_matches_current_formatter():
    assert build_fixture() == fixture()


def test_original_header_parity():
    data = fixture()
    actual = candidate_header(data["text"]["original"])
    assert actual == {
        "status": "available",
        "features": data["expected_header"],
    }


def test_compacted_header_parity():
    data = fixture()
    assert candidate_header(
        data["text"]["compacted"]
    ) == candidate_header(data["text"]["original"])


def test_token_truncation_fails_closed():
    data = fixture()
    assert candidate_header(
        data["text"]["character_token_truncated"]
    ) == {
        "status": "unavailable",
        "reason": "token_truncated",
    }


def test_complete_canary_count():
    data = fixture()
    assert candidate_triggered_canaries(
        data["text"]["original"]
    ) == {
        "status": "available",
        "triggered_count": 2,
    }


def test_compacted_canary_count_unavailable():
    data = fixture()
    assert candidate_triggered_canaries(
        data["text"]["compacted"]
    ) == {
        "status": "unavailable",
        "reason": "incomplete_canary_rows",
    }


def test_compaction_removes_old_canaries_and_clean_workers():
    compacted = fixture()["text"]["compacted"]
    assert "clean loyal workers omitted: 2" in compacted
    assert "canary-001" not in compacted
    assert "canary-002" not in compacted
    assert "canary-008" in compacted





def test_original_workforce_parity():
    result = candidate_workforce_summary(fixture()["text"]["original"])
    assert result == {
        "status": "available",
        "features": {
            "worker_count": 3,
            "displayed_high_suspicion_count": 1,
            "unique_worker_departments": 3,
        },
    }


def test_compacted_workforce_has_partial_recoverability():
    result = candidate_workforce_summary(fixture()["text"]["compacted"])
    assert result["status"] == "available"
    assert result["features"] == {
        "worker_count": 3,
        "displayed_high_suspicion_count": 1,
    }
    assert result["unavailable_fields"] == [
        "unique_worker_departments"
    ]


def test_truncated_workforce_is_unavailable():
    assert candidate_workforce_summary(
        fixture()["text"]["character_token_truncated"]
    ) == {
        "status": "unavailable",
        "reason": "token_truncated",
    }


def test_missing_omission_summary_is_rejected():
    compacted = fixture()["text"]["compacted"]
    corrupted = compacted.replace(
        "  clean loyal workers omitted: 2", "", 1
    )
    assert candidate_workforce_summary(corrupted) == {
        "status": "unavailable",
        "reason": "incomplete_worker_rows",
    }


def test_displayed_suspicion_threshold_uses_rounded_value():
    formatter, _ = load_formatter()
    observation = synthetic_observation()
    observation.workers[0].suspicion_level = 0.504

    rendered = formatter["format_observation"](observation)
    assert "suspicion=50%" in rendered

    result = candidate_workforce_summary(rendered)
    assert result["features"]["displayed_high_suspicion_count"] == 0





def test_learner_and_original_training_text_match():
    learner = candidate_learner_features(
        synthetic_observation().model_dump()
    )
    text = fixture()["text"]["original"]

    assert learner["status"] == "available"
    assert learner["header"] == candidate_header(text)["features"]
    assert learner["workforce"] == (
        candidate_workforce_summary(text)["features"]
    )
    assert learner["triggered_canaries"] == (
        candidate_triggered_canaries(text)["triggered_count"]
    )


def test_learner_and_compacted_text_match_where_recoverable():
    learner = candidate_learner_features(
        synthetic_observation().model_dump()
    )
    text = fixture()["text"]["compacted"]

    assert learner["header"] == candidate_header(text)["features"]

    recovered = candidate_workforce_summary(text)
    assert recovered["status"] == "available"

    for name, value in recovered["features"].items():
        assert learner["workforce"][name] == value

    assert "unique_worker_departments" in (
        recovered["unavailable_fields"]
    )
    assert candidate_triggered_canaries(text)["status"] == "unavailable"


def test_learner_uses_displayed_suspicion_precision():
    observation = synthetic_observation()
    observation.workers[0].suspicion_level = 0.504

    learner = candidate_learner_features(
        observation.model_dump()
    )

    assert (
        learner["workforce"]["displayed_high_suspicion_count"]
        == 0
    )


@pytest.mark.parametrize(
    "missing",
    ["workers", "canary_traps", "enterprise_revenue", "turn"],
)
def test_missing_learner_field_fails_closed(missing):
    record = synthetic_observation().model_dump()
    del record[missing]

    assert candidate_learner_features(record)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("workers", "invalid"),
        ("enterprise_revenue", True),
        ("turn", False),
    ],
)
def test_malformed_learner_field_fails_closed(field, value):
    record = synthetic_observation().model_dump()
    record[field] = value

    assert candidate_learner_features(record)["status"] == "unavailable"


def test_hidden_worker_attributes_are_not_required():
    record = synthetic_observation().model_dump()

    hidden = (
        "hidden_state", "is_sleeper", "generation",
        "cover_integrity", "leak_cooldown", "activation_turn",
        "false_flag_target", "dead_switch_armed",
    )

    expected = candidate_learner_features(record)

    for worker in record["workers"]:
        for field in hidden:
            worker.pop(field, None)

    assert candidate_learner_features(record) == expected







@pytest.mark.parametrize("variant", ["original", "compacted"])
def test_leak_asset_parity_with_learner(variant):
    learner = candidate_learner_leak_asset_features(
        synthetic_observation().model_dump()
    )
    rendered = candidate_text_leak_asset_features(
        fixture()["text"][variant]
    )

    assert learner["status"] == "available"
    assert rendered == learner
    assert rendered["features"] == {
        "active_leak_count": 1,
        "displayed_canary_match_count": 1,
        "displayed_double_agent_count": 1,
        "operational_double_agent_count": 1,
    }


def test_truncated_leak_asset_features_unavailable():
    result = candidate_text_leak_asset_features(
        fixture()["text"]["character_token_truncated"]
    )
    assert result == {
        "status": "unavailable",
        "reason": "token_truncated",
    }


def test_missing_leak_row_is_rejected():
    text = fixture()["text"]["original"]
    corrupted = text.replace(
        "  leak-001 dept=engineering channel=dark_web"
        " [CANARY MATCH]\n",
        "",
        1,
    )

    assert corrupted != text
    assert candidate_text_leak_asset_features(
        corrupted
    )["status"] == "unavailable"


def test_duplicate_leak_row_is_rejected():
    text = fixture()["text"]["original"]
    leak_line = (
        "  leak-001 dept=engineering channel=dark_web"
        " [CANARY MATCH]\n"
    )
    corrupted = text.replace(leak_line, leak_line * 2, 1)

    assert corrupted != text
    assert candidate_text_leak_asset_features(
        corrupted
    )["status"] == "unavailable"


def test_inactive_agent_is_not_operational():
    formatter, _ = load_formatter()
    observation = synthetic_observation()
    observation.double_agents[0].active = False

    text_result = candidate_text_leak_asset_features(
        formatter["format_observation"](observation)
    )
    learner_result = candidate_learner_leak_asset_features(
        observation.model_dump()
    )

    assert text_result == learner_result
    assert text_result["features"]["displayed_double_agent_count"] == 1
    assert text_result["features"]["operational_double_agent_count"] == 0


@pytest.mark.parametrize("field", ["active_leaks", "double_agents"])
def test_missing_learner_leak_asset_field_rejected(field):
    record = synthetic_observation().model_dump()
    del record[field]

    assert candidate_learner_leak_asset_features(
        record
    )["status"] == "unavailable"


@pytest.mark.parametrize("field,bad_value", [
    ("active_leaks", "is_canary"),
    ("double_agents", "active"),
])
def test_malformed_learner_leak_asset_flag_rejected(
    field, bad_value
):
    record = synthetic_observation().model_dump()
    record[field][0][bad_value] = "true"

    assert candidate_learner_leak_asset_features(
        record
    )["status"] == "unavailable"



@pytest.mark.parametrize("bad", [None, 42, [], {}])
def test_canary_extractor_rejects_invalid_text(bad):
    assert candidate_triggered_canaries(bad) == {
        "status": "unavailable",
        "reason": "invalid_text",
    }


@pytest.mark.parametrize("bad", [None, 42, [], {}])
def test_workforce_extractor_rejects_invalid_text(bad):
    assert candidate_workforce_summary(bad) == {
        "status": "unavailable",
        "reason": "invalid_text",
    }


def test_duplicate_canary_heading_fails_closed():
    text = fixture()["text"]["original"]
    corrupted = text.replace(
        "Canary Traps (8):",
        "Canary Traps (8):\nCanary Traps (8):",
        1,
    )
    assert corrupted != text
    assert candidate_triggered_canaries(corrupted) == {
        "status": "unavailable",
        "reason": "duplicate_canary_section",
    }
