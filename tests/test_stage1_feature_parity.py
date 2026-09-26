"""Synthetic formatter and feature-recoverability tests.

The character tokenizer below tests truncation behavior only.
It does not establish parity with the historical model tokenizer.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

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
TRUNCATION_MARKER = "[... compacted for training context ...]"

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


def candidate_header(text):
    if not isinstance(text, str):
        return {"status": "unavailable", "reason": "invalid_text"}

    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    first_line = text.splitlines()[0] if text else ""
    match = re.fullmatch(
        r"Turn (\d+)/(\d+) \| Phase: (.+?) \((\d+)\)"
        r" \| Revenue: (-?\d+) \| Security: (-?\d+)",
        first_line,
    )

    if match is None:
        return {"status": "unavailable", "reason": "missing_header"}

    turn, maximum, phase, number, revenue, security = (
        match.groups()
    )
    return {
        "status": "available",
        "features": {
            "turn": int(turn),
            "max_turns": int(maximum),
            "phase": phase,
            "phase_number": int(number),
            "displayed_revenue": int(revenue),
            "displayed_security": int(security),
        },
    }


def candidate_triggered_canaries(text):
    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    match = re.search(
        r"^Canary Traps \((\d+)\):$",
        text,
        re.MULTILINE,
    )
    if match is None:
        return {
            "status": "unavailable",
            "reason": "missing_canary_section",
        }

    declared_count = int(match.group(1))
    rows = re.findall(
        r"^  canary-\d+ dept=\S+ triggered=(True|False)$",
        text,
        re.MULTILINE,
    )

    if len(rows) != declared_count:
        return {
            "status": "unavailable",
            "reason": "incomplete_canary_rows",
        }

    return {
        "status": "available",
        "triggered_count": rows.count("True"),
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



def candidate_workforce_summary(text):
    """Conservative synthetic candidate; not the approved extractor."""
    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    lines = text.splitlines()
    workers = [
        (i, re.fullmatch(r"Workers \((\d+)\):", line))
        for i, line in enumerate(lines)
        if re.fullmatch(r"Workers \((\d+)\):", line)
    ]
    leaks = [
        i for i, line in enumerate(lines)
        if re.fullmatch(r"Active Leaks \(\d+\):", line)
    ]

    if len(workers) != 1 or len(leaks) != 1:
        return {"status": "unavailable", "reason": "missing_sections"}

    start, heading = workers[0]
    end = leaks[0]

    if end <= start:
        return {"status": "unavailable", "reason": "section_order"}

    declared = int(heading.group(1))
    worker_lines = lines[start + 1:end]

    omitted_matches = re.findall(
        r"^  clean loyal workers omitted: (\d+)$",
        text,
        re.MULTILINE,
    )
    if len(omitted_matches) > 1:
        return {"status": "unavailable", "reason": "invalid_omissions"}

    omitted = int(omitted_matches[0]) if omitted_matches else 0

    if worker_lines == ["  (none)"] and declared == 0:
        worker_lines = []

    pattern = re.compile(
        r"^  w-[0-9]+ .+? dept=(\S+) state=\S+ "
        r"(?:clean|suspicion=(\d+)%)(?: turning=\d+/4)?$"
    )
    parsed = [pattern.fullmatch(line) for line in worker_lines]

    if any(match is None for match in parsed):
        return {"status": "unavailable", "reason": "invalid_worker_rows"}

    if len(parsed) + omitted != declared:
        return {"status": "unavailable", "reason": "incomplete_worker_rows"}

    features = {
        "worker_count": declared,
        "displayed_high_suspicion_count": sum(
            match.group(2) is not None
            and int(match.group(2)) > 50
            for match in parsed
        ),
    }

    result = {"status": "available", "features": features}

    if omitted:
        result["unavailable_fields"] = ["unique_worker_departments"]
    else:
        features["unique_worker_departments"] = len({
            match.group(1) for match in parsed
        })

    return result


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
