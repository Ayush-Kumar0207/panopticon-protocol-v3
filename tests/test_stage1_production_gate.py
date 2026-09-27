"""Production JSONL regression through both synthetic gates."""

import ast
import hashlib
import runpy

import pytest
from pathlib import Path

from research_paper.stage1_state_coverage.persisted_text import (
    MODEL_ID,
    MODEL_REVISION,
    CHAT_TEMPLATE_SHA256,
)
from research_paper.stage1_state_coverage.synthetic_mapping import (
    ROW_MAP_VERSION,
    LEDGER_VERSION,
    verify_synthetic_row_mapping,
)

from research_paper.stage1_state_coverage.production_identity import (
    PINNED_PIPELINE,
)
from research_paper.stage1_state_coverage.provenance import (
    validate_training_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / (
    "tests/fixtures/stage1_state_coverage/"
    "production_pinned_two_turn_v1.jsonl"
)
EXPECTED_HASH = (
    "fc92a78607dde094b32c84a5ba68a2db0ac7cc8411179b368495d6861d7d22e5"
)
SOURCE_COMMIT = "2fd1652560352652d0e29f9fadc2b725c77f44f9"


def _helpers():
    return runpy.run_path(
        str(ROOT / "tests/test_stage1_synthetic_gate.py")
    )


def _case(tmp_path):
    h = _helpers()
    case = h["_two_turn_case"](tmp_path)

    content = FIXTURE.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_HASH
    lines = content.split(b"\n")
    assert lines[-1] == b""
    lines.pop()
    assert len(lines) == 3
    assert lines[0] == lines[1]
    assert lines[1] != lines[2]

    for row in (case["prior"], case["current"]):
        observation = row["observation_before"]
        observation["workers"] = observation["workers"][:1]
        for field in (
            "active_leaks",
            "canary_traps",
            "intel_reports",
            "double_agents",
        ):
            observation[field] = []
        row["source_commit"] = SOURCE_COMMIT

    case["header"]["source_commit"] = SOURCE_COMMIT

    manifest = case["manifest"]
    manifest["source_commit"] = SOURCE_COMMIT
    manifest["formatter"]["source_commit"] = SOURCE_COMMIT
    manifest["production_pipeline"] = dict(PINNED_PIPELINE)
    manifest["tokenizer"] = {
        "identifier": MODEL_ID,
        "revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    manifest["transforms"] = {
        "compaction": "none",
        "token_truncation": "none",
    }

    dataset = manifest["dataset_files"][0]
    (case["root"] / dataset["path"]).write_bytes(content)
    dataset["sha256"] = hashlib.sha256(content).hexdigest()
    dataset["bytes"] = len(content)
    dataset["rows"] = 3

    witnesses = [
        {
            **case["map_row"],
            "row_index": index,
            "turn": 0 if index < 2 else 1,
            "row_sha256": hashlib.sha256(line).hexdigest(),
        }
        for index, line in enumerate(lines)
    ]

    manifest["mapping"]["mapped_rows"] = 3

    for key, version in (
        ("mapping_file", ROW_MAP_VERSION),
        ("independent_evidence", LEDGER_VERSION),
    ):
        entry = manifest["mapping"][key]
        data = h["_encode"]({
            "schema_version": version,
            "synthetic": True,
            "rows": witnesses,
        })
        (case["root"] / entry["path"]).write_bytes(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["bytes"] = len(data)

    case["expected"] = h["_expectations"](manifest)
    return h, case


def test_production_bytes_pass_both_gates(tmp_path):
    h, case = _case(tmp_path)

    mapping = verify_synthetic_row_mapping(
        case["manifest"], case["expected"], case["root"]
    )
    assert mapping["status"] == "unverified"
    assert mapping["checked_row_occurrences"] == 3
    assert mapping["weighted_duplicate_occurrences"] == 1

    for index in (0, 1):
        first = h["_run"](
            case, row_index=index, current_row=case["prior"]
        )
        assert first["status"] == "unverified", first
        assert first["feature_count"] == 14
        assert first["representation"] == "pinned_qwen_chat"

    second = h["_run_two_turn_gate"](
        case, row_index=2
    )
    assert second["status"] == "unverified", second
    assert second["checked_turns"] == 2
    assert second["feature_count"] == 14
    assert second["representation"] == "pinned_qwen_chat"


def test_raw_text_rejected_under_production_identity(tmp_path):
    h = _helpers()
    case = h["_case"](tmp_path)
    case["manifest"]["tokenizer"] = {
        "identifier": MODEL_ID,
        "revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    case["manifest"]["production_pipeline"] = dict(PINNED_PIPELINE)
    case["manifest"]["source_commit"] = SOURCE_COMMIT
    case["manifest"]["formatter"]["source_commit"] = SOURCE_COMMIT
    case["header"]["source_commit"] = SOURCE_COMMIT
    case["current"]["source_commit"] = SOURCE_COMMIT
    case["expected"] = h["_expectations"](case["manifest"])

    result = h["_run"](case)
    assert result["status"] == "unavailable"
    assert result["reason"] == (
        "training_text_representation_unavailable"
    )
    assert result["detail"]["reason"] == (
        "invalid_template_boundaries"
    )


def test_pinned_source_matches_checkout():
    source = (ROOT / "train_trl_v2.py").read_text(
        encoding="utf-8-sig"
    ).replace("\r\n", "\n")

    assert hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest() == PINNED_PIPELINE["source_sha256"]

    tree = ast.parse(source)

    for name, expected_field in (
        ("format_observation", "formatter_sha256"),
        ("render_training_text", "renderer_sha256"),
        ("save_training_data_with_template", "writer_sha256"),
    ):
        matches = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ]
        assert len(matches) == 1, name

        function_source = ast.get_source_segment(
            source, matches[0]
        )

        assert hashlib.sha256(
            function_source.encode("utf-8")
        ).hexdigest() == PINNED_PIPELINE[expected_field], name


def test_production_manifest_requires_pipeline(tmp_path):
    _, case = _case(tmp_path)
    del case["manifest"]["production_pipeline"]

    result = validate_training_manifest(
        case["manifest"], case["expected"]
    )
    assert result["status"] == "rejected"
    assert any(
        "production_pipeline_required" in reason
        for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "formatter_sha256",
        "renderer_sha256",
        "writer_sha256",
    ],
)
def test_production_pipeline_drift_rejected(tmp_path, field):
    _, case = _case(tmp_path)
    case["manifest"]["production_pipeline"][field] = "f" * 64

    result = validate_training_manifest(
        case["manifest"], case["expected"]
    )
    assert result["status"] == "rejected"
    assert any(
        "unsupported_production_pipeline_identity" in reason
        for reason in result["reasons"]
    )


def test_production_source_commit_mismatch_rejected(tmp_path):
    _, case = _case(tmp_path)
    case["manifest"]["formatter"]["source_commit"] = "3" * 40

    result = validate_training_manifest(
        case["manifest"], case["expected"]
    )
    assert result["status"] == "rejected"
    assert any(
        "inconsistent_production_pipeline_source" in reason
        for reason in result["reasons"]
    )


def test_pipeline_cannot_be_claimed_by_raw_synthetic_fixture(tmp_path):
    helpers = _helpers()
    case = helpers["_case"](tmp_path)
    case["manifest"]["production_pipeline"] = dict(PINNED_PIPELINE)

    result = validate_training_manifest(
        case["manifest"], case["expected"]
    )
    assert result["status"] == "rejected"
